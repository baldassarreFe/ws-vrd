from typing import Dict

import torch
from torch_geometric.data import Batch

from ..utils import scatter_topk_2d_flat


class VisualRelationExplainer(torch.nn.Module):
    # Gradients will be computed w.r.t. these input fields
    GRAD_WRT = (
        "object_linear_features",
        "object_conv_features",
        "relation_linear_features",
    )

    def __init__(
        self,
        model: torch.nn.Module,
        top_k_predicates: int = 10,
        top_x_relations: int = 100,
        activations: bool = True,
        channel_mean: bool = True,
        relevance_fn: str = "relu_sum",
        object_scores: bool = True,
    ):
        """

        Args:
            model:
            top_k_predicates: For each graph, try to explain only the TOP_K_PREDICATES
            top_x_relations: For each graph and predicate to explain, keep only
                             the TOP_X_RELATIONS (subject, predicate, object) as explanations
            activations: whether to use gradients only or (activations * gradients)
            channel_mean: if using activations, whether to aggregate 2D gradients along H and W
                          before multiplying with their tensor
            relevance_fn: aggregate node/edge relevance using [relu_sum, l1, l2]
            object_scores: use subject/object scores when ranking relations
        """
        super().__init__()

        self.model = model
        self.top_k_predicates = top_k_predicates
        self.top_x_relations = top_x_relations

        if not activations and channel_mean:
            raise ValueError(
                "`channel_mean` has no effect if `activations` are not used"
            )
        self.activations = activations
        self.cam = channel_mean
        self.relevance_fn = relevance_fn

        self.object_scores = object_scores

    @staticmethod
    def _zero_grad_(tensor: torch.Tensor):
        if tensor.grad is not None:
            tensor.grad.detach_()
            tensor.grad.zero_()
        return tensor

    def forward(self, inputs: Batch) -> Dict[str, Batch]:
        with torch.enable_grad():
            return self._explain_visual_relations(inputs)

    def _explain_visual_relations(self, inputs: Batch) -> Dict[str, Batch]:
        # Prepare input graphs
        inputs.apply(torch.Tensor.requires_grad_, *VisualRelationExplainer.GRAD_WRT)
        edge_to_graph_assignment = inputs.batch[inputs.relation_indexes[0]]

        # B = number of graphs in the batch
        # N = total number of nodes = N1 + N2 + N3 + ...
        # E = total number of edges = E1 + E2 + E3 + ...
        B = inputs.num_graphs
        N = inputs.num_nodes
        E = inputs.n_edges.sum().item()

        # Prepare relevance tensors
        # relevance_nodes[n, k] = relevance of node n
        # w.r.t. k-th top predicate predicted for the graph that n belongs to
        # [N, TOP_K_PREDICATES]
        relevance_nodes = inputs.object_linear_features.new_zeros(
            size=(N, self.top_k_predicates)
        )

        # relevance_edges[e, k] = relevance of edge e
        # w.r.t. k-th top predicate predicted for the graph that e belongs to
        # [E, TOP_K_PREDICATES]
        relevance_edges = inputs.relation_linear_features.new_zeros(
            size=(E, self.top_k_predicates)
        )

        with torch.enable_grad():
            # Forward pass to get predicate predictions
            outputs = self.model(inputs)

            # Sort predicate predictions per graph and iterate through each one of the TOP_K_PREDICATES predictions
            # [B, TOP_K_PREDICATES]
            predicate_scores_sorted, predicate_classes_sorted = torch.topk(
                outputs.predicate_scores.sigmoid(), self.top_k_predicates, dim=1
            )

            for k, k_th_predicate_score in enumerate(
                predicate_scores_sorted.unbind(dim=1)
            ):
                still_needed = k != self.top_k_predicates - 1

                # Propagate gradient of prediction to outputs, use L1 norm of the gradient as relevance
                inputs.apply(
                    VisualRelationExplainer._zero_grad_,
                    *VisualRelationExplainer.GRAD_WRT
                )
                k_th_predicate_score.backward(
                    torch.ones_like(k_th_predicate_score), retain_graph=still_needed
                )

                # Keep track of the gradient w.r.t. the k-th top scoring class in these two matrices
                relevance_nodes[:, k] = torch.add(
                    _relevance_fn(
                        _relevance_1d(inputs.object_linear_features, self.activations),
                        self.relevance_fn,
                    ),
                    _relevance_fn(
                        _relevance_2d(
                            inputs.object_conv_features, self.activations, self.cam
                        ),
                        self.relevance_fn,
                    ),
                )
                relevance_edges[:, k] = _relevance_fn(
                    _relevance_1d(inputs.relation_linear_features, self.activations),
                    self.relevance_fn,
                )

            predicate_scores_sorted.detach_()

        # Each (subject, predicate, object) triplet receives a relevance score proportional to
        # - relevance of the subject
        # - relevance of the object
        # - relevance of the edge that connects them,
        # - the score of the predicate that was used to compute those relevances
        # - optionally, the confidence score given to the subject from the object detector
        # - optionally, the confidence score given to the object from the object detector
        # [E x TOP_K_PREDICATES]
        relation_scores = (
            relevance_nodes[inputs.relation_indexes[0]]
            * relevance_edges
            * relevance_nodes[inputs.relation_indexes[1]]
            * predicate_scores_sorted.detach()[edge_to_graph_assignment]
        )

        if self.object_scores:
            relation_scores = (
                relation_scores
                * inputs.object_scores[inputs.relation_indexes[0], None]
                * inputs.object_scores[inputs.relation_indexes[1], None]
            )

        relations = self._keep_top_x_relations(
            B,
            edge_to_graph_assignment,
            inputs,
            predicate_classes_sorted,
            predicate_scores_sorted,
            relation_scores,
        )

        return relations

    def _keep_top_x_relations(
        self,
        B,
        edge_to_graph_assignment,
        inputs,
        predicate_classes_sorted,
        predicate_scores_sorted,
        relation_scores,
    ):
        # For each graph, retain the TOP_X_RELATIONS relations
        (
            relation_scores_sorted,  # [B x TOP_X_RELATIONS]
            (relation_indexes_index_sorted, _),  # [B * TOP_X_RELATIONS]
            (_, predicate_scores_index_sorted),  # [TOP_X_RELATIONS]
        ) = scatter_topk_2d_flat(
            relation_scores,
            edge_to_graph_assignment,
            self.top_x_relations,
            dim_size=B,
            fill_value=float("-inf"),
        )

        # Final number of relations per graph, could be less than TOP_X_RELATIONS if the
        # graph had less than (TOP_X_RELATIONS // TOP_K_PREDICATES) from the start
        n_relations = (relation_indexes_index_sorted != -1).int().sum(dim=1)

        # Skip locations where relation_indexes_sorted = -1, i.e. there were fewer than TOP_X_RELATIONS to rank.
        # [n_relations.sum()]
        relation_scores_sorted = relation_scores_sorted.flatten()[
            relation_indexes_index_sorted.flatten() != -1
        ]

        # Index into relation_indexes to retrieve subj and obj for the top x scoring relations per graph.
        # Skip locations where relation_indexes_sorted = -1, i.e. there were fewer than TOP_X_RELATIONS to rank.
        # [2, n_relations.sum()]
        relation_indexes_index_sorted = relation_indexes_index_sorted.flatten()
        relation_indexes_index_sorted = relation_indexes_index_sorted[
            relation_indexes_index_sorted != -1
        ]
        relation_indexes_sorted = inputs.relation_indexes[
            :, relation_indexes_index_sorted
        ]

        # Index into predicate_scores_sorted and predicate_classes_sorted
        # to retrieve the top x scoring relations per graph.
        # Skip locations where predicate_scores_index_sorted = -1, i.e. there were fewer than TOP_X_RELATIONS to rank.
        # When applying gather turn -1 into 0, otherwise cuda complains, but then remove the gathered values.
        # [n_relations.sum()]
        predicate_scores_sorted = predicate_scores_sorted.gather(
            dim=1, index=predicate_scores_index_sorted.clamp(min=0)
        )
        predicate_scores_sorted = predicate_scores_sorted.flatten()[
            predicate_scores_index_sorted.flatten() != -1
        ]
        predicate_classes_sorted = predicate_classes_sorted.gather(
            dim=1, index=predicate_scores_index_sorted.clamp(min=0)
        )
        predicate_classes_sorted = predicate_classes_sorted.flatten()[
            predicate_scores_index_sorted.flatten() != -1
        ]

        relations = Batch(
            num_nodes=inputs.num_nodes,
            n_edges=n_relations,
            relation_scores=relation_scores_sorted,
            relation_indexes=relation_indexes_sorted,
            predicate_scores=predicate_scores_sorted,
            predicate_classes=predicate_classes_sorted,
            **{
                k: inputs[k]
                for k in (
                    "n_nodes",
                    "batch",
                    "object_boxes",
                    "object_classes",
                    "object_image_size",
                )
            }
        )
        return relations


def _relevance_1d(x: torch.Tensor, activations: bool):
    # x: [B, F]
    if not activations:
        relevance = x.grad
    else:
        relevance = x * x.grad
    return relevance


def _relevance_2d(x: torch.Tensor, activations: bool, cam: bool):
    # x: [B, C, H, W]
    if not activations:
        relevance = x.grad
    else:
        if not cam:
            relevance = x * x.grad
        else:
            # [B, C]
            channel_rel = x.grad.flatten(start_dim=2).mean(dim=2)
            relevance = x * channel_rel[:, :, None, None]
    return relevance


def _relevance_fn(relevance, pool: str):
    relevance = relevance.flatten(start_dim=1)
    if pool == "relu_sum":
        return relevance.relu().sum(dim=1)
    elif pool == "l1":
        return relevance.abs().sum(dim=1)
    elif pool == "l2":
        return relevance.pow(2).sum(dim=1)
