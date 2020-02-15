from enum import Enum
from typing import Union

import torch
from torch_geometric.data import Batch

from ..utils import scatter_topk_2d_flat


class VisualRelationExplainer(torch.nn.Module):
    # Gradients will be computed w.r.t. these input fields
    GRAD_WRT = ('object_linear_features', 'object_conv_features', 'relation_linear_features')

    class ReadoutMode(Enum):
        FC_MAX = 0
        MAX_FC = 1

    def __init__(
            self,
            model: torch.nn.Module,
            mode: Union[str, ReadoutMode],
            top_k_predicates: int = 10,
            top_x_relations: int = 100,
            **__
    ):
        """

        Args:
            model:
            mode:
            top_k_predicates: For each graph, try to explain only the TOP_K_PREDICATES
            top_x_relations: For each graph and predicate to explain, keep only
                             the TOP_X_RELATIONS (subject, predicate, object) as explanations
        """
        super().__init__()

        if isinstance(mode, str):
            mode = VisualRelationExplainer.ReadoutMode[mode]
        if not isinstance(mode, VisualRelationExplainer.ReadoutMode):
            raise ValueError(f'Invalid explanation mode: {mode}')

        self.mode = mode
        self.model = model
        self.top_k_predicates = top_k_predicates
        self.top_x_relations = top_x_relations

    @staticmethod
    def _zero_grad_(tensor: torch.Tensor):
        if tensor.grad is not None:
            tensor.grad.detach_()
            tensor.grad.zero_()
        return tensor

    def forward(self, inputs: Batch) -> Batch:
        with torch.enable_grad():
            if self.mode is VisualRelationExplainer.ReadoutMode.FC_MAX:
                return self._explain_fc_max(inputs)
            elif self.mode is VisualRelationExplainer.ReadoutMode.MAX_FC:
                return self.explain_max_fc(inputs)
            raise ValueError(f'Invalid explanation mode: {self.mode}')

    def _explain_fc_max(self, inputs: Batch) -> Batch:
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
        relevance_nodes = inputs.object_linear_features.new_zeros(size=(N, self.top_k_predicates))

        # relevance_edges[e, k] = relevance of edge e
        # w.r.t. k-th top predicate predicted for the graph that e belongs to
        # [E, TOP_K_PREDICATES]
        relevance_edges = inputs.relation_linear_features.new_zeros(size=(E, self.top_k_predicates))

        with torch.enable_grad():

            # Forward pass to get predicate predictions
            outputs = self.model(inputs)

            # Sort predicate predictions per graph and iterate through each one of the TOP_K_PREDICATES predictions
            # [B, TOP_K_PREDICATES]
            predicate_scores_sorted, predicate_classes_sorted = torch.topk(outputs.predicate_scores.sigmoid(),
                                                                           self.top_k_predicates, dim=1)

            for k, k_th_predicate_score in enumerate(predicate_scores_sorted.unbind(dim=1)):
                still_needed = k != self.top_k_predicates - 1

                # Propagate gradient of prediction to outputs, use L1 norm of the gradient as relevance
                inputs.apply(VisualRelationExplainer._zero_grad_, *VisualRelationExplainer.GRAD_WRT)
                k_th_predicate_score.backward(torch.ones_like(k_th_predicate_score), retain_graph=still_needed)

                # Keep track of the gradient w.r.t. the k-th top scoring class in these two matrices
                relevance_nodes[:, k] = (
                        inputs.object_linear_features.grad.norm(dim=1, p=1) +
                        inputs.object_conv_features.grad.flatten(start_dim=1).norm(dim=1, p=1)
                )
                relevance_edges[:, k] = inputs.relation_linear_features.grad.norm(dim=1, p=1)

            predicate_scores_sorted.detach_()

        # Each (subject, predicate, object) triplet receives a relevance score proportional to
        # the relevance of the subject, the object, the edge that connects them,
        # and the score of the predicate that was used to compute those relevances.
        # [E x TOP_K_PREDICATES]
        relation_scores = (
                relevance_nodes[inputs.relation_indexes[0]] *
                relevance_edges *
                relevance_nodes[inputs.relation_indexes[1]] *
                predicate_scores_sorted.detach()[edge_to_graph_assignment]
        )

        # For each graph, retain the TOP_X_RELATIONS relations
        (
            relation_scores_sorted,  # [B x TOP_X_RELATIONS] 
            (relation_indexes_index_sorted, _),  # [B * TOP_X_RELATIONS]
            (_, predicate_scores_index_sorted)  # [TOP_X_RELATIONS]
        ) = scatter_topk_2d_flat(relation_scores, edge_to_graph_assignment, self.top_x_relations,
                                 dim_size=B, fill_value=float('-inf'))

        # Final number of relations per graph, could be less than TOP_X_RELATIONS if the
        # graph had less than (TOP_X_RELATIONS // TOP_K_PREDICATES) from the start
        n_relations = (relation_indexes_index_sorted != -1).int().sum(dim=1)

        # Skip locations where relation_indexes_sorted = -1, i.e. there were fewer than TOP_X_RELATIONS to rank.
        # [n_relations.sum()]
        relation_scores_sorted = relation_scores_sorted.flatten()[relation_indexes_index_sorted.flatten() != -1]

        # Index into relation_indexes to retrieve subj and obj for the top x scoring relations per graph.
        # Skip locations where relation_indexes_sorted = -1, i.e. there were fewer than TOP_X_RELATIONS to rank.
        # [2, n_relations.sum()]
        relation_indexes_index_sorted = relation_indexes_index_sorted.flatten()
        relation_indexes_index_sorted = relation_indexes_index_sorted[relation_indexes_index_sorted != -1]
        relation_indexes_sorted = inputs.relation_indexes[:, relation_indexes_index_sorted]

        # Index into predicate_scores_sorted and predicate_classes_sorted
        # to retrieve the top x scoring relations per graph.
        # Skip locations where predicate_scores_index_sorted = -1, i.e. there were fewer than TOP_X_RELATIONS to rank.
        # When applying gather turn -1 into 0, otherwise cuda complains, but then remove the gathered values.
        # [n_relations.sum()]
        predicate_scores_sorted = predicate_scores_sorted.gather(
            dim=1, index=predicate_scores_index_sorted.clamp(min=0))
        predicate_scores_sorted = predicate_scores_sorted.flatten()[predicate_scores_index_sorted.flatten() != -1]
        predicate_classes_sorted = predicate_classes_sorted.gather(
            dim=1, index=predicate_scores_index_sorted.clamp(min=0))
        predicate_classes_sorted = predicate_classes_sorted.flatten()[predicate_scores_index_sorted.flatten() != -1]

        relations = Batch(
            num_nodes=inputs.num_nodes,
            n_edges=n_relations,
            relation_scores=relation_scores_sorted,
            relation_indexes=relation_indexes_sorted,
            predicate_scores=predicate_scores_sorted,
            predicate_classes=predicate_classes_sorted,
            **{k: inputs[k] for k in ('n_nodes', 'batch', 'object_boxes', 'object_classes', 'object_image_size')}
        )

        return relations
