import torch
import torch.nn as nn
from torch_geometric.data import Batch

from .common import InputNodeModel, InputEdgeModel, EdgeModel, OutputGlobalModel


class GNN(nn.Module):
    def __init__(
            self,
            layers=1,
            in_channels=256,
            in_edge_size=8,
    ):
        pass


class RelationalNetwork(nn.Module):
    def __init__(
            self,
            input_node_model: InputNodeModel,
            input_edge_model: InputEdgeModel,
            edge_model: EdgeModel,
            output_global_model: OutputGlobalModel
    ):
        super(RelationalNetwork, self).__init__()
        self.input_node_model = input_node_model
        self.input_edge_model = input_edge_model
        self.edge_model = edge_model
        self.output_global_model = output_global_model

    def forward(self, inputs: Batch) -> Batch:
        # Input
        node_features = self.input_node_model(
            linear_features=inputs.object_linear_features,
            conv_features=inputs.object_conv_features
        )
        edge_features = self.input_edge_model(
            linear_features=inputs.relation_linear_features
        )

        # Message passing
        edge_features = self.edge_model(
            nodes=node_features,
            edges=edge_features,
            edge_indices=inputs.relation_indexes
        )

        # Readout
        global_features = self.output_global_model(
            edges=edge_features,
            edge_indices=inputs.relation_indexes,
            node_to_graph_idx=inputs.batch,
            num_graphs=inputs.num_graphs
        )

        # Build output batch so that it can be split back into graphs using Batch.to_data_list()
        keys_to_copy = (
            'n_edges',
            'n_nodes',
            'object_boxes', 'object_classes',
            'relation_indexes', 'object_image_size'
        )
        outputs = Batch(
            num_nodes=inputs.num_nodes,
            batch=inputs.batch,
            predicate_scores=global_features,
            **{k: inputs[k] for k in keys_to_copy}
        )
        outputs.__slices__ = {
            'predicate_scores': inputs.__slices__['relation_indexes'],
            **{k: inputs.__slices__[k] for k in keys_to_copy}
        }

        return outputs


def build_relational_network(conf):
    input_node_model = InputNodeModel(**conf.input_node_model)
    input_edge_model = InputEdgeModel(**conf.input_edge_model)

    conf.edge_model.in_node_features = input_node_model.out_features
    conf.edge_model.in_edge_features = input_edge_model.out_features
    edge_model = EdgeModel(**conf.edge_model)

    conf.output_global_model.in_edge_features = edge_model.out_features
    output_global_model = OutputGlobalModel(**conf.output_global_model)

    return RelationalNetwork(
        input_node_model,
        input_edge_model,
        edge_model,
        output_global_model,
    )
