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

    def forward(self, graphs: Batch) -> torch.Tensor:
        node_features = self.input_node_model(
            linear_features=graphs.node_linear_features,
            conv_features=graphs.node_conv_features
        )
        edge_features = self.input_edge_model(
            linear_features=graphs.edge_attr
        )

        edge_features = self.edge_model(
            nodes=node_features,
            edges=edge_features,
            edge_indices=graphs.edge_index
        )

        global_features = self.output_global_model(
            edges=edge_features,
            edge_indices=graphs.edge_index,
            node_to_graph_idx=graphs.batch,
            num_graphs=graphs.num_graphs
        )

        return global_features


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
