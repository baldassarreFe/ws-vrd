from __future__ import annotations
from collections import OrderedDict

import torch
from torch import nn as nn
from torch_geometric.utils import scatter_


class InputNodeModel(nn.Module):
    def __init__(
            self,
            in_conv_shape=None,
            in_linear_features=None,
            conv_channels=0,
            conv_layers=0,
            fc_features=None,
            fc_layers=0,
            combined_fc_features=None,
            combined_fc_layers=0,
    ):
        """Input node model without message passing, transforms 1D/3D node features into a 1D vector.

        Args:
            in_conv_shape:
            in_linear_features:
            conv_channels:
            conv_layers:
            fc_features:
            fc_layers:
            combined_fc_features:
            combined_fc_layers:
        """
        super(InputNodeModel, self).__init__()

        if in_conv_shape is None:
            in_conv_shape = (0, 0, 0)
        if in_linear_features is None:
            in_linear_features = 0

        in_channels = in_conv_shape[0]
        in_features = in_linear_features

        convs = OrderedDict()
        for i in range(conv_layers):
            convs[f'conv{i}'] = nn.Conv2d(
                in_channels,
                conv_channels,
                kernel_size=3,
                padding=1,
            )
            convs[f'relu{i}'] = nn.ReLU()
            in_channels = conv_channels
        self.convs = nn.Sequential(convs)

        linear_fcs = OrderedDict()
        for i in range(fc_layers):
            linear_fcs[f'linear{i}'] = nn.Linear(in_features, fc_features)
            linear_fcs[f'relu{i}'] = nn.ReLU()
            in_features = fc_features
        self.linear_fcs = nn.Sequential(linear_fcs)

        # Flatten conv features and concatenate with linear features
        in_features = (in_channels * in_conv_shape[1] * in_conv_shape[2]) + in_features

        combined_fcs = OrderedDict()
        for i in range(combined_fc_layers):
            combined_fcs[f'linear{i}'] = nn.Linear(in_features, combined_fc_features)
            combined_fcs[f'relu{i}'] = nn.ReLU()
            in_features = fc_features
        self.combined_fcs = nn.Sequential(combined_fcs)

        self.out_features = in_features

    def forward(self, conv_features: torch.Tensor, linear_features: torch.Tensor) -> torch.Tensor:
        conv_features = self.convs(conv_features)
        conv_features = torch.flatten(conv_features, start_dim=1)

        linear_features = self.linear_fcs(linear_features)

        combined = torch.cat([conv_features, linear_features], dim=1)
        combined = self.combined_fcs(combined)

        return combined


class InputEdgeModel(nn.Module):
    def __init__(
            self,
            in_linear_features=None,
            fc_features=None,
            fc_layers=0,
    ):
        super(InputEdgeModel, self).__init__()

        if in_linear_features is None:
            in_linear_features = 0

        in_features = in_linear_features

        linear_fcs = OrderedDict()
        for i in range(fc_layers):
            linear_fcs[f'linear{i}'] = nn.Linear(in_features, fc_features)
            linear_fcs[f'relu{i}'] = nn.ReLU()
            in_features = fc_features
        self.linear_fcs = nn.Sequential(linear_fcs)

        self.out_features = in_features

    def forward(self, linear_features: torch.Tensor) -> torch.Tensor:
        out = self.linear_fcs(linear_features)
        return out


# class InputEdgeModel(nn.Module):
#     def __init__(
#             self,
#             in_conv_shape=None,
#             in_linear_features=None,
#             conv_channels=None,
#             conv_layers=0,
#             fc_features=None,
#             fc_layers=0,
#             combined_fc_features=None,
#             combined_fc_layers=0,
#     ):
#         """Input edge model without message passing, transforms 1D/3D edge features into a 1D vector.
#
#         Args:
#             in_conv_shape:
#             in_linear_features:
#             conv_channels:
#             conv_layers:
#             fc_features:
#             fc_layers:
#             combined_fc_features:
#             combined_fc_layers:
#         """
#         super(InputEdgeModel, self).__init__()
#
#         if in_conv_shape is None:
#             in_conv_shape = (0, 0, 0)
#         if in_linear_features is None:
#             in_linear_features = 0
#
#         in_channels = in_conv_shape[0]
#         in_features = in_linear_features
#
#         convs = OrderedDict()
#         for i in range(conv_layers):
#             convs[f'conv{i}'] = nn.Conv2d(
#                 in_channels,
#                 conv_channels,
#                 kernel_size=3,
#                 padding=1,
#             )
#             convs[f'relu{i}'] = nn.ReLU()
#             in_channels = conv_channels
#         self.convs = nn.Sequential(convs)
#
#         linear_fcs = OrderedDict()
#         for i in range(fc_layers):
#             linear_fcs[f'linear{i}'] = nn.Linear(in_features, fc_features)
#             linear_fcs[f'relu{i}'] = nn.ReLU()
#             in_features = fc_features
#         self.linear_fcs = nn.Sequential(linear_fcs)
#
#         # Flatten conv features and concatenate with linear features
#         in_features = (in_channels * in_conv_shape[1] * in_conv_shape[2]) + in_features
#
#         combined_fcs = OrderedDict()
#         for i in range(combined_fc_layers):
#             combined_fcs[f'linear{i}'] = nn.Linear(in_features, combined_fc_features)
#             combined_fcs[f'relu{i}'] = nn.ReLU()
#             in_features = fc_features
#         self.combined_fcs = nn.Sequential(combined_fcs)
#
#         self.out_features = in_features
#
#     def forward(self, conv_features: torch.Tensor, linear_features: torch.Tensor) -> torch.Tensor:
#         conv_features = self.convs(conv_features)
#         conv_features = torch.flatten(conv_features, start_dim=1)
#
#         linear_features = self.linear_fcs(linear_features)
#
#         combined = torch.cat([conv_features, linear_features], dim=1)
#         combined = self.combined_fcs(combined)
#
#         return combined


class EdgeModel(torch.nn.Module):
    def __init__(
            self,
            in_node_features,
            in_edge_features,
            fc_features,
            fc_layers
    ):
        super(EdgeModel, self).__init__()
        in_features = in_node_features + in_edge_features + in_node_features

        fcs = OrderedDict()
        for i in range(fc_layers):
            fcs[f'linear{i}'] = nn.Linear(in_features, fc_features)
            fcs[f'relu{i}'] = nn.ReLU()
            in_features = fc_features
        self.fcs = nn.Sequential(fcs)

        self.out_features = in_features

    def forward(self, nodes, edges, edge_indices):
        senders = nodes[edge_indices[0]]
        receivers = nodes[edge_indices[1]]

        out = torch.cat([senders, edges, receivers], dim=1)
        out = self.fcs(out)
        return out

    def old_forward(self, senders, receivers, edges, globals, edge_to_graph_idx):
        # senders, receivers: [E, in_node_channels, H, W]
        # edges: [E, in_edge_size]
        # globals: [B, in_global_size]
        # edge_to_graph_idx: [E] with max entry B - 1
        # where E is the number of edges, B is the number of graphs

        edges = edges[:, :, None, None]
        out = torch.cat([senders, receivers, edges, globals[edge_to_graph_idx]], dim=1)
        out = self.conv(out)
        out = self.relu(out)
        return out


class OutputGlobalModel(nn.Module):
    def __init__(self, in_edge_features, fc_features, fc_layers, num_classes):
        super(OutputGlobalModel, self).__init__()
        in_features = in_edge_features

        fcs = OrderedDict()
        for i in range(fc_layers):
            fcs[f'linear{i}'] = nn.Linear(in_features, fc_features)
            fcs[f'relu{i}'] = nn.ReLU()
            in_features = fc_features
        fcs['output'] = nn.Linear(in_features, num_classes)
        self.fcs = nn.Sequential(fcs)

        self.out_features = num_classes

    def forward(
            self,
            edges: torch.Tensor,
            edge_indices: torch.LongTensor,
            node_to_graph_idx: torch.LongTensor,
            num_graphs: int
    ) -> torch.Tensor:
        edge_to_graph_idx = node_to_graph_idx[edge_indices[0]]
        out = scatter_('max', edges, index=edge_to_graph_idx, dim=0, dim_size=num_graphs)
        out = self.fcs(out)
        return out
