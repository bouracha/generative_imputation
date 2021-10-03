from models.layers import *

import numpy as np

import models.utils as utils

class GraphVDEncoder(nn.Module):
    def __init__(self, input_n=[96, 10], act_fn=nn.GELU(), device="cuda", batch_norm=False, p_dropout=0.0):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GraphVDEncoder, self).__init__()
        self.activation = act_fn
        self.device = device
        self.batch_norm = batch_norm
        self.p_dropout = p_dropout
        self.node_n, self.features = input_n[0], input_n[1]

        # input_n -> input_n corresponding to z_bottom -> .... -> N_{z_0} corresponding to z_top
        #self.level_output_sizes = [[self.node_n, self.features], [self.node_n, 128]]
        self.level_output_sizes = [[self.node_n, self.features], [self.node_n, 128], [24, 128], [8, 128], [1, 256]]
        #self.level_output_sizes = [[self.node_n, self.features], [self.node_n, 64], [8, 128], [1, 256]]
        #self.level_output_sizes = [[self.node_n, self.features], [self.node_n, 256], [48, 256], [32, 256], [16, 256], [8, 256], [4, 256], [2, 256], [1, 256]]
        print(self.level_output_sizes)

        #Bottom Up
        self.graphconv_blocks = []
        self.graphconv_reductions = []
        self.graphconv_residual_reductions = []
        self.rezeros = []
        for i in range(len(self.level_output_sizes)-1):
            in_graph_size = self.level_output_sizes[i][0]
            in_feature_size = self.level_output_sizes[i][1]
            out_graph_size = self.level_output_sizes[i+1][0]
            out_feature_size = self.level_output_sizes[i+1][1]
            self.graphconv_reductions.append(GraphConvolution(in_feature_size, out_feature_size, bias=True, node_n=in_graph_size, out_node_n=out_graph_size))
            self.graphconv_blocks.append(GC_Block(out_feature_size, p_dropout, bias=True, node_n=out_graph_size, activation=nn.GELU()))
            self.graphconv_residual_reductions.append(GraphConvolution(in_feature_size, out_feature_size, bias=True, node_n=in_graph_size, out_node_n=out_graph_size))
            self.rezeros.append(ReZero())
        self.graphconv_blocks = nn.ModuleList(self.graphconv_blocks)
        self.graphconv_reductions = nn.ModuleList(self.graphconv_reductions)
        self.graphconv_residual_reductions = nn.ModuleList(self.graphconv_residual_reductions)
        self.rezeros = nn.ModuleList(self.rezeros)

    def forward(self, x):
        #Bottom Up
        self.activations = []
        y = x
        for i in range(len(self.level_output_sizes)-1):
            conv_y = self.graphconv_reductions[i](y)
            encoder_output = self.graphconv_blocks[i](conv_y)

            self.activations.append(encoder_output)

            res = self.graphconv_residual_reductions[i](y)
            y = res + self.rezeros[i](encoder_output)

        return self.activations