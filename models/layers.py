import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
import numpy as np

class ParametiseInputs(nn.Module):
    def __init__(self, inputs):
        """
        ParametiseInputs make masked inputs parameters
        :param self:
        :return:
        """
        super(ParametiseInputs, self).__init__()
        #self.inputs_as_parameters = Parameter(name="inputs_as_params", param=torch.FloatTensor(inputs.shape))
        self.inputs_as_parameters = Parameter(torch.randn(inputs.shape))

    def forward(self, inputs, mask_occluded):
        inputs_with_parameters = (1.0 - mask_occluded)*inputs + mask_occluded*self.inputs_as_parameters
        return inputs_with_parameters

class ReZero(nn.Module):
    def __init__(self):
        """
        ReZero layer, learnable weight for residuals
        :param self:
        :return:
        """
        super(ReZero, self).__init__()
        self.resweight = Parameter(torch.Tensor([0.0]))

    def forward(self, input):
        output = torch.mul(input, self.resweight)
        return output

class FullyConnected(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        """
        Fully connected layer of learnable weights with learnable bias
        :param self:
        :param in_features: number neurons in
        :param out_features: num neurons out
        :param bias: to use bias (boole)
        :return:
        """
        super(FullyConnected, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class NeuralNetworkBlock(nn.Module):
    def __init__(self, layers=[48, 100, 50, 2], activation=nn.LeakyReLU(0.1), batch_norm=False, p_dropout=0.0):
        """
        :param layers: array where each entry is number of neurons at that layer
        :param activation: activation function to use
        :param batch_norm: to use batch norm (boole).
        :param p_dropout: drop out prob.
        """
        super(NeuralNetworkBlock, self).__init__()

        self.n_x = layers[0]
        self.n_z = layers[-1]
        self.layers = np.array(layers)
        self.n_layers = self.layers.shape[0] - 1

        self.fc_blocks = []
        for i in range(self.n_layers):
            self.fc_blocks.append(FC_Block(self.layers[i], self.layers[i + 1], activation=activation, batch_norm=batch_norm, p_dropout=p_dropout, bias=True))
        self.fc_blocks = nn.ModuleList(self.fc_blocks)

    def forward(self, x):
        y = x
        for i in range(self.n_layers):
            y = self.fc_blocks[i](y)

        return y

class GraphGaussianBlock(nn.Module):
    def __init__(self, in_nodes, in_features, n_z_nodes, n_z_features):
        """
        :param input_feature: num of input feature
        :param n_z: dim of distribution
        """
        super(GraphGaussianBlock, self).__init__()
        self.in_features = in_features
        self.in_nodes = in_nodes
        self.n_z_features = n_z_features
        self.n_z_nodes = n_z_nodes

        self.z_mu_graphconv = GraphConvolution(in_features, n_z_features, bias=True, node_n=in_nodes, out_node_n=n_z_nodes) #FullyConnected(self.n_x, self.n_z)
        self.z_log_var_graphconv = GraphConvolution(in_features, n_z_features, bias=True, node_n=in_nodes, out_node_n=n_z_nodes)

    def forward(self, x):
        y = x

        mu = self.z_mu_graphconv(y)
        log_var = self.z_log_var_graphconv(y)
        log_var = torch.clamp(log_var, min=-20.0, max=3.0)

        return mu, log_var

class GaussianBlock(nn.Module):
    def __init__(self, in_features, n_z):
        """
        :param input_feature: num of input feature
        :param n_z: dim of distribution
        """
        super(GaussianBlock, self).__init__()
        self.n_x = in_features
        self.n_z = n_z

        self.z_mu_fc = FullyConnected(self.n_x, self.n_z)
        self.z_log_var_fc = FullyConnected(self.n_x, self.n_z)

    def forward(self, x):
        y = x

        mu = self.z_mu_fc(y)
        log_var = self.z_log_var_fc(y)
        log_var = torch.clamp(log_var, min=-20.0, max=3.0)

        return mu, log_var

class GraphLayer(nn.Module):

    def __init__(self, node_n=48, out_node_n=None):
        super(GraphConvolution, self).__init__()
        if out_node_n is None:
            out_node_n = node_n
        self.att = Parameter(torch.FloatTensor(out_node_n, node_n))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.matmul(self.att, input)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class Graph_Block(nn.Module):
    def __init__(self, p_dropout, node_n=48, activation=nn.GELU()):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()

        self.g1 = GraphLayer(node_n=node_n)
        self.g2 = GraphLayer(node_n=node_n)
        self.rezero = ReZero()

        self.do = nn.Dropout(p_dropout)
        self.act_f = activation

    def forward(self, x):
        y = self.g1(x)
        y = self.act_f(y)
        y = self.do(y)

        y = self.g2(y)
        y = self.act_f(y)
        y = self.do(y)

        return x + self.rezero(y)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True, node_n=48, out_node_n=None):
        super(GraphConvolution, self).__init__()
        if out_node_n is None:
            out_node_n = node_n
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(out_node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48, activation=nn.GELU()):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.rezero = ReZero()

        self.do = nn.Dropout(p_dropout)
        self.act_f = activation

    def forward(self, x):
        y = self.gc1(x)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        y = self.act_f(y)
        y = self.do(y)

        return x + self.rezero(y)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class FC_Block(nn.Module):
    def __init__(self, in_features, out_features, activation=nn.LeakyReLU(0.1), batch_norm=False, p_dropout=0.0, bias=True):
        """
        Define a fully connected block
        """
        super(FC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act_f = activation
        self.batch_norm = batch_norm
        self.p_dropout = p_dropout
        self.do = nn.Dropout(p_dropout)

        self.fc = FullyConnected(self.in_features, self.out_features, bias=bias)
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(out_features)


    def forward(self, x):
        y = self.fc(x)
        y = self.act_f(y)
        if self.batch_norm:
            b, f = y.shape
            y = self.bn(y.view(b, -1)).view(b, f)
        y = self.do(y)
        return y

    def __repr__(self):
        representation = self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')' \
               + ', dropout: '+str(self.p_dropout)
        if self.batch_norm:
            representation = representation+', batch norm'
        representation = representation+', act_fn: {0}'.format(self.act_f)
        return representation