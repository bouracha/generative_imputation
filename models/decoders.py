import torch.nn as nn
import torch

from models.layers import *

import numpy as np

import models.utils as utils

class GraphVDDecoder(nn.Module):
    def __init__(self, input_n=[96, 10], encoder_activation_sizes=[[96, 8], [24, 64], [8, 128], [1, 256]], act_fn=nn.GELU(), device="cuda", batch_norm=False, p_dropout=0.0, residual_size=256):
        """

        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GraphVDDecoder, self).__init__()
        self.activation = act_fn
        self.device = device
        self.batch_norm = batch_norm
        self.p_dropout = p_dropout
        self.node_input_n = encoder_activation_sizes[0][0]
        self.input_temp_n = encoder_activation_sizes[0][1]
        self.residual_size = residual_size
        self.encoder_activation_sizes = encoder_activation_sizes

        self.z_mus = {}
        self.z_log_vars = {}
        self.z_posterior_mus = {}
        self.z_posterior_log_vars = {}
        self.z_prior_mus = {}
        self.z_prior_log_vars = {}
        self.KLs = {}
        self.zs = {}
        self.residuals_dict = {}

        #Top Down
        n_z_0_nodes = encoder_activation_sizes[-1][0]
        n_z_0_features = encoder_activation_sizes[-1][1]
        print("n_z_0_features: ", n_z_0_features)
        self.reparametisation_latent_0 = GraphGaussianBlock(in_nodes=n_z_0_nodes, in_features=n_z_0_features, n_z_nodes=n_z_0_nodes, n_z_features=n_z_0_features)
        self.resize_conv_0 = GraphConvolution(in_features=n_z_0_features, out_features=self.residual_size, bias=True, node_n=n_z_0_nodes, out_node_n=self.node_input_n)
        #self.reshape_z0_linearly = FullyConnected(in_features=self.feature_hierachies[-1], out_features=self.residual_size, bias=True)

        self.decoder_units = []
        for i in range(len(self.encoder_activation_sizes)-2):
            rezero1 = ReZero()
            rezero2 = ReZero()
            rezero3 = ReZero()
            in_node_size = encoder_activation_sizes[-1-i][0]
            in_feature_size = encoder_activation_sizes[-1-i][1]
            out_node_size = encoder_activation_sizes[-2-i][0]
            out_feature_size = encoder_activation_sizes[-2-i][1]
            begin_decoder_block = GC_Block(self.residual_size, p_dropout, bias=True, node_n=self.node_input_n, activation=nn.GELU())

            posterior_decoder_resize = GraphConvolution(in_features=self.residual_size, out_features=out_feature_size, bias=True, node_n=self.node_input_n, out_node_n=out_node_size)
            posterior_decoder_block = GC_Block((out_feature_size+out_feature_size), p_dropout, bias=True, node_n=out_node_size, activation=nn.GELU())
            reparametisation_posterior = GraphGaussianBlock(in_nodes=out_node_size, in_features=(out_feature_size+out_feature_size), n_z_nodes=out_node_size, n_z_features=out_feature_size)

            prior_decoder_block = GC_Block(self.residual_size, p_dropout, bias=True, node_n=self.node_input_n, activation=nn.GELU())
            reparametisation_prior = GraphGaussianBlock(in_nodes=self.node_input_n, in_features=self.residual_size, n_z_nodes=out_node_size, n_z_features=out_feature_size)

            reshape_z = GraphConvolution(in_features=out_feature_size, out_features=self.residual_size, bias=True, node_n=out_node_size, out_node_n=self.node_input_n)

            self.decoder_units.append({
                "decoder_block":begin_decoder_block,
                "posterior_decoder_resize":posterior_decoder_resize,
                "posterior_decoder_block":posterior_decoder_block,
                "reparametisation_posterior":reparametisation_posterior,
                "prior_decoder_block":prior_decoder_block,
                "reparametisation_prior":reparametisation_prior,
                "reshape_z":reshape_z,
                "rezero1":rezero1,
                "rezero2":rezero2,
                "rezero3":rezero3
            })
            self.decoder_units[i] = nn.ModuleDict(self.decoder_units[i])
        self.decoder_units = nn.ModuleList(self.decoder_units)

        self.decoder_block_final = GC_Block(self.residual_size, p_dropout, bias=True, node_n=self.node_input_n, activation=nn.GELU())
        self.reparametisation_output = GraphGaussianBlock(in_nodes=self.node_input_n, in_features=self.residual_size, n_z_nodes=self.node_input_n, n_z_features=self.input_temp_n)

    def forward(self, encoder_activations=None, latent_resolution=999, z_0=None, one_hot_labels=None):
        if z_0 is None:
            self.z_posterior_mus["0"], self.z_posterior_log_vars["0"] = self.reparametisation_latent_0(encoder_activations[-1])
            self.z_prior_mus["0"], self.z_prior_log_vars["0"] = torch.zeros_like(self.z_posterior_mus["0"]), torch.zeros_like(self.z_posterior_log_vars["0"])
            self.z_mus["0"], self.z_log_vars["0"] = self.z_posterior_mus["0"], self.z_posterior_log_vars["0"]

            self.KLs["0"] = utils.kullback_leibler_divergence(self.z_mus["0"], self.z_log_vars["0"], mu_2=self.z_prior_mus["0"], log_var_2=self.z_prior_log_vars["0"], graph=True)
            self.zs["0"] = utils.reparametisation_trick(self.z_mus["0"], self.z_log_vars["0"], self.device)
        else:
            self.zs["0"] = z_0

        if one_hot_labels is not None:
            self.zs["0"] = torch.cat((self.zs["0"], one_hot_labels), dim=2)

        self.residuals_dict["0"] = self.resize_conv_0(self.zs["0"])

        for i in range(len(self.encoder_activation_sizes)-2):
            self.KLs[str(i+1)], self.zs[str(i+1)] = self.top_down_decode(level=i, encoder_activation=encoder_activations[-2-i], X_supplied=True, latent_resolution=latent_resolution)

        decoder_output_final = self.decoder_block_final(self.residuals_dict[str(len(self.encoder_activation_sizes)-2)])
        self.reconstructions_mu, self.reconstructions_log_var = self.reparametisation_output(decoder_output_final)

        sig = nn.Sigmoid()
        self.bernoulli_output = sig(self.reconstructions_mu)

        return self.reconstructions_mu, self.reconstructions_log_var, self.zs, self.KLs

    def top_down_decode(self, level, encoder_activation=None, X_supplied=True, latent_resolution=999):
        res_block_output = self.decoder_units[level]["decoder_block"](self.residuals_dict[str(level)])
        self.residuals_dict[str(level+1)] = self.residuals_dict[str(level)] + self.decoder_units[level]["rezero1"](res_block_output)

        #Posterior Route
        if X_supplied==True:
            posterior_decoder_resize_output = self.decoder_units[level]["posterior_decoder_resize"](self.residuals_dict[str(level+1)])
            concat_for_posterior = torch.cat((posterior_decoder_resize_output, encoder_activation), dim=2)
            posterior_decoder_block_output = self.decoder_units[level]["posterior_decoder_block"](concat_for_posterior)
            self.z_posterior_mus[str(level + 1)], self.z_posterior_log_vars[str(level + 1)] = self.decoder_units[level]["reparametisation_posterior"](posterior_decoder_block_output)

        #Prior route
        prior_decoder_block_output = self.decoder_units[level]["prior_decoder_block"](self.residuals_dict[str(level+1)])
        self.z_prior_mus[str(level + 1)], self.z_prior_log_vars[str(level + 1)] = self.decoder_units[level]["reparametisation_prior"](prior_decoder_block_output)
        self.residuals_dict[str(level+1)] = self.residuals_dict[str(level+1)] + self.decoder_units[level]["rezero2"](prior_decoder_block_output)

        #Sample from the posterior while training, prior while sampling
        if X_supplied==True:
            self.z_mus[str(level + 1)] = self.z_posterior_mus[str(level+1)]
            self.z_log_vars[str(level + 1)] = self.z_posterior_log_vars[str(level+1)]
        else:
            self.z_mus[str(level + 1)] = self.z_prior_mus[str(level+1)]
            self.z_log_vars[str(level + 1)] = self.z_prior_log_vars[str(level+1)]

        #Sample z_level, or take the mean
        self.KLs[str(level + 1)] = utils.kullback_leibler_divergence(self.z_mus[str(level + 1)], self.z_log_vars[str(level + 1)], mu_2=self.z_prior_mus[str(level + 1)], log_var_2=self.z_prior_log_vars[str(level + 1)], graph=True)
        if level < latent_resolution:
            self.zs[str(level+1)] = utils.reparametisation_trick(self.z_mus[str(level+1)], self.z_log_vars[str(level+1)], self.device)
        else:
            self.zs[str(level + 1)] = self.z_mus[str(level+1)]

        reshaped_z = self.decoder_units[level]["reshape_z"](self.zs[str(level+1)])
        self.residuals_dict[str(level+1)] = self.residuals_dict[str(level+1)] + self.decoder_units[level]["rezero3"](reshaped_z)

        return self.KLs[str(level+1)], self.zs[str(level+1)]
