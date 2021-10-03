import os
import argparse
from pprint import pprint


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================

        # ===============================================================
        #                     Architecture options
        # ===============================================================
        self.parser.add_argument('--use_dct', dest='use_dct', action='store_true', help='toggle to transform from temporal to frequency space')
        self.parser.set_defaults(use_dct=True)
        self.parser.add_argument('--variational', dest='variational', action='store_true', help='toggle VAE or AE')
        self.parser.set_defaults(variational=False)
        self.parser.add_argument('--output_variance', dest='output_variance', action='store_true', help='toggle model output variance or use as constant')
        self.parser.set_defaults(output_variance=False)
        self.parser.add_argument('--batch_norm', dest='batch_norm', action='store_true', help='toggle use batch_norm or not')
        self.parser.set_defaults(batch_norm=False)
        self.parser.add_argument('--p_drop', type=float, default=0.0, help='dropout rate')
        self.parser.add_argument('--l2_reg', type=float, default=1e-4, help='dropout rate')
        # ===============================================================
        #                    Initialise options
        # ===============================================================
        self.parser.add_argument('--start_epoch', type=int, default=1, help='If not 1, load checkpoint at this epoch')
        self.parser.add_argument('--name', type=str, default="model_1", help='Name of master folder containing model')
        # ===============================================================
        #                     Running options
        # ===============================================================
        self.parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        self.parser.add_argument('--beta', type=float, default=1.0, help='Downweighting of the KL divergence')
        self.parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs to train for')
        self.parser.add_argument('--train_batch_size', type=int, default=256, help='Number of epochs to train for')
        self.parser.add_argument('--test_batch_size', type=int, default=256, help='If not 1, load checkpoint at this epoch')
        self.parser.add_argument('--warmup_time', type=int, default=0, help='number of epochs to warm up the KL')
        self.parser.add_argument('--beta_final', type=float, default=1.0, help='Final downweighting of the KL divergence')

        # ===============================================================
        #                     Experiments
        # ===============================================================

        # ===============================================================
        #                     Experiment options
        # ===============================================================


    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()
        self._print()
        return self.opt