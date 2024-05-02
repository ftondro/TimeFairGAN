""" Options

This script is largely based on junyanz/pytorch-CycleGAN-and-pix2pix.

Returns:
    [argparse]: Class containing argparse
"""

import argparse
import os
import torch


class Options():
    """Options class

    Returns:
        [argparse]: argparse containing train and test options
    """

    def __init__(self):
        # Inputs for the train function
        self.parser = argparse.ArgumentParser(description="Script Description")
        self.subparser = self.parser.add_subparsers(dest='command')
        self.with_fairness = self.subparser.add_parser('with_fairness')
        self.no_fairness = self.subparser.add_parser('no_fairness')
        # with_fairness
        self.with_fairness.add_argument('--df_name', type=str, help='Dataframe name', default='maintenance')
        # self.with_fairness.add_argument('--num_epochs', type=int, help='Number of training epochs', default=50)
        self.with_fairness.add_argument('--batch_size', type=int, help='The batch size', default= 128)
        self.with_fairness.add_argument('--seq_len', type=int, help='The sequence length', default= 24)
        self.with_fairness.add_argument('--z_dim', help='Z or data dimension', default=8, type=int)
        self.with_fairness.add_argument('--module', type=str, help='gru, lstm, or lstmLN', default='gru')
        self.with_fairness.add_argument('--hidden_dim', type=int, help='The hidden dimensions', default= 24)
        self.with_fairness.add_argument('--num_layer', type=int, help='Number of layers', default= 3)
        self.with_fairness.add_argument('--iteration', type=int, help='Number of training iterations', default=50000)
        self.with_fairness.add_argument('--S', type=str, help='Protected attribute', default='Color')
        self.with_fairness.add_argument('--Y', type=str, help='Label (decision)', default='Maintenance_Required')
        self.with_fairness.add_argument('--underprivileged_value', type=str, help='Value for underprivileged group', default='white')
        self.with_fairness.add_argument('--desirable_value', type=str, help='Desired label (decision)', default='Yes')
        self.with_fairness.add_argument('--lamda_val', type=float, help='Lambda hyperparameter', default=0.5)
        self.with_fairness.add_argument('--metric_iteration', type=int, help='Number of iterations for metric computation', default= 10)
        self.with_fairness.add_argument('--fake_name', type=str, help='Name of the produced csv file', default='Synthetic_maintenance')
        self.with_fairness.add_argument('--size_of_fake_data', type=int, help='How many data records to generate', default= 10000)
        # Add
        self.with_fairness.add_argument('--workers', type=int, help='number of data loading workers', default=8)
        self.with_fairness.add_argument('--device', type=str, default='cpu', help='Device: gpu | cpu')
        self.with_fairness.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.with_fairness.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
        self.with_fairness.add_argument('--model', type=str, default='TimeGAN', help='chooses which model to use timegan')
        self.with_fairness.add_argument('--outf', default='./output', help='folder to output images and model checkpoints')
        self.with_fairness.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        self.with_fairness.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        self.with_fairness.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.with_fairness.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        self.with_fairness.add_argument('--display', action='store_true', help='Use visdom.')
        self.with_fairness.add_argument('--manualseed', default=-1, type=int, help='manual seed')
        # Train
        self.with_fairness.add_argument('--print_freq', type=int, default=1000, help='frequency of showing training results on console')
        self.with_fairness.add_argument('--load_weights', action='store_true', help='Load the pretrained weights')
        self.with_fairness.add_argument('--resume', default='', help="path to checkpoints (to continue training)")
        self.with_fairness.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        self.with_fairness.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
        self.with_fairness.add_argument('--w_gamma', type=float, default=1, help='Gamma weight')
        self.with_fairness.add_argument('--w_es', type=float, default=0.1, help='Encoder loss weight')
        self.with_fairness.add_argument('--w_e0', type=float, default=10, help='Encoder loss weight')
        self.with_fairness.add_argument('--w_g', type=float, default=100, help='Generator loss weight')
        self.with_fairness.add_argument('--w_g_f', type=float, default=4, help='Generator fair loss weight') 

        # no_fairness
        self.no_fairness.add_argument('--df_name', type=str, help='Dataframe name', default='maintenance')
        # self.no_fairness.add_argument('--num_epochs', type=int, help='Number of training epochs', default=50)
        self.no_fairness.add_argument('--batch_size', type=int, help='The batch size', default= 128)
        self.no_fairness.add_argument('--seq_len', type=int, help='The sequence length', default= 24)
        self.no_fairness.add_argument('--z_dim', help='Z or data dimension', default=8, type=int)
        self.no_fairness.add_argument('--module', type=str, help='gru, lstm, or lstmLN', default='gru')
        self.no_fairness.add_argument('--hidden_dim', type=int, help='The hidden dimensions', default= 24)
        self.no_fairness.add_argument('--num_layer', type=int, help='Number of layers', default= 3)
        self.no_fairness.add_argument('--iteration', type=int, help='Number of training iterations', default=50000)
        self.no_fairness.add_argument('--metric_iteration', type=int, help='Number of iterations for metric computation', default= 10)
        self.no_fairness.add_argument('--fake_name', type=str, help='Name of the produced csv file', default='Synthetic_maintenace')
        self.no_fairness.add_argument('--size_of_fake_data', type=int, help='How many data records to generate', default= 10000)
        # Add
        self.no_fairness.add_argument('--workers', type=int, help='number of data loading workers', default=8)
        self.no_fairness.add_argument('--device', type=str, default='cpu', help='Device: gpu | cpu')
        self.no_fairness.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.no_fairness.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
        self.no_fairness.add_argument('--model', type=str, default='TimeGAN', help='chooses which model to use. timegan')
        self.no_fairness.add_argument('--outf', default='./output', help='folder to output images and model checkpoints')
        self.no_fairness.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        self.no_fairness.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        self.no_fairness.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.no_fairness.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        self.no_fairness.add_argument('--display', action='store_true', help='Use visdom.')
        self.no_fairness.add_argument('--manualseed', default=-1, type=int, help='manual seed')
        # Train
        self.no_fairness.add_argument('--print_freq', type=int, default=1000, help='frequency of showing training results on console')
        self.no_fairness.add_argument('--load_weights', action='store_true', help='Load the pretrained weights')
        self.no_fairness.add_argument('--resume', default='', help="path to checkpoints (to continue training)")
        self.no_fairness.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        self.no_fairness.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
        self.no_fairness.add_argument('--w_gamma', type=float, default=1, help='Gamma weight')
        self.no_fairness.add_argument('--w_es', type=float, default=0.1, help='Encoder loss weight')
        self.no_fairness.add_argument('--w_e0', type=float, default=10, help='Encoder loss weight')
        self.no_fairness.add_argument('--w_g', type=float, default=100, help='Generator loss weight')
        self.isTrain = True
        self.opt = None

    def set_z_dim_value(self, num):
        self.z_dim =  num 

    def parse(self):
        """ Parse Arguments.
        """

        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if self.opt.device == 'gpu':
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        # save to the disk
        if self.opt.name == 'experiment_name':
            self.opt.name = "%s/%s" % (self.opt.model, self.opt.df_name)
        expr_dir = os.path.join(self.opt.outf, self.opt.name)

        if not os.path.isdir(expr_dir):
            os.makedirs(expr_dir)

        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt