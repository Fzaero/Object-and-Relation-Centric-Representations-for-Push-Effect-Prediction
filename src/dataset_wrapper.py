import torch
from .articulated_multi_object import *


class DatasetWrapper:
    def __init__(self, **params):
        self.device = params['device']  
        self.dataset_path = params['path']  
        self.num_of_traj = params['num_of_traj']
        self.no_angle = params['no_angle']
        self.my_data = My_Data(num_of_traj=self.num_of_traj, PATH=self.dataset_path, no_angle=self.no_angle)
        self.data_split = params['data_split']
        if 'scaler' in params:
            self.scaler = params['scaler']
            self.my_data.scaler = self.scaler
            self.scaler = self.my_data.split_data(
                self.data_split[0], self.data_split[1], self.data_split[2], self.device, self.scaler)
        else:
            self.scaler = self.my_data.split_data(
                self.data_split[0], self.data_split[1], self.data_split[2], self.device)

        net = params['Network']  # "net"

        self.gp_br = Graph_Processer_Belief_Regulator(
            self.scaler, net, self.device)   
        self.gp_pp = Graph_Processer_Physic_Predictor(
            self.scaler, net, self.device, torch.nn.L1Loss())
        val_len=50
        if "val_len" in params:
            val_len = params['val_len']
        self.val_tester = ValidatorAndTester(
            self.my_data, self.gp_pp, self.gp_br, self.device,val_len)

        batch_size_train = params['batch_size_train']
        batch_size_val = params['batch_size_val']

        num_of_batch_train = params['num_of_batch_train']
        num_of_batch_val = params['num_of_batch_val']

        shuffle = params['shuffle']

        self.my_dataset_train = My_Dataset(self.my_data)
        self.my_dataset_train.setup_dataloaders(
            num_of_batch_train[0], num_of_batch_train[1], batch_size=batch_size_train,  num_workers=8)

        self.my_dataset_val = My_Dataset(self.my_data, True)
        self.my_dataset_val.setup_dataloaders(
            num_of_batch_val[0], num_of_batch_val[1], batch_size=batch_size_val, num_workers=1)


class NetWrapper:
    def __init__(self, **params):
        self.device = params['device'] 
        self.scaler = params['scaler']
        net = params['Network'] 
        self.gp_br = Graph_Processer_Belief_Regulator(
            self.scaler, net, self.device)
        self.gp_pp = Graph_Processer_Physic_Predictor(
            self.scaler, net, self.device, torch.nn.L1Loss())
