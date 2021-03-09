import itertools
from torch.utils import data
import torch
import numpy as np


class MySampler(data.Sampler):
    def __init__(self, data_source, len_of_data, shuffle=True):
        self.data_source = data_source
        self.len_of_data = len_of_data
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            return iter(torch.randperm(len(self.data_source))[:self.len_of_data].tolist())
        else:
            return iter(range(self.len_of_data))

    def __len__(self):
        return self.len_of_data


class My_Dataset:
    def __init__(self, my_data, val=False):

        if val:
            self.trajectory_indexes = my_data.val_indexes
        else:
            self.trajectory_indexes = my_data.train_indexes

        self.br = self.My_Dataset_Belief_Regulator(
            my_data, self.trajectory_indexes)       
        self.pp = self.My_Dataset_Physic_Predictor(
            my_data, self.trajectory_indexes)

    def setup_dataloaders(self, number_of_batch_br, number_of_batch_pp, **params):
        batch_size = 32
        if 'batch_size' in params.keys():
            batch_size = params['batch_size']
        self.br_dl = data.DataLoader(self.br, sampler=MySampler(
            self.br, batch_size*number_of_batch_br), **params)          
        self.pp_dl = data.DataLoader(self.pp, sampler=MySampler(
            self.pp, batch_size*number_of_batch_pp), **params)

        self.number_of_batch = number_of_batch_br + number_of_batch_pp
        self.pp_start = number_of_batch_br

    def on_epoch_start(self):
        self.br_dl_iterator = iter(self.br_dl)
        self.pp_dl_iterator = iter(self.pp_dl)
        self.batch_order = np.random.permutation(self.number_of_batch)

    def get_next_batch(self, i):
        if self.batch_order[i] < self.pp_start:
            dataloader_iterator = self.br_dl_iterator
            batch_name = 'br'
        else:
            dataloader_iterator = self.pp_dl_iterator
            batch_name = 'pp'

        x, y = next(dataloader_iterator)
        return x, y, batch_name

    class My_Dataset_Belief_Regulator(data.Dataset):

        def __init__(self, my_data, trajectory_indexes, mass_prediction=True, rel_prediction=True):
            self.data = my_data
            list_indexes = list()
            for i in trajectory_indexes:
                timesteps = range(my_data.traj_lens[i]-100)
                list_indexes.extend(list(itertools.product([i, ], timesteps)))
            self.list_indexes = list_indexes
            self.n_objects = self.data.num_of_objects
            self.n_relations = self.n_objects*(self.n_objects-1)
            self.mass_prediction = mass_prediction
            self.rel_prediction = rel_prediction

        def __len__(self):
            return int(len(self.list_indexes))

        def __getitem__(self, index):
            (traj_ind, timestep) = self.list_indexes[index]

            objects_state = torch.from_numpy(
                self.data.obj_states[traj_ind][timestep:timestep+100, :, :]).float()
            objects_shape = torch.from_numpy(
                self.data.obj_shapes[traj_ind]).float()
            objects_shape_repeated = torch.stack([objects_shape]*100)

            if self.mass_prediction:
                obj_to_predict = objects_shape_repeated[:, :, 2:3].clone()
                objects_shape_repeated[:, :, 2] = 0.6  # about average weight
            relations = torch.from_numpy(
                self.data.edge_features[traj_ind]).float()
            relations_repeated = torch.stack([relations]*100)
            if self.rel_prediction:
                rel_to_predict = relations_repeated.clone()
                relations_repeated[:, :, :] = 0  # Fully unknown
            x = {'objects_state': objects_state,
                 'objects_shape': objects_shape_repeated,
                 'relation_info': relations_repeated}
            y = dict()
            if self.mass_prediction:
                y['obj_to_predict'] = obj_to_predict                
            if self.rel_prediction:
                y['rel_to_predict'] = rel_to_predict
            return x, y

    class My_Dataset_Physic_Predictor(data.Dataset):

        def __init__(self, my_data, trajectory_indexes):
            self.data = my_data
            list_indexes = list()
            for i in trajectory_indexes:
                timesteps = range(my_data.traj_lens[i]-6)
                list_indexes.extend(list(itertools.product([i, ], timesteps)))
            self.list_indexes = list_indexes
            self.n_objects = self.data.num_of_objects
            self.n_relations = self.n_objects*(self.n_objects-1)

        def __len__(self):
            return int(len(self.list_indexes))

        def __getitem__(self, index):
            (traj_ind, timestep) = self.list_indexes[index]

            objects_state = torch.from_numpy(
                self.data.obj_states[traj_ind][timestep:timestep+5, :, :]).float()
            objects_shape = torch.from_numpy(
                self.data.obj_shapes[traj_ind]).float()
            relations = torch.from_numpy(
                self.data.edge_features[traj_ind]).float()

            objects_state_next = torch.from_numpy(
                self.data.obj_states[traj_ind][timestep+1:timestep+6, 1:, :6]).float()

            x = {'objects_state': objects_state,
                 'objects_shape': objects_shape,
                 'relation_info': relations}
            y = {'objects_state_next': objects_state_next, }

            return x, y