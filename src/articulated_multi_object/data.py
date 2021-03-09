import json
from os.path import join
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import math


class Scaler:
    def __init__(self, dataset, num_of_traj, device, start_traj_ind=0):
        self.pos_scaler = StandardScaler()
        self.vel_scaler = StandardScaler()
        self.angle_scaler = StandardScaler()

        temp_torch_pos = torch.zeros(
            num_of_traj, max(dataset.traj_lens.values()), 10, 2)
        for ind, obj_state in enumerate(dataset.obj_states[start_traj_ind:start_traj_ind+num_of_traj]):
            temp_torch_pos[ind, :dataset.traj_lens[start_traj_ind+ind]
                           ] = torch.from_numpy(obj_state[:, :, :2])

        for i in range(int(math.ceil(num_of_traj/2000))):
            print(i)
            pos_dists_pre = torch.cdist(
                temp_torch_pos[i*2000:(i+1)*2000], temp_torch_pos[i*2000:(i+1)*2000]).triu()
            pos_dists_pre = pos_dists_pre[pos_dists_pre > 0.0001]
            pos_dists_pre = pos_dists_pre[pos_dists_pre < 0.35]
            if i == 0:
                pos_dists = pos_dists_pre
            else:
                pos_dists = torch.cat([pos_dists, pos_dists_pre])
        pos_dists = torch.cat([pos_dists, -pos_dists])
        self.pos_scaler.fit(pos_dists.numpy().reshape(-1, 1))

        temp_vel = np.zeros((num_of_traj, max(dataset.traj_lens.values()), 2))
        for ind, obj_state in enumerate(dataset.obj_states[start_traj_ind:start_traj_ind+num_of_traj]):
            temp_vel[ind, :dataset.traj_lens[start_traj_ind+ind]
                     ] = obj_state[:, 0, 3:5]
        temp_angle = np.zeros(
            (num_of_traj, max(dataset.traj_lens.values()), 10, 1))
        for ind, obj_state in enumerate(dataset.obj_states[start_traj_ind:start_traj_ind+num_of_traj]):
            temp_angle[ind, :dataset.traj_lens[start_traj_ind+ind]
                       ] = obj_state[:, :, 5:6]
        # Velocity Scaling
        vels = np.linalg.norm(temp_vel, axis=-1)
        # Remove 0 values, lower their contribution on variance.
        vels = vels[vels > 0.0001]
        vels = np.concatenate([vels, -vels])
        self.vel_scaler.fit(vels.reshape(-1, 1))
        # Angle Scaling
        angles = np.linalg.norm(temp_angle, axis=-1)
        # Remove 0 values, lower their contribution on variance.
        angles = angles[angles > 0.00001]
        angles = np.concatenate([angles, -angles])


        scaler_var_state = torch.ones(8)
        scaler_var_shape = torch.ones(5)
        scaler_mean_shape = torch.zeros(5)

        scaler_var_state[0:2] = np.sqrt(self.pos_scaler.var_[0])
        scaler_var_state[3:5] = np.sqrt(self.vel_scaler.var_[0])
        if len(angles)>0:
            self.angle_scaler.fit(angles.reshape(-1, 1))
            scaler_var_state[5:6] = np.sqrt(self.angle_scaler.var_[0])
        scaler_var_state[6:8] = np.sqrt(self.vel_scaler.var_[0])

        self.mass_scaler = StandardScaler()
        self.shape_scaler = StandardScaler()

        self.mass_scaler.fit(
            dataset.obj_shapes[start_traj_ind:start_traj_ind+num_of_traj, :, 2:3].reshape(-1, 1))
        self.shape_scaler.fit(
            dataset.obj_shapes[start_traj_ind:start_traj_ind+num_of_traj, :, 3:5].reshape(-1, 1))

        scaler_var_shape[3:5] = np.sqrt(self.shape_scaler.var_[0])
        scaler_var_shape[2:3] = np.sqrt(self.mass_scaler.var_[0])

        scaler_mean_shape[2:3] = self.mass_scaler.mean_[0]
        scaler_mean_shape[3:5] = self.shape_scaler.mean_[0]

        self.scaler_var_state = scaler_var_state.float().to(device)
        self.scaler_mean_state = torch.zeros(8).float().to(device)
        self.scaler_var_shape = scaler_var_shape.float().to(device)
        self.scaler_mean_shape = scaler_mean_shape.float().to(device)

    def transform(self, data_state, data_shape):
        return (data_state-self.scaler_mean_state)/self.scaler_var_state, (data_shape-self.scaler_mean_shape)/self.scaler_var_shape

    def transform_vel(self, data):
        return (data-self.scaler_mean_state[3:6])/self.scaler_var_state[3:6]

    def transform_mass(self, data):
        return (data-self.scaler_mean_shape[2:3])/self.scaler_var_shape[2:3]

    def inv_transform(self, data_state, data_shape):
        return data_state*self.scaler_var_state + self.scaler_mean_state, data_shape*self.scaler_var_shape + self.scaler_mean_shape

    def inv_transform_vel(self, data):
        return data*self.scaler_var_state[3:6]+self.scaler_mean_state[3:6]

    def inv_transform_mass(self, data):
        return data*self.scaler_var_shape[2:3]+self.scaler_mean_shape[2:3]


def fix_angle(array):
    while True:
        indexes = np.where(array > np.pi/2)
        if len(indexes[0]) == 0:
            break
        array[indexes] -= np.pi
    while True:
        indexes = np.where(array < -np.pi/2)
        if len(indexes[0]) == 0:
            break
        array[indexes] += np.pi
    return array


class My_Data:
    def __init__(self, **kwargs):
        num_of_traj = kwargs['num_of_traj']
        self.num_of_traj=num_of_traj
        PATH = kwargs['PATH']
        self.PATH=PATH
        sample_file = join(PATH, '1.txt')
        NO_ANGLE = kwargs['no_angle']
        with open(sample_file) as f_in:
            sample_data = json.load(f_in)
        num_of_objects = 0
        for key in sample_data.keys():
            if key.isdigit() and int(key)+1 > num_of_objects:
                num_of_objects = int(key)+1
        num_of_rel = num_of_objects*(num_of_objects-1)
        obj_shape_features = np.zeros(
            (num_of_traj,) + np.array(sample_data['shapes']).shape)
        traj_lens = dict()
        obj_states = list()
        edge_features = np.zeros((num_of_traj, num_of_rel, 4))
        for i in range(num_of_traj):
            sample_file = join(PATH, str(i)+'.txt')
            with open(sample_file) as f_in:
                sample_data = json.load(f_in)
            num_of_timesteps = np.array(sample_data['0']).shape[0]
            traj_lens[i] = num_of_timesteps
            state_features = np.zeros((num_of_timesteps, num_of_objects, 8))
            for obj in range(num_of_objects):
                state_features[:, obj, :3] = sample_data[str(obj)]
            if NO_ANGLE:
                state_features[:, :, 2] = 0
            state_features[1:, :, 3:6] = state_features[1:,
                                                        :, :3]-state_features[:-1, :, :3]
            state_features[:, :, 5][state_features[:, :, 5] > 2] -= 2*np.pi
            state_features[:, :, 5][state_features[:, :, 5] < -2] += 2*np.pi
            state_features[:-1, 0, 6:] = state_features[1:, 0, 3:5]
            obj_states.append(state_features)


            obj_shape_features[i] = np.array(
                sample_data['shapes'])  # ToDo angle vs shape
            swap_ind = np.where(
                obj_shape_features[i, :, -1] > obj_shape_features[i, :, -2])
            obj_shape_features[i, swap_ind, -2], obj_shape_features[i, swap_ind, -
                                                                    1] = obj_shape_features[i, swap_ind, -1], obj_shape_features[i, swap_ind, -2]
            if not NO_ANGLE:
                state_features[:, swap_ind, 2] -= np.pi/2
                state_features[:, :, 2] = fix_angle(state_features[:, :, 2])
                      
            edges_np = np.array(sample_data['edges'])
            obj1 = edges_np[:, 1]+1
            obj2 = edges_np[:, 2]+1
            edges1 = np.int32(obj1*(num_of_objects-1) + obj2 - (obj2 > obj1))
            edges2 = np.int32(obj2*(num_of_objects-1) + obj1 - (obj1 > obj2))
            edge_features[i, list(set(range(num_of_rel)) -
                                  set(edges1)-set(edges2)), 0] = 1
            edge_features[i, list(edges1), np.int32(edges_np[:, 0])] = 1
            edge_features[i, list(edges2), np.int32(edges_np[:, 0])] = 1
        self.obj_states = obj_states
        self.obj_shapes = obj_shape_features
        self.edge_features = edge_features
        self.num_of_objects = num_of_objects
        self.num_of_rel = num_of_rel
        self.traj_lens = traj_lens

    def split_data(self, train, val, test, device, scaler = None):
        trajs = list(range(train+val+test))
        self.train_indexes = trajs[: train]
        if val == 0:
            self.val_indexes = trajs[0:1]  # Dummy
        else:
            self.val_indexes = trajs[train: train + val]
        self.test_indexes = trajs[train+val:]
        if scaler is None:
            scaler = Scaler(self, train, device)
        self.scaler = scaler
        return self.scaler