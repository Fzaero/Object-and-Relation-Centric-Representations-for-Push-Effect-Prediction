import torch
import math
import numpy as np

class ValidatorAndTester:
    def __init__(self, my_data, gp_pp, gp_br, device, val_len = 50):
        self.gp_pp = gp_pp
        self.gp_br = gp_br
        self.my_data = my_data
        self.device=device
        self.num_of_objects = my_data.num_of_objects
        self.num_of_rel = my_data.num_of_rel

        self.val_len = val_len
        # Validation Data
        self.validate_states_pp = torch.zeros(
            len(my_data.val_indexes), val_len, self.num_of_objects, 8)
        self.validate_states_br = torch.zeros(
            len(my_data.val_indexes), 200, self.num_of_objects, 8)
        self.validate_states_br_lens = list()
        for ind, val_ind in enumerate(my_data.val_indexes):
            traj_len = my_data.obj_states[val_ind].shape[0]
            self.validate_states_br_lens.append(traj_len)
            self.validate_states_pp[ind, :val_len] = torch.from_numpy(
                my_data.obj_states[val_ind][:val_len]).float()
            self.validate_states_br[ind, :min(traj_len, 200)] = torch.from_numpy(
                my_data.obj_states[val_ind][:min(traj_len, 200)]).float()
        self.validate_shapes = torch.from_numpy(
            my_data.obj_shapes[my_data.val_indexes]).float()
        self.validate_rels = torch.from_numpy(
            my_data.edge_features[my_data.val_indexes]).float()

        # Test Data
        max_test_len = 0
        self.test_traj_lens = list()
        for test_ind in my_data.test_indexes:
            traj_len = my_data.obj_states[test_ind].shape[0]
            self.test_traj_lens.append(traj_len)
            if max_test_len < traj_len:
                max_test_len = traj_len
        self.max_test_len = max_test_len
        self.test_states = torch.zeros(
            len(my_data.test_indexes), max_test_len, self.num_of_objects, 8)
        for ind, test_ind in enumerate(my_data.test_indexes):
            traj_len = my_data.obj_states[test_ind].shape[0]
            self.test_states[ind, :traj_len] = torch.from_numpy(
                my_data.obj_states[test_ind][:traj_len]).float()
            self.test_states[ind, traj_len:, :,
                             :3] = self.test_states[ind, traj_len-1, :, :3]
        self.test_shapes = torch.from_numpy(
            my_data.obj_shapes[my_data.test_indexes]).float()
        self.test_rels = torch.from_numpy(
            my_data.edge_features[my_data.test_indexes]).float()
        self.test_rels_no_joint = torch.stack(
            [self.test_rels] * 200, dim=1)
        self.test_rels_no_joint[:,:,:,:]=0
        self.test_rels_no_joint[:,:,:,0]=1



    def validate_pp(self):
        #
        to_be_pred = self.validate_states_pp.clone().to(self.device)
        to_be_pred[:, 1:, 1:, :] = 0
        x = dict()
        x['objects_shape'] = self.validate_shapes.to(self.device)
        x['relation_info'] = self.validate_rels.to(self.device)
        for timestep in range(1, self.val_len):
            x['objects_state'] = to_be_pred[:, timestep-1, :, :] 
            to_be_pred[:, timestep, 1:, :6] = self.gp_pp(x)
        to_be_pred_cpu = to_be_pred.cpu()
        del to_be_pred, x
        return to_be_pred_cpu
    
    def validate_br(self):
        #

        x = dict()
        x['objects_state'] = self.validate_states_br.clone().to(self.device)
        x['objects_shape'] = torch.stack([self.validate_shapes]*200, dim=1).to(self.device)
        x['relation_info'] = torch.stack([self.validate_rels]*200, dim=1).to(self.device)
        ground_truth = dict()
        ground_truth['obj_to_predict'] = x['objects_shape'][:,
                                                            :, :, 2:3].clone()
        ground_truth['rel_to_predict'] = x['relation_info'].clone()
        x['objects_shape'][:, :, :, 2] = 0.6
        x['relation_info'][:, :, :, :] = 0
        batch_size = 16
        # dividing it to batches to that it fits.
        to_be_pred = dict()
        to_be_pred['rel_to_predict'] = torch.zeros_like(x['relation_info'])
        to_be_pred['obj_to_predict'] = torch.zeros_like(
            x['objects_shape'][:, :, :, 2:3])
        number_of_batch = math.ceil(
            len(self.validate_states_br_lens)/batch_size)
        for i in range(number_of_batch):
            x_batch = dict()
            for key in ['objects_state', 'relation_info', 'objects_shape']:
                x_batch[key] = x[key][i*batch_size:(i+1) * batch_size]
            pred = self.gp_br(x_batch)
            for key in ['rel_to_predict', 'obj_to_predict']:
                to_be_pred[key][i * batch_size:(i+1)*batch_size] = pred[key]
            del pred, x_batch
        del x
        for ind, traj_len in enumerate(self.validate_states_br_lens):
            for key in ['rel_to_predict', 'obj_to_predict']:
                if traj_len < 200:
                    to_be_pred[key][ind, traj_len -
                                    1:] = to_be_pred[key][ind, traj_len-1]
        return to_be_pred, ground_truth

    def test_pp(self, number_of_timestep=None):
        if number_of_timestep == None:
            number_of_timestep = self.max_test_len
        to_be_pred = self.test_states.clone()[:,:number_of_timestep].to(self.device)
        to_be_pred[:, 1:, 1:, :] = 0
        x = dict()
        x['objects_shape'] = self.test_shapes.to(self.device)
        x['relation_info'] = self.test_rels.to(self.device)
        for timestep in range(1, number_of_timestep):
            x['objects_state'] = to_be_pred[:, timestep-1, :, :]
            to_be_pred[:, timestep, 1:, :6] = self.gp_pp(x)
        to_be_pred_cpu = to_be_pred.cpu()
        del to_be_pred, x
        return to_be_pred_cpu

    def test_br(self, batch_size=16, test_len=200):
        x = dict()
        x['objects_state'] = self.test_states.clone()[:, :test_len].to(self.device)
        x['objects_shape'] = torch.stack(
            [self.test_shapes] * test_len, dim=1).to(self.device)
        x['relation_info'] = torch.stack(
            [self.test_rels] * test_len, dim=1).to(self.device)
        ground_truth = dict()
        ground_truth['obj_to_predict'] = x['objects_shape'][:,
                                                            :, :, 2:3].clone()
        ground_truth['rel_to_predict'] = x['relation_info'].clone()
        x['objects_shape'][:, :, :, 2] = 0.6
        x['relation_info'][:, :, :, :] = 0
        # dividing it to batches to that it fits.
        to_be_pred = dict()
        to_be_pred['rel_to_predict'] = torch.zeros_like(x['relation_info'])
        to_be_pred['obj_to_predict'] = torch.zeros_like(
            x['objects_shape'][:, :, :, 2:3])
        number_of_batch = math.ceil(
            len(self.test_traj_lens)/batch_size)
        for i in range(number_of_batch):
            x_batch = dict()
            for key in ['objects_state', 'relation_info', 'objects_shape']:
                x_batch[key] = x[key][i*batch_size:(i+1) * batch_size]
            pred = self.gp_br(x_batch)
            for key in ['rel_to_predict', 'obj_to_predict']:
                to_be_pred[key][i * batch_size:(i+1)*batch_size] = pred[key]
        del x
        for ind, traj_len in enumerate(self.test_traj_lens):
            for key in ['rel_to_predict', 'obj_to_predict']:
                to_be_pred[key][ind, min(traj_len - 1, test_len - 1):] = to_be_pred[key][ind, min(test_len - 1, traj_len - 1)]
        return to_be_pred, ground_truth
