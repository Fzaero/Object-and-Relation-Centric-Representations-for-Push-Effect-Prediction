import math
import torch


class Graph_Processer_Belief_Regulator(torch.nn.Module):
    def __init__(self, scaler, Net, device, loss=torch.nn.CrossEntropyLoss(reduction='none')):
        super(Graph_Processer_Belief_Regulator, self).__init__()
        self.scaler = scaler
        self.Net = Net
        self.device = device
        self.loss = loss.to(device)
        self.loss2 = torch.nn.MSELoss(reduction='none').to(device)


    def pre_proces(self, x):
        n_objects = x['objects_state'].shape[-2]
        n_relations = n_objects*(n_objects-1)
        distances_between_objects = torch.cdist(
            x['objects_state'][:, :, :, 0:2], x['objects_state'][:, :, :, 0:2])
        receiver_relations = torch.zeros(
            x['objects_state'].shape[0], x['objects_state'].shape[1], n_relations, n_objects).to(self.device)
        sender_relations = torch.zeros(
            x['objects_state'].shape[0], x['objects_state'].shape[1], n_relations, n_objects).to(self.device)
        distances_between_close_objects = torch.where(distances_between_objects < 0.35,
                                                      distances_between_objects,
                                                      torch.zeros(x['objects_state'].shape[0], x['objects_state'].shape[1], n_objects, n_objects).to(self.device))
        edge_indexes = distances_between_close_objects.nonzero()
        batchs = edge_indexes[:, 0]
        timesteps = edge_indexes[:, 1]
        obj1 = edge_indexes[:, 2]
        obj2 = edge_indexes[:, 3]
        receiver_relations[batchs, timesteps, obj1 *
                           (n_objects-1)+obj2-(obj2 > obj1).long(), obj1] = 1
        sender_relations[batchs, timesteps, obj1 *
                         (n_objects-1)+obj2-(obj2 > obj1).long(), obj2] = 1

        data_state, data_shape = self.scaler.transform(
            x['objects_state'], x['objects_shape'])

        senders_state = torch.matmul(sender_relations, data_state)
        receivers_state = torch.matmul(receiver_relations, data_state)

        senders_shape = torch.matmul(sender_relations, data_shape)
        receivers_shape = torch.matmul(receiver_relations, data_shape)

        receivers_angles = receivers_state[:, :, :, 2:3]

        receivers_basis_first = torch.cat(
            [torch.cos(receivers_angles), torch.sin(receivers_angles)], dim=-1)
        receivers_basis_second = torch.cat([torch.cos(
            receivers_angles+math.pi/2), torch.sin(receivers_angles+math.pi/2)], dim=-1)

        diff_state = senders_state - receivers_state

        diff_pos = diff_state[:, :, :, 0:2]
        diff_vel = diff_state[:, :, :, 3:5]
        control = senders_state[:, :, :, 6:8]  # If not work, try both way
        angle_diff = diff_state[:, :, :, 2:3]
        angle_diff_vel = diff_state[:, :, :, 5:6]

        basis = torch.stack(
            [receivers_basis_first, receivers_basis_second], dim=-2)
        temp_state_to_tf = torch.stack([diff_pos, diff_vel, control], dim=-1)
        state_tfed = torch.matmul(basis, temp_state_to_tf)

        diff_pos = state_tfed[:, :, :, :, 0]
        diff_vel = state_tfed[:, :, :, :, 1]
        control = state_tfed[:, :, :, :, 2]  # If not work, try both way

        # Making continous angle values
        diff_angle_2d = torch.cat(
            [torch.sin(2*angle_diff), torch.cos(2*angle_diff)], dim=-1)

        diff_state = torch.cat(
            [diff_pos, diff_vel, control, diff_angle_2d, angle_diff_vel], dim=-1)

        # angle diff in vel, check
        rel_data = torch.cat(
            [x['relation_info'], diff_state, receivers_shape, senders_shape], dim=-1)

        object_angles = data_state[:, :, :, 2:3]
        objects_basis_first = torch.cat(
            [torch.cos(object_angles), torch.sin(object_angles)], dim=-1)
        objects_basis_second = torch.cat(
            [torch.cos(object_angles+math.pi/2), torch.sin(object_angles+math.pi/2)], dim=-1)

        object_basis = torch.stack(
            [objects_basis_first, objects_basis_second], dim=-2)
        obj_vel = data_state[:, :, :, 3:5]
        obj_vel = obj_vel.reshape(obj_vel.shape+(1,))
        obj_vel_tfed = torch.matmul(object_basis, obj_vel)[:, :, :,:, 0]

        obj_data = torch.cat(
            [obj_vel_tfed, data_state[:, :, :, 5:6], data_shape], dim=-1)

        return rel_data, obj_data, receiver_relations, sender_relations

    def post_proces(self, _, network_outs):
        outs = dict()
        outs['obj_to_predict'] = self.scaler.inv_transform_mass(network_outs['obj_to_predict'])
        outs['rel_to_predict'] = network_outs['rel_to_predict']
        return outs


    def forward(self, x):
        rel_data, obj_data, receiver_relations, sender_relations = self.pre_proces(
            x)
        outs = self.Net(rel_data, obj_data, sender_relations,
                        receiver_relations, self.device, True)
        self.outs = outs
        y = self.post_proces(x, outs)
        return y


    def forward_latent(self, x):
        rel_data, obj_data, receiver_relations, sender_relations = self.pre_proces(
            x)
        outs = self.Net.forward_latent(rel_data, obj_data, sender_relations,
                        receiver_relations, self.device, True)
        return outs


    def forward_transfer(self, x):
        rel_data, obj_data, receiver_relations, sender_relations = self.pre_proces(
            x)
        outs = self.Net.forward_transfer(rel_data, obj_data, sender_relations,
                        receiver_relations, self.device, True)
        return outs


class Graph_Processer_Physic_Predictor(torch.nn.Module):
    def __init__(self, scaler, Net, device, loss=torch.nn.MSELoss()):
        super(Graph_Processer_Physic_Predictor, self).__init__()
        self.scaler = scaler
        self.Net = Net
        self.device = device
        self.loss = loss.to(device)
        self.trans_matrix_pre = torch.eye(6, 6).to(device)
        self.trans_matrix_pre[3:, 3:] = 0
        self.trans_matrix_post = torch.zeros(3, 6).to(device)
        for i in range(3):
            self.trans_matrix_post[i, i] = 1
            self.trans_matrix_post[i, 3+i] = 1

    def pre_proces(self, x):
        n_objects = x['objects_state'].shape[-2]
        n_relations = n_objects*(n_objects-1)
        distances_between_objects = torch.cdist(
            x['objects_state'][:, :, 0:2], x['objects_state'][:, :, 0:2])
        receiver_relations = torch.zeros(
            x['objects_state'].shape[0], n_relations, n_objects).to(self.device)
        sender_relations = torch.zeros(
            x['objects_state'].shape[0], n_relations, n_objects).to(self.device)
        distances_between_close_objects = torch.where(distances_between_objects < 0.35,
                                                      distances_between_objects,
                                                      torch.zeros(x['objects_state'].shape[0], n_objects, n_objects).to(self.device))
        edge_indexes = distances_between_close_objects.nonzero()
        batchs = edge_indexes[:, 0]
        obj1 = edge_indexes[:, 1]
        obj2 = edge_indexes[:, 2]
        receiver_relations[batchs, obj1 *
                           (n_objects-1)+obj2-(obj2 > obj1).long(), obj1] = 1
        sender_relations[batchs, obj1 *
                         (n_objects-1)+obj2-(obj2 > obj1).long(), obj2] = 1

        data_state, data_shape = self.scaler.transform(
            x['objects_state'], x['objects_shape'])

        senders_state = torch.matmul(sender_relations, data_state)
        receivers_state = torch.matmul(receiver_relations, data_state)

        senders_shape = torch.matmul(sender_relations, data_shape)
        receivers_shape = torch.matmul(receiver_relations, data_shape)

        receivers_angles = receivers_state[:, :, 2:3]

        receivers_basis_first = torch.cat(
            [torch.cos(receivers_angles), torch.sin(receivers_angles)], dim=-1)
        receivers_basis_second = torch.cat([torch.cos(
            receivers_angles+math.pi/2), torch.sin(receivers_angles+math.pi/2)], dim=-1)

        diff_state = senders_state - receivers_state

        diff_pos = diff_state[:, :, 0:2]
        diff_vel = diff_state[:, :, 3:5]
        control = senders_state[:, :, 6:8]  # If not work, try both way
        angle_diff = diff_state[:, :, 2:3]
        angle_diff_vel = diff_state[:, :, 5:6]

        basis = torch.stack(
            [receivers_basis_first, receivers_basis_second], dim=-2)
        temp_state_to_tf = torch.stack([diff_pos, diff_vel, control], dim=-1)
        state_tfed = torch.matmul(basis, temp_state_to_tf)

        diff_pos = state_tfed[:, :, :, 0]
        diff_vel = state_tfed[:, :, :, 1]
        control = state_tfed[:, :, :, 2]  # If not work, try both way

        # Making continous angle values
        diff_angle_2d = torch.cat(
            [torch.sin(2*angle_diff), torch.cos(2*angle_diff)], dim=-1)

        diff_state = torch.cat(
            [diff_pos, diff_vel, control, diff_angle_2d, angle_diff_vel], dim=-1)

        # angle diff in vel, check

        rel_data = torch.cat(
            [x['relation_info'], diff_state, receivers_shape, senders_shape], dim=-1)

        object_angles = data_state[:, :, 2:3]
        objects_basis_first = torch.cat(
            [torch.cos(object_angles), torch.sin(object_angles)], dim=-1)
        objects_basis_second = torch.cat(
            [torch.cos(object_angles+math.pi/2), torch.sin(object_angles+math.pi/2)], dim=-1)

        object_basis = torch.stack(
            [objects_basis_first, objects_basis_second], dim=-2)
        obj_vel = data_state[:, :, 3:5]
        obj_vel = obj_vel.reshape(obj_vel.shape+(1,))
        obj_vel_tfed = torch.matmul(object_basis, obj_vel)[:, :, :, 0]

        obj_data = torch.cat(
            [obj_vel_tfed, data_state[:, :, 5:6], data_shape], dim=-1)

        return rel_data, obj_data, receiver_relations, sender_relations

    def post_proces(self, x, network_outs):
        angles = x[:, 1:, 2:3]

        pred = self.scaler.inv_transform_vel(network_outs['out_vel'][:, 1:, :])
        pred_vel = pred[:, :, :2]
        pred_angle_dot = pred[:, :, 2:]

        basis_first = torch.cat(
            [torch.cos(-angles), torch.sin(-angles)], dim=-1)
        basis_second = torch.cat(
            [torch.cos(-angles+math.pi/2), torch.sin(-angles+math.pi/2)], dim=-1)

        basis = torch.stack([basis_first, basis_second], dim=-2)

        vel_transformed_back = torch.matmul(
            basis, pred_vel.reshape(pred_vel.shape + (1,)))[:, :, :, 0]

        x_dot = torch.cat([vel_transformed_back, pred_angle_dot], dim=-1)

        x_next = torch.matmul(x[:, 1:, :6], self.trans_matrix_pre) + \
            torch.matmul(x_dot, self.trans_matrix_post)
        return x_next

    def forward(self, x):
        rel_data, obj_data, receiver_relations, sender_relations = self.pre_proces(
            x)
        outs = self.Net(rel_data, obj_data, sender_relations,
                        receiver_relations, self.device, False)
        self.outs = outs
        y = self.post_proces(x['objects_state'], outs)
        return y
