import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from typing import *

class MLP(torch.nn.Module):
    def __init__(self, layer_info, activation=torch.nn.ReLU()):
        super(MLP, self).__init__()
        layers = []
        in_dim = layer_info[0]
        for l in layer_info[1:-1]:
            layer = torch.nn.Linear(in_features=in_dim, out_features=l)
            layers.append(layer)
            layers.append(activation)
            in_dim = l
        layers.append(torch.nn.Linear(
            in_features=in_dim, out_features=layer_info[-1]))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        x_shape = x.shape
        x_in = x.view(-1, x_shape[-1])
        x_out = self.layers(x_in)
        return x_out.view(x_shape[:-1]+(-1,))


class GraphLSTM(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(GraphLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        for name, param in self.lstm.named_parameters():
            if "weight_hh" in name:
                torch.nn.init.orthogonal_(param.data)
            elif "weight_ih" in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif "bias" in name:
                torch.nn.init.zeros_(param.data)
                param.data[hidden_dim:2 * hidden_dim] = 1

    def forward(self, x):
        # batch_size, num_of_objects, timesteps, features
        x = x.permute(0, 2, 1, 3).contiguous()
        x_shape = x.shape
        # batch_size*num_of_objects, timesteps, features
        x = x.reshape(-1, x_shape[2], x_shape[3])
        y, _ = self.lstm(x)
        y = y.view(x_shape)  # batch_size , num_of_objects, timesteps, features
        y = y.permute(0, 2, 1, 3)
        return y


class PropagationNetwork(torch.nn.Module):
    def __init__(self, relation_dim=1, object_dim=1, hidden_dim=128, prop_step=5, dropout_rate=0.10, outputs=None):
        super(PropagationNetwork, self).__init__()
        self.rm = MLP([relation_dim, hidden_dim, hidden_dim, hidden_dim])
        self.om = MLP([object_dim, hidden_dim, hidden_dim])
        self.rmp = MLP([3*hidden_dim, hidden_dim])
        self.omp = MLP([3*hidden_dim, hidden_dim])
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.hidden_dim = hidden_dim
        self.loss = torch.nn.MSELoss()
        self.prop_step = prop_step
        if outputs:
            self.outputs = outputs
        else:
            self.outputs = {'object': [{'name': 'out_vel', 'activation': 'linear', 'temporal': False, 'out_size': 3},
                                       {'name': 'obj_to_predict', 'activation': 'linear', 'temporal': True, 'out_size': 1},],
                            'relation': [{'name': 'rel_to_predict', 'activation': 'linear', 'temporal': True, 'out_size': 4},]}
        self.obj_output_layers = {}
        self.rel_output_layers = {}
        self.other_layers = {}
        activation_functions = dict()
        activation_functions['softmax'] = lambda: torch.nn.Softmax(dim=-1)
        activation_functions['sigmoid'] = lambda: torch.nn.Sigmoid(dim=-1)

        for output in self.outputs['object']:
            layers = []
            if output['temporal']:
                layers.append(GraphLSTM(hidden_dim))
            else:
                layer = torch.nn.Linear(
                    in_features=hidden_dim, out_features=hidden_dim)
                layers.append(layer)
                layers.append(torch.nn.ReLU())

            layers.append(torch.nn.Linear(in_features=hidden_dim,
                                          out_features=output['out_size']))
            if output['activation'] != 'linear':
                layers.append(activation_functions[output['activation']]())
            self.other_layers[output['name']] = layers
            layers = torch.nn.Sequential(*layers)
            self.obj_output_layers[output['name']] = (
                layers, output['temporal'])
            self.add_module(
                output['name'], self.obj_output_layers[output['name']][0])
        for output in self.outputs['relation']:
            layers = []
            if output['temporal']:
                layers.append(GraphLSTM(hidden_dim))
            else:
                layer = torch.nn.Linear(
                    in_features=hidden_dim, out_features=hidden_dim)
                layers.append(layer)
                layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(in_features=hidden_dim,
                                          out_features=output['out_size']))
            if output['activation'] != 'linear':
                layers.append(activation_functions[output['activation']]())
            self.other_layers[output['name']] = layers
            layers = torch.nn.Sequential(*layers)
            self.rel_output_layers[output['name']] = (
                layers, output['temporal'])
            self.add_module(
                output['name'], self.rel_output_layers[output['name']][0])

    def forward(self, rel_data, obj_data, sender_relations, receiver_relations, device, temporal=False):
        num_of_dim = len(receiver_relations.shape)
        receiver_relations_T = receiver_relations.permute(
            tuple(range(num_of_dim-2)) + tuple(range(num_of_dim-1, num_of_dim-3, -1)))

        rel_encoding = self.dropout(F.relu(self.rm(rel_data)))
        obj_encoding = self.dropout(F.relu(self.om(obj_data)))
        if self.training:
            self.lstm_loss=0
        propagation = torch.zeros(
            obj_data.shape[:-1]+(self.hidden_dim,), device=device)

        for _ in range(self.prop_step):
            senders_prop = torch.matmul(sender_relations, propagation)
            receivers_prop = torch.matmul(receiver_relations, propagation)
            rmp_vector = torch.cat(
                [rel_encoding, senders_prop, receivers_prop], dim=-1)
            rel_out = F.relu(self.rmp(rmp_vector))
            effect_receivers = torch.matmul(receiver_relations_T, rel_out)
            omp_vector = torch.cat(
                [obj_encoding, effect_receivers, propagation], dim=-1)
            x = self.omp(omp_vector)
            propagation = F.relu(x + propagation)
        outs = dict()
        for out_name in self.obj_output_layers.keys():
            if self.obj_output_layers[out_name][1] == temporal:
                lstm_output = self.obj_output_layers[out_name][0][:-1](propagation)
                outs[out_name] = self.obj_output_layers[out_name][0][-1:](lstm_output)
                if self.training:
                    self.lstm_loss= self.lstm_loss + self.loss(lstm_output[:,1:],lstm_output[:,:-1])
                
        for out_name in self.rel_output_layers.keys():
            if self.rel_output_layers[out_name][1] == temporal:
                lstm_output = self.rel_output_layers[out_name][0][:-1](rel_out)
                outs[out_name] = self.rel_output_layers[out_name][0][-1:](lstm_output)
                if self.training:
                    self.lstm_loss= self.lstm_loss + self.loss(lstm_output[:,1:],lstm_output[:,:-1])
        return outs      

    def forward_transfer(self, rel_data, obj_data, sender_relations, receiver_relations, device, temporal=False):
        num_of_dim = len(receiver_relations.shape)
        receiver_relations_T = receiver_relations.permute(
            tuple(range(num_of_dim-2)) + tuple(range(num_of_dim-1, num_of_dim-3, -1)))

        rel_encoding = self.dropout(F.relu(self.rm(rel_data)))
        obj_encoding = self.dropout(F.relu(self.om(obj_data)))

        propagation = torch.zeros(
            obj_data.shape[:-1]+(self.hidden_dim,), device=device)

        for _ in range(self.prop_step):
            senders_prop = torch.matmul(sender_relations, propagation)
            receivers_prop = torch.matmul(receiver_relations, propagation)
            rmp_vector = torch.cat(
                [rel_encoding, senders_prop, receivers_prop], dim=-1)
            rel_out = F.relu(self.rmp(rmp_vector))
            effect_receivers = torch.matmul(receiver_relations_T, rel_out)
            omp_vector = torch.cat(
                [obj_encoding, effect_receivers, propagation], dim=-1)
            x = self.omp(omp_vector)
            propagation = F.relu(x + propagation)
        outs = dict()
        outs['rel_latent']=rel_out
        outs['obj_latent']=propagation
        return outs

    def forward_latent(self, rel_data, obj_data, sender_relations, receiver_relations, device, temporal=False):
        num_of_dim = len(receiver_relations.shape)
        receiver_relations_T = receiver_relations.permute(
            tuple(range(num_of_dim-2)) + tuple(range(num_of_dim-1, num_of_dim-3, -1)))

        rel_encoding = self.dropout(F.relu(self.rm(rel_data)))
        obj_encoding = self.dropout(F.relu(self.om(obj_data)))

        propagation = torch.zeros(
            obj_data.shape[:-1]+(self.hidden_dim,), device=device)

        for _ in range(self.prop_step):
            senders_prop = torch.matmul(sender_relations, propagation)
            receivers_prop = torch.matmul(receiver_relations, propagation)
            rmp_vector = torch.cat(
                [rel_encoding, senders_prop, receivers_prop], dim=-1)
            rel_out = F.relu(self.rmp(rmp_vector))
            effect_receivers = torch.matmul(receiver_relations_T, rel_out)
            omp_vector = torch.cat(
                [obj_encoding, effect_receivers, propagation], dim=-1)
            x = self.omp(omp_vector)
            propagation = F.relu(x + propagation)
        outs = dict()
        for out_name in self.obj_output_layers.keys():
            if self.obj_output_layers[out_name][1] == temporal:
                outs[out_name] = self.obj_output_layers[out_name][0][:-1](propagation)
        for out_name in self.rel_output_layers.keys():
            if self.rel_output_layers[out_name][1] == temporal:
                outs[out_name] = self.rel_output_layers[out_name][0][:-1](rel_out)
        return outs
