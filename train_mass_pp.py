from src.dataset_wrapper import DatasetWrapper
from src.networks import PropagationNetwork
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda:0'
mult = torch.linspace(0, 2, 100).reshape(1, 100, 1).to(device)
mult2 = torch.linspace(0, 2, 100).reshape(1, 100, 1, 1).to(device)


def get_regularization_loss():
    l2_const = torch.tensor(1e-7).to(device)
    l2_reg = torch.tensor(0.).to(device)
    for param in Net.rm.parameters():
        l2_reg += torch.norm(param)
    for param in Net.om.parameters():
        l2_reg += torch.norm(param)
    for param in Net.rmp.parameters():
        l2_reg += torch.norm(param)
    for param in Net.omp.parameters():
        l2_reg += torch.norm(param)
    for out in ['out_vel', ]:
        for layer in Net.other_layers[out][:-1]:
            for name, param in layer.named_parameters():
                if "weight" == name:
                    l2_reg += torch.norm(param)
                elif "bias" in name:
                    l2_reg += torch.norm(param)
                elif "weight_ih" in name:
                    l2_reg += torch.norm(param)
    return l2_reg*l2_const

net_params = {
    "relation_dim": 23,
    "object_dim": 8,
    "hidden_dim": 256,

}

dataset_params = {
    "device": device,
    "path": 'Data/cylinder_mass_9',
    "num_of_traj": 32000,
    "device": device,
    "data_split": (30000, 1000, 1000),
    "batch_size_train": 32,
    "batch_size_val": 64,
    'num_workers': 8,
    "num_of_batch_train": (0, 10000),  # (0, 10000),
    "num_of_batch_val": (0, 500),  # (0, 500),
    'shuffle': True,
    'no_angle': True
}

Net = PropagationNetwork(**net_params).to(device)
dataset_params['Network'] = Net

dataset = DatasetWrapper(**dataset_params)

optimizer_PP = torch.optim.Adam(Net.obj_output_layers['out_vel'][0].parameters(), lr=3e-4, weight_decay=0, amsgrad=True)  # ,1e-9
optimizer_PP.add_param_group({'params': Net.rm.parameters()})
optimizer_PP.add_param_group({'params': Net.rmp.parameters()})
optimizer_PP.add_param_group({'params': Net.om.parameters()})
optimizer_PP.add_param_group({'params': Net.omp.parameters()})

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_PP, 'min', factor=0.8, verbose=True, patience=20)

loss_train_traj_PP = list()
loss_val_traj_PP = list()

loss_val_full_traj_PP = list()

metric1 = torch.nn.MSELoss(reduction='none')
metric2 = torch.nn.MSELoss()

for epoch in range(1000):
    f, ax = plt.subplots(1, 2, figsize=(16, 10))
    # Training
    losses_train_PP = list()
    losses_val_PP = list()
    dataset.my_dataset_train.on_epoch_start()
    dataset.my_dataset_val.on_epoch_start()

    Net.train()
    print('----- ', epoch, ' ------')
    for i in range(dataset.my_dataset_train.number_of_batch):
        x, y, batch_name = dataset.my_dataset_train.get_next_batch(i)
        x, y = {k: v.to(device) for k, v in x.items()}, {
            k: v.to(device) for k, v in y.items()}
        if batch_name == 'pp':

            ts_cnt = 0
            x_to_give = dict()
            x_to_give['objects_state'] = x['objects_state'][:,
                                                            ts_cnt, :, :].clone()
            x_to_give['objects_shape'] = x['objects_shape']
            x_to_give['relation_info'] = x['relation_info']
            y_pred = torch.zeros_like(y['objects_state_next']).to(device)
            y_pred[:, 0, :, :] = dataset.gp_pp(x_to_give)
            while np.random.uniform() < 0.5 and ts_cnt < 4:
                ts_cnt = ts_cnt+1
                x_to_give['objects_state'] = x['objects_state'][:,
                                                                ts_cnt, :, :].clone()
                x_to_give['objects_state'][:, 1:,
                                           :6] = y_pred[:, ts_cnt-1, :, :]
                y_pred[:, ts_cnt, :, :] = dataset.gp_pp(x_to_give)
            loss = dataset.gp_pp.loss(dataset.scaler.transform_vel(y['objects_state_next'][:, :ts_cnt+1, :, 3:6]),
                                        dataset.scaler.transform_vel(y_pred[:, :ts_cnt+1, :, 3:6])) + get_regularization_loss()
            losses_train_PP.append(loss.item())
            loss.backward()
       	    optimizer_PP.step()
            optimizer_PP.zero_grad()
            del x, y, y_pred, loss
    loss_train_traj_PP.append(np.mean(losses_train_PP))
    with torch.set_grad_enabled(False):
        with torch.no_grad():
            Net.eval()
            for i in range(dataset.my_dataset_val.number_of_batch):
                x, y, batch_name = dataset.my_dataset_val.get_next_batch(i)
                x, y = {k: v.to(device) for k, v in x.items()}, {
                    k: v.to(device) for k, v in y.items()}
                if batch_name == 'pp':
                    ts_cnt = 0
                    x_to_give = dict()
                    x_to_give['objects_state'] = x['objects_state'][:,
                                                                    ts_cnt, :, :].clone()
                    x_to_give['objects_shape'] = x['objects_shape']
                    x_to_give['relation_info'] = x['relation_info']
                    y_pred = torch.zeros_like(y['objects_state_next']).to(device)
                    y_pred[:, 0, :, :] = dataset.gp_pp(x_to_give)
                    while np.random.uniform() < 0.5 and ts_cnt < 4:
                        ts_cnt = ts_cnt+1
                        x_to_give['objects_state'] = x['objects_state'][:,
                                                                        ts_cnt, :, :].clone()
                        x_to_give['objects_state'][:, 1:,
                                                :6] = y_pred[:, ts_cnt-1, :, :]
                        y_pred[:, ts_cnt, :, :] = dataset.gp_pp(x_to_give)
                    loss = dataset.gp_pp.loss(dataset.scaler.transform_vel(y['objects_state_next'][:, :ts_cnt+1, :, 3:6]),
                                                dataset.scaler.transform_vel(y_pred[:, :ts_cnt+1, :, 3:6])) + get_regularization_loss()
                    losses_val_PP.append(loss.item())
                    del x, y, y_pred
            traj_pred = dataset.val_tester.validate_pp()
            loss_val_full_traj_PP.append(
                100*np.sqrt(metric2(dataset.val_tester.validate_states_pp[:, :, :, 0:2], traj_pred[:, :, :, 0:2]).item()))

            del traj_pred
    scheduler.step(np.mean(losses_val_PP))
    loss_val_traj_PP.append(np.mean(losses_val_PP))
    ax[0].set_title('PP Loss')
    ax[0].plot(np.log(loss_train_traj_PP))
    ax[0].plot(np.log(loss_val_traj_PP))
    ax[1].set_title('PP full trajectory Error')
    ax[1].plot(np.log(loss_val_full_traj_PP))
    plt.savefig('cylinders_pp.png')
    plt.close(f)
    torch.save(Net.state_dict(), 'cylinders_PP.pt')
