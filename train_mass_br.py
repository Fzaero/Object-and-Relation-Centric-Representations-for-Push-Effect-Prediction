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


def get_loss_BR_obj(ypred, ytarget):
    loss = (dataset.gp_br.loss2(ypred, ytarget)*mult2).mean()
    return loss

def getMassErrorOverTime(ypred, ytarget):
    return metric1(ypred['obj_to_predict'], ytarget['obj_to_predict']).mean(dim=[0, 2, 3]).cpu().numpy()


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
    "batch_size_train": 16,
    "batch_size_val": 32,
    'num_workers': 8,
    "num_of_batch_train": (100, 0),  # (0, 10000),
    "num_of_batch_val": (20, 0),  # (0, 500),
    'shuffle': True,   
    'no_angle': True
}

Net = PropagationNetwork(**net_params).to(device)
Net.load_state_dict(torch.load('cylinders_PP.pt',map_location='cuda:0'))
dataset_params['Network'] = Net

dataset = DatasetWrapper(**dataset_params)

optimizer_BR = torch.optim.Adam(Net.obj_output_layers['obj_to_predict'][0].parameters(), lr=3e-4, eps=1e-7, weight_decay=0,amsgrad=True)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_BR, 'min', factor=0.8, verbose=True, patience=4)

loss_train_traj_BR_obj = list()
loss_val_traj_BR_obj = list()

metric1 = torch.nn.MSELoss(reduction='none')
metric2 = torch.nn.MSELoss()

for epoch in range(1000):
    f, ax = plt.subplots(1, 2, figsize=(16, 10))
    # Training
    losses_train_BR_obj = list()
    losses_val_BR_obj = list()
    
    dataset.my_dataset_train.on_epoch_start()
    dataset.my_dataset_val.on_epoch_start()

    Net.train()
    print('----- ', epoch, ' ------')
    for i in range(dataset.my_dataset_train.number_of_batch):
        x, y, batch_name = dataset.my_dataset_train.get_next_batch(i)
        x, y = {k: v.to(device) for k, v in x.items()}, {
            k: v.to(device) for k, v in y.items()}
        if batch_name == 'br':
            y_pred = dataset.gp_br(x)

            loss_obj = get_loss_BR_obj(dataset.scaler.transform_mass(
                y_pred['obj_to_predict']), dataset.scaler.transform_mass(y['obj_to_predict']))
            loss = loss_obj +  get_regularization_loss() + 1e-6 * Net.lstm_loss

            losses_train_BR_obj.append(loss_obj.item())

            loss.backward()
       	    optimizer_BR.step()
            optimizer_BR.zero_grad()
            del x, y, y_pred, loss

    loss_train_traj_BR_obj.append(np.mean(losses_train_BR_obj))

    with torch.set_grad_enabled(False):
        with torch.no_grad():
            Net.eval()
            for i in range(dataset.my_dataset_val.number_of_batch):
                x, y, batch_name = dataset.my_dataset_val.get_next_batch(i)
                x, y = {k: v.to(device) for k, v in x.items()}, {
                    k: v.to(device) for k, v in y.items()}
                if batch_name == 'br':
                    y_pred = dataset.gp_br(x)
                    obj_loss = get_loss_BR_obj(dataset.scaler.transform_mass(
                        y_pred['obj_to_predict']), dataset.scaler.transform_mass(y['obj_to_predict']))

                    losses_val_BR_obj.append(obj_loss.item())
                    del x, y, y_pred
            predicted_br, ground_truth = dataset.val_tester.validate_br()
            mass_error_over_time = getMassErrorOverTime(
                predicted_br, ground_truth)

            del predicted_br, ground_truth
    scheduler.step(np.mean(losses_val_BR_obj))
    loss_val_traj_BR_obj.append(np.mean(losses_val_BR_obj))

    ax[0].set_title('BR Loss Object')
    ax[0].plot(np.log(loss_train_traj_BR_obj))
    ax[0].plot(np.log(loss_val_traj_BR_obj))
    ax[1].set_title('BR Mass Error Over Time')
    ax[1].plot(mass_error_over_time)
    plt.savefig('cylinders_br.png')
    plt.close(f)
    torch.save(Net.state_dict(), 'cylinders_BR.pt')
