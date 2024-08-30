# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
'''Example script for training a model on AISTAP-SIM data
'''

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import os 
import warnings
import matplotlib.pyplot as plt
import scipy.io as sio
import functools
from tqdm import tqdm
from aistap_sim.utils.data_utils import SimulatedDataset, view_complex, gen_tapered_noise_torch
from aistap_sim.utils.visualizations import make_color_plots
from aistap_sim.utils.postprocessing import save_sinr_loss_roc, plot_individual
from aistap_sim.MTI_supervised.supervised_net import AISTAP_FC, AISTAP_CNN
from aistap_sim.MTI_supervised.solver import SupervisedSolver, MSELossWrapper

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

def createDir(dirpath):
    dirpath = os.path.normpath(dirpath)
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except OSError:
            pass
    else:
        warnings.warn('result_dir already exists, data may be overwritten')
    return dirpath

'''-----Params-----'''
result_dir = createDir('results/training_test')
dataset_path_prefix = 'data/simMed/'
static_dataset = ''
batch_size = 1
model_type = 'AISTAP_CNN'
kernel_size = 5
dropout = 0.1
norm_output = False
learning_rate = 0.01
num_epochs = 2 
print_every = 100
transform_output = False
roc_mask_shape = (5,5)
norm_mask_shape = (3,3)
norm_filt_shape = (33, 3)
seed = 0
'''----------------'''

print('seed: ' + str(seed))
np.random.seed(seed)
torch.manual_seed(seed)

print('Loading data')
train_dataset = SimulatedDataset(dataset_path_prefix, static_dataset, mode='train', device=device)
val_dataset = SimulatedDataset(dataset_path_prefix, static_dataset, mode='test', device=device)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print('Done loading data')

data_shape = next(iter(train_dataloader))[0].shape
if model_type == 'AISTAP_CNN':
    model = AISTAP_CNN(data_shape, kernel_size=kernel_size, norm_output=norm_output)
elif model_type == 'AISTAP_FC':
    model = AISTAP_FC(data_shape, p=dropout, filter_size=kernel_size, norm_output=norm_output)
else:
    raise ValueError('model_type must be either "AISTAP_CNN" or "AISTAP_FC"')

loss_fn = MSELossWrapper()
update_rule = functools.partial(torch.optim.Adam, lr=learning_rate)
    
solver = SupervisedSolver(model.to(device), loss_fn, train_dataloader, val_dataloader, device, update_rule, num_epochs, print_every)
print('Beginning training')
loss_history, running_loss_history_train, running_loss_history_val = solver.train()
print('Done training')

torch.save(model.state_dict(), os.path.join(result_dir, 'model.pt'))

lx, ly = zip(*loss_history)
rlx, rly = zip(*running_loss_history_train)
rlvx, rlvy = zip(*running_loss_history_val)

sio.savemat(os.path.join(result_dir, 'loss.mat'),
            {'train_loss_x': rlx, 'train_loss_y': rly, 'val_loss_x': rlvx, 'val_loss_y': rlvy})
plt.plot(rlx, rly, label='Training loss')
plt.plot(rlvx, rlvy, label='Validation loss')
plt.title('Training Loss')
plt.legend()
plt.savefig(os.path.join(result_dir,'training_loss.png'))
plt.close()

print('Beginning eval')
model.eval()
out_list = []
file_list = []
for (X, y, cache, file_) in tqdm(val_dataloader):
    X = X.to(device)
    pred, _ = model(X, val_dataset.noise_var)
    if transform_output and cache.numel():
        if cache.numel() == 1:
            pred = val_dataset.transform.inverse(pred, cache.to(device))
        else:
            pred = val_dataset.transform.inverse(pred, cache.transpose(0, 1).to(device))
    out_list.append(pred.detach().cpu())
    file_list += file_
test_data_out = torch.cat(out_list, dim=0)
if torch.is_complex(test_data_out): 
    test_data_out = test_data_out.detach().cpu().numpy()
else:
    test_data_out = view_complex(test_data_out, chan_dim=-1)
    test_data_out = test_data_out.detach().cpu().numpy()

rd_img, rd_pred, rd_targ_only, metadata, meta_per_image = val_dataset.save_and_get_data(test_data_out, file_list, os.path.join(result_dir, 'validation_output.mat'))
print('Done eval')

print('Calculating statistics and plotting')
plot_data = save_sinr_loss_roc(rd_img, rd_pred, metadata, meta_per_image, result_dir, roc_mask_shape, norm_mask_shape, norm_filt_shape)
plot_individual(result_dir, plot_data)

# Visually plot a random sample
data_x, data_y, _, _ = next(iter(val_dataloader))
model.eval()
out, _ = model(data_x.to(device), val_dataset.noise_var)
out = out.detach().cpu()
data_x = data_x.cpu()
# noise = gen_tapered_noise_torch(data_y.shape, val_dataset.noise_var, device)
# data_y += noise
data_y = data_y.cpu()

color_plot_dir = os.path.join(result_dir, 'mag_color_plots')
createDir(color_plot_dir)
mag_range, power_range = make_color_plots(data_x[0], os.path.join(color_plot_dir, 'data_plot.png'),
                                        title='Input Data', normalize_power=False)
make_color_plots(data_y[0], os.path.join(color_plot_dir, 'target_plot.png'), title='Target Data',
                normalize_power=False)
make_color_plots(out[0], os.path.join(color_plot_dir, 'output_plot.png'), title='NN Output',
                normalize_power=False)

