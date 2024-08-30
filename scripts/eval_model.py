# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

'''Evaluate a pre-trained model on validation data'''

import numpy as np
import torch
from torch.utils.data import DataLoader
import os 
import warnings
from tqdm import tqdm
from aistap_sim.utils.data_utils import SimulatedDataset, view_complex, view_real, gen_tapered_noise_torch
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
model_path = 'your/model/file/here'
result_dir = createDir('results/training_test')
dataset_path_prefix = './simMed/'
static_dataset = ''
batch_size = 1
model_type = 'AISTAP_CNN'
kernel_size = 5
norm_output = False
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
dataset = SimulatedDataset(dataset_path_prefix, static_dataset, mode='val', device=device)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
print('Done loading data')

data_shape = next(iter(dataloader))[0].shape
if model_type == 'AISTAP_CNN':
    model = AISTAP_CNN(data_shape, kernel_size=kernel_size, norm_output=norm_output)
elif model_type == 'AISTAP_FC':
    model = AISTAP_FC(data_shape, filter_size=kernel_size, norm_output=norm_output)
else:
    raise ValueError('model_type must be either "AISTAP_CNN" or "AISTAP_FC"')

model.load_state_dict(torch.load(model_path))
model.to(device)
    
print('Beginning eval')
model.eval()
out_list = []
file_list = []
for (X, y, cache, file_) in tqdm(dataloader):
    X = X.to(device)
    pred, _ = model(X, dataset.noise_var)
    if transform_output and cache.numel():
        if cache.numel() == 1:
            pred = dataset.transform.inverse(pred, cache.to(device))
        else:
            pred = dataset.transform.inverse(pred, cache.transpose(0, 1).to(device))
    out_list.append(pred.detach().cpu())
    file_list += file_
test_data_out = torch.cat(out_list, dim=0)
if torch.is_complex(test_data_out): 
    test_data_out = test_data_out.detach().cpu().numpy()
else:
    test_data_out = view_complex(test_data_out, chan_dim=-1)
    test_data_out = test_data_out.detach().cpu().numpy()

rd_img, rd_pred, rd_targ_only, metadata, meta_per_image = dataset.save_and_get_data(test_data_out, file_list, os.path.join(result_dir, 'validation_output.mat'))
print('Done eval')

print('Calculating statistics and plotting')
plot_data = save_sinr_loss_roc(rd_img, rd_pred, metadata, meta_per_image, result_dir, roc_mask_shape, norm_mask_shape, norm_filt_shape)
plot_individual(result_dir, plot_data)

# Visually plot a random sample
data_x, data_y, _, _ = next(iter(dataloader))
model.eval()
out, _ = model(data_x.to(device), dataset.noise_var)
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