# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

'''Example script for training model using Hydra-Zen (https://mit-ll-responsible-ai.github.io/hydra-zen/)
'''
import numpy as np
import torch
import os 
import sys
import matplotlib.pyplot as plt
import scipy.io as sio
import functools
from tqdm import tqdm
from hydra_zen import store, zen, launch, multirun
from hydra.conf import HydraConf, JobConf, RunDir, SweepDir
from aistap_sim.utils.data_utils import view_complex, gen_tapered_noise_torch
from aistap_sim.utils.visualizations import make_color_plots
from aistap_sim.utils.postprocessing import save_sinr_loss_roc, plot_sinr_loss_roc

# Note: modify default settings in configs.py if desired
import configs

@zen
def pre_seed(seed):
    print('seed: ' + str(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)

@store(name='run_train', transform_output=False, zen_meta={'seed': 0}, hydra_defaults=['_self_',
    {'model': 'AISTAP_CNN'},
    {'dataset': 'default_dataset'},
    {'val_dataset': 'default_val_dataset'},
    {'dataloader': 'default_dataloader'},
    {'update_rule': 'adam'},
    {'loss_fn': 'mse_loss'},
    {'solver': 'default_solver'}])
def run_train(model, dataset, val_dataset, dataloader, update_rule, loss_fn, solver, transform_output, mode='val_data',
              roc_mask_shape=(5,5), norm_mask_shape=(3,3), norm_filt_shape=(33,3)):
    '''Main task function to load data, train the model, and save output. Should be called by hydra_zen.launch.

    Parameters
    ----------
    model : partial(torch.nn.Module)
        Model to train
    dataset : partial(torch.utils.data.Dataset)
        PyTorch dataset containing training data
    val_dataset : partial(torch.utils.data.Dataset)
        PyTorch dataset containing validation data
    dataloader : partial(torch.utils.data.DataLoader)
        PyTorch DataLoader for loading data
    update_rule : partial(torch.optim.Optimizer)
        Optimizer to use for training
    loss_fn : torch.nn._Loss
        Loss function to use for training
    solver : partial(aistap_sim.MTI_supervised.solver.Solver)
        Solver class to train model
    transform_output : bool
        If True, invoke dataset inverse transform (if exists) at validation time
    mode : str
        If 'val_data', validation dataset is used for validation. If 'train_data',
        traning data is used for validation data.
    roc_mask_shape : tuple[int, int]
        Shape of mask for collection of ROC statistics
    norm_mask_shape : tuple[int, int]
        Shape of target mask used during adaptive CFAR normalization filter
    norm_filt_shape : tuple[int, int]
        Shape of convolutional filter used during adaptive CFAR normalization
    '''
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    def createDir(dirpath):
        dirpath = os.path.normpath(dirpath)
        if not os.path.exists(dirpath):
            try:
                os.makedirs(dirpath)
            except OSError:
                pass
        return dirpath

    result_dir = '.'

    # merge configs to allow val_dataset to inherit keyword arguments from dataset
    args_train, kwargs_train = dataset.args, dataset.keywords
    args_val, kwargs_val = val_dataset.args, val_dataset.keywords
    merged_kwargs = {**kwargs_train, **kwargs_val}  # dictionary unpacking, kwargs_val takes priority
    val_dataset = functools.partial(val_dataset.func, *args_val, **merged_kwargs)
    
    print('Loading data')
    train_dataset = dataset(device=device)
    val_dataset = val_dataset(device=device)

    train_dataloader = dataloader(train_dataset, shuffle=True, drop_last=True)
    val_dataloader = dataloader(val_dataset, batch_size=1, shuffle=False)
    print('Done loading data')

    data_shape = next(iter(train_dataloader))[0].shape
    model = model(data_shape)
    solver = solver(model.to(device), loss_fn, train_dataloader, val_dataloader, device, update_rule=update_rule)

    print('Beginning training')
    loss_history, running_loss_history_train, running_loss_history_val = solver.train()
    print('Done training')
    
    torch.save(model.state_dict(), 'model.pt')

    lx, ly = zip(*loss_history)
    rlx, rly = zip(*running_loss_history_train)
    rlvx, rlvy = zip(*running_loss_history_val)

    sio.savemat(os.path.join(result_dir, 'loss.mat'),
                {'train_loss_x': rlx, 'train_loss_y': rly, 'val_loss_x': rlvx, 'val_loss_y': rlvy})

    plt.plot(rlx, rly, label='Training loss')
    plt.plot(rlvx, rlvy, label='Validation loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.close()

    def test_loop(model, dataloader_eval, dataset_eval):
        model.eval()
        out_list = []
        file_list = []
        for (X, y, cache, file_) in tqdm(dataloader_eval, file=sys.stdout, mininterval=10):
            X = X.to(device)
            pred, _ = model(X, dataset_eval.noise_var)
            if transform_output and cache.numel():
                if cache.numel() == 1:
                    pred = dataset_eval.transform.inverse(pred, cache.to(device))
                else:
                    pred = dataset_eval.transform.inverse(pred, cache.transpose(0, 1).to(device))
            out_list.append(pred.detach().cpu())
            file_list += file_
        test_data_out = torch.cat(out_list, dim=0)
        if torch.is_complex(test_data_out): 
            test_data_out = test_data_out.detach().cpu().numpy()
        else:
            test_data_out = view_complex(test_data_out, chan_dim=-1)
            test_data_out = test_data_out.detach().cpu().numpy()

        return test_data_out, file_list

    print('Beginning eval')
    if mode == 'val_data':
        test_data_out, file_list = test_loop(model, val_dataloader, val_dataset)
        rd_img, rd_pred, rd_targ_only, metadata, meta_per_image = val_dataset.save_and_get_data(test_data_out, file_list, os.path.join(result_dir, 'validation_output.mat'))
    else:
        test_data_out, file_list = test_loop(model, train_dataloader, train_dataset)
        rd_img, rd_pred, rd_targ_only, metadata, meta_per_image = train_dataset.save_and_get_data(test_data_out, file_list, os.path.join(result_dir, 'validation_output.mat'))
    print('Done eval')

    save_sinr_loss_roc(rd_img, rd_pred, metadata, meta_per_image, '.', tuple(roc_mask_shape), tuple(norm_mask_shape), tuple(norm_filt_shape))

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

    train_dataset.close()
    val_dataset.close()


if __name__ == '__main__':
    run_dir = 'results/${now:%Y-%m-%d}/${now:%H-%M-%S}_${hydra.job.name}'
    store(HydraConf(
        job=JobConf(chdir=True),
        run=RunDir(run_dir),
        sweep=SweepDir(dir=run_dir, subdir='job_${hydra.job.num}')))
    
    store.add_to_hydra_store()

    task_function = zen(run_train, pre_call=pre_seed)

    jobname = 'jobname'

    # Use absolute paths for dataset path prefix
    jobs = launch(
        store[None, 'run_train'],
        task_function,
        overrides={
            'seed': 0,
            'dataset.dataset_path_prefix': '/path/to/data/simMed/',  
            'dataset.static_dataset': '',
            'dataset.pre_load': True,
            'solver.num_epochs': 2,
            'model': 'AISTAP_FC',
        },
        multirun=True,
        job_name=jobname,
        version_base='1.2'
    )
    
    # Edit this to add labels to plots
    overrides_of_interest = []

    job_dir, _ = os.path.split(jobs[0][0].working_dir)
    plot_sinr_loss_roc(job_dir, overrides_of_interest)