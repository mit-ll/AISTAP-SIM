# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import torch
from torch.utils.data import DataLoader
from hydra_zen import store, make_custom_builds_fn
from aistap_sim.utils.data_utils import SimulatedDataset, ZScoreStandardize, MedianNormalize, RandomNoiseTransform, RollTransform
from aistap_sim.MTI_supervised.supervised_net import AISTAP_FC, AISTAP_CNN
from aistap_sim.MTI_supervised.solver import SupervisedSolver, MSELossWrapper


pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)
builds = make_custom_builds_fn(populate_full_signature=True)
builds_nofill = make_custom_builds_fn(populate_full_signature=False)

''' ----- Model Configs ----- '''

store(pbuilds(AISTAP_FC,
              p=0.1,
              filter_size=3,
              norm_output=False),
      group='model', name='AISTAP_FC')

store(pbuilds(AISTAP_CNN,
              kernel_size=5,
              norm_output=False),
      group='model', name='AISTAP_CNN')


''' ----- Dataset Configs ----- '''

store(pbuilds(SimulatedDataset,
              dataset_path_prefix='./simMed',
              static_dataset='',
              mode='train',
              pre_load=True,
              length=None,
              transform=None,
              target_transform=None,
              share_transforms=False,
              x_cplx=True,
              y_cplx=True,
              add_noise=True,
              sl_pctl=0.5,
              noise_pctl=0.5,
              sl_db_thresh=35),
      group='dataset', name='default_dataset')

store(builds_nofill(SimulatedDataset,
                    mode='test',
        zen_partial=True),
      group='val_dataset', name='default_val_dataset')


''' ----- DataLoader Configs ----- '''

store(pbuilds(DataLoader,
              batch_size=1,
              shuffle=True,
              drop_last=False),
      group='dataloader', name='default_dataloader')


''' ----- Optimizer Configs ----- '''

store(pbuilds(torch.optim.Adam,
              lr=0.01),
      group='update_rule', name='adam')


''' ----- Loss Configs ----- '''

store(builds(MSELossWrapper),
      group='loss_fn', name='mse_loss')


''' ----- Solver Configs ----- '''

store(pbuilds(SupervisedSolver,
              num_epochs=50,
              print_every=100),
      group='solver', name='default_solver')


''' ----- Transform Configs ----- '''

store(builds(MedianNormalize,
             dim=None),
      group='dataset/transform', name='median_normalize')
store(builds(ZScoreStandardize,
             dim=0),
      group='dataset/transform', name='z_score')
store(builds(RandomNoiseTransform,
             mag_low=0,
             mag_high=1),
      group='dataset/transform', name='noise_transform')
store(builds(RollTransform,
             roll_dim=1),
      group='dataset/transform', name='roll_transform')

# copy configs so they can also be used as target transform
for cfg in store['dataset/transform']:
    store(store[cfg], group='dataset/target_transform', name=cfg[1])
    store(store[cfg], group='val_dataset/transform', name=cfg[1])
    store(store[cfg], group='val_dataset/target_transform', name=cfg[1])