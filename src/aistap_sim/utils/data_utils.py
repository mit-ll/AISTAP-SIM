# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import torch
import os
import glob
import copy
import scipy.io as sio
from scipy.signal import convolve
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from aistap_sim.utils import LazyMatfileReader
from aistap_sim.utils.postprocessing import get_targ_coords, get_target_mask, get_masked_values


def view_real(x_cplx, chan_dim=-1, reshape=False):
    '''Creates a view of a complex tensor as real tensor with imaginary parts interleaved as real numbers along the 
    channel axis
    
    Parameters
    ----------
    x_cplx : torch.Tensor(torch.cfloat)
        Complex data
    chan_dim : int
        Index of the channel dimension in x_cplx
    reshape : bool
        Indicates whethere the returned tensor should be a view or a copy (torch.reshape)
    
    Returns
    -------
    x_real : torch.Tensor(torch.float32)
        Real view tensor
    '''
    new_shape = list(x_cplx.shape)
    new_shape[chan_dim] *= 2
    x_real = torch.view_as_real(x_cplx)
    if chan_dim != -1:
        num_dim = len(new_shape)
        permute_to = list(range(len(new_shape)))
        permute_to.insert(chan_dim + 1, num_dim)
        x_real = torch.permute(x_real, permute_to)
        
    if reshape:
        x_real = x_real.reshape(new_shape)
    else:
        x_real = x_real.view(new_shape)
    return x_real


def view_complex(x_real, chan_dim=-1, reshape=False):
    '''Creates a view of an interleaved real tensor as a complex tensor
    
    Parameters
    ----------
    x_real : torch.tensor(torch.float32)
        Interleaved real tensor
    chan_dim : int
        Index of the channel dimension in x_real
    reshape : bool
        Indicates whethere the returned tensor should be a view or a copy (torch.reshape)
    
    Returns
    -------
    x_cplx : torch.tensor(torch.cfloat)
        Complex view tensor
    '''
    new_shape = list(x_real.shape)
    new_shape[chan_dim] //= 2

    if chan_dim != -1:
        new_shape.insert(chan_dim + 1, 2)
        permute_to = list(range(len(new_shape)))
        permute_to.pop(chan_dim + 1)
        permute_to.append(chan_dim + 1)
        if reshape:
            x_real = x_real.reshape(new_shape)
            x_real = torch.permute(x_real, permute_to).contiguous()
        else:
            x_real = x_real.view(new_shape)
            x_real = torch.permute(x_real, permute_to)
        x_cplx = torch.view_as_complex(x_real)

    else:
        new_shape.append(2)
        if reshape:
            x_cplx = torch.view_as_complex(x_real.reshape(*new_shape))
        else:
            x_cplx = torch.view_as_complex(x_real.view(*new_shape))

    return x_cplx
    

def get_ideal_noise_var(rd_targ_only, meta_per_image, sl_db_thresh, sidelobe_percentile=0.5, noise_percentile=0.5):
    ''' Computes ideal noise variance (for circularly complex multivariate normal distribution) to add to data based on 
    sample of sidelobe power distribution from rd_targ_only

    Parameters
    ----------
    rd_targ_only : numpy.ndarray (N, R, D, C)
        Target training labels
    meta_per_image : list[dict]
        Metadata for each image containing range-doppler coordinates of targets
    sl_db_thresh : int or float
        Threshold for detecting sidelobes, specified as dB power difference between peak target response and mean sidelobe power
    sidelobe_percentile : float
        Specifies the percentile of sidelobe amplitude distribution at which to place the specified percentile of the noise distribution
    noise_percentile : float
        Specifies the percentile of the generated noise amplitude distribution (Rayleigh) to align with the specified sidelobe distribution percentile

    Returns
    -------
    var : float
        Calculated variance, intended to be used as diagonal entries of 2D multivariate normal distribution covariance matrix
    '''

    rd_mag = np.abs(np.sum(rd_targ_only, axis=-1))
    true_range, true_dop, true_xr = get_targ_coords(meta_per_image, rd_targ_only.shape)
    targ_coords = np.stack((true_range, true_dop), axis=-1)
    mask_win = (5,5)
    mask = get_target_mask(rd_mag.shape, targ_coords, range_win=mask_win[0], dop_win=mask_win[1])
    fa = rd_mag[~mask]
    det = np.nanmax(get_masked_values(rd_mag, targ_coords, range_win=mask_win[0], dop_win=mask_win[1]), axis=1)
    det_low = 20*np.log10(np.percentile(det, 1))
    fa_sl = fa[20*np.log10(fa + 1e-12) > det_low - sl_db_thresh]  # detect sidelobes using sl_db_thresh cutoff
    # Place scale such that x percentile of Rayleigh distribution occurs at y percentile of sidelobe amplitude
    scale = np.percentile(fa_sl, sidelobe_percentile*100) / np.sqrt(-2*np.log(1-noise_percentile))
    num_channels = rd_targ_only.shape[3]
    # variance of Gaussian noise for each complex part, pre-sum channel
    var = scale**2 / num_channels
    return var


def gen_tapered_noise(shape, var):
    ''' Generates circular complex noise in NumPy format with a convolutional taper (blur) of variance from get_ideal_noise_var

    Parameters
    ----------
    shape : tuple[int]
        Shape of output noise
    var : float
        Variance of noise

    Returns
    -------
    noise_conv : numpy.ndarray
        Output noise
    '''

    noise = np.random.multivariate_normal(np.zeros(2), np.eye(2)*var, size=(shape + np.array([2, 2, 0]))).view(np.complex128).squeeze()
    winfilt = np.array([[0.1186, 0.3443, 0.1186],
                        [0.3443, 1.0000, 0.3443],
                        [0.1186, 0.3443, 0.1186]])
    winfilt = winfilt / np.sqrt(np.sum(winfilt**2))
    winfilt = np.expand_dims(winfilt, 2)
    noise_conv = convolve(noise, winfilt, mode='valid')
    return noise_conv

def gen_tapered_noise_torch(shape, var, device):
    ''' Generates circular complex noise in PyTorch format with a convolutional taper (blur) of variance from get_ideal_noise_var

    Parameters
    ----------
    shape : torch.Size
        Shape of output noise
    var : float
        Variance of noise
    device : torch.device
        Device on which to create the noise tensor
        
    Returns
    -------
    noise_conv : torch.Tensor
        Output noise
    '''

    if var > 0:
        squeeze = False
        if len(shape) < 4:
            squeeze = True
            shape = [1, *shape]
        num_chan = shape[3]
        dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2).to(device), torch.eye(2).to(device)*var)
        noise = torch.view_as_complex(dist.sample((shape[0], num_chan, shape[1]+2, shape[2]+2)))
        winfilt = torch.tensor([[0.1186, 0.3443, 0.1186],
                                [0.3443, 1.0000, 0.3443],
                                [0.1186, 0.3443, 0.1186]], dtype=torch.cfloat, device=device)
        winfilt = (winfilt / torch.sqrt(torch.sum(winfilt**2))).expand(num_chan, 1, 3, 3)
        noise_conv = F.conv2d(noise, winfilt, stride=1, padding='valid', groups=num_chan)
        if squeeze:
            return noise_conv.squeeze().permute(1, 2, 0)
        else:
            return noise_conv.permute(0, 2, 3, 1)
    else:
        return torch.zeros(shape, device=device)

        
class SimulatedDataset(Dataset):
    '''PyTorch Dataset for simulated radar data
    '''
    def __init__(self, dataset_path_prefix, static_dataset, mode='train', pre_load=True, length=None, transform=None,
                 target_transform=None, share_transforms=False, x_cplx=True, y_cplx=True, add_noise=True, 
                 sl_pctl=0.5, noise_pctl=0.5, sl_db_thresh=35, device=None):
        '''
        Parameters
        ----------
        static_dataset : str
            Name of dataset file on disk to read
        dataset_path_prefix : str
            Path to parent directory of dataset folder
        mode : str
            'sample', 'train' or 'test' to specify dataset split to use (sample just loads a small file)
        pre_load : bool
            Specifies whether to load all data to memory during initialization or to use lazy loading
        length : int
            Truncated length of dataset to use in number of images, if None the entire dataset is used
        transform
            Transform class applied to input data
        target_transform
            Transform class applied to training labels
        share_transforms : bool
            True if target_transform should be ignored and transform should
            be used for both input and training labels, sharing parameters
        x_plx : bool
            If False, complex input data is split into real and imaginary parts
        y_plx : bool
            If False, complex training label data is split into real and imaginary parts
        add_noise : bool
            Specifies whether a noise variance will be calculated, otherwise it is set to zero
        sl_pctl, noise_pctl, sl_db_thresh, device
            See get_ideal_noise_var
        '''

        self.static_dataset = static_dataset
        search_mats = os.path.join(dataset_path_prefix + self.static_dataset, '*.mat')
        all_mat_files = glob.glob(search_mats)
        if len(all_mat_files) == 0:
            print("Absolute path is", os.path.abspath(os.path.curdir))
            raise FileNotFoundError(f"No .mat files found in {os.path.curdir} with {search_mats}")
        if mode == 'sample':
            sample_files = [f for f in all_mat_files if 'sample' in os.path.basename(f)]
            if len(sample_files) == 0:
                raise FileNotFoundError(f"No *sample*.mat files found at {search_mats}")
            elif len(sample_files) > 1:
                sample_files.sort()
                print("Using first sample file found {sample_files[0]}")
            static_dataset_path = sample_files[0]
        elif mode == 'train':
            train_files = [f for f in all_mat_files if 'train' in os.path.basename(f)]
            if len(train_files) == 0:
                raise FileNotFoundError(f"No *train*.mat files found at {search_mats}")
            elif len(train_files) > 1:
                train_files.sort()
                print(f"Using first training file found {train_files[0]}")
            static_dataset_path = train_files[0]
        elif mode == 'test':
            test_files = [f for f in all_mat_files if 'test' in os.path.basename(f)]
            if len(test_files) == 0:
                raise FileNotFoundError(f"No *test*.mat files found at {search_mats}")
            elif len(test_files) > 1:
                test_files.sort()
                print(f"Using first test file found {test_files[0]}")
            static_dataset_path = test_files[0]
 
        else:
            raise ValueError(f"mode must be either 'sample', 'train' or 'test', got {mode}")

            
        if pre_load:
            lmfr = LazyMatfileReader(static_dataset_path, pre_load=True)
            self.data = {}
            for key in lmfr.keys():
                self.data[key] = lmfr[key]
        else:
            self.data = LazyMatfileReader(static_dataset_path)

        self.length = length
        self.device = device
        self.x_cplx = x_cplx
        self.y_cplx = y_cplx
        self.transform = transform
        if share_transforms:
            self.target_transform = copy.deepcopy(self.transform)
        else:
            self.target_transform = target_transform

        if add_noise:
            self.noise_var = get_ideal_noise_var(self.data['rd_targ_only'][:].view(complex).transpose(0, 3, 2, 1), self.data['meta_per_image'][:],
                                                sl_db_thresh, sidelobe_percentile=sl_pctl, noise_percentile=noise_pctl)
        else:
            self.noise_var = 0

            
    def __len__(self):
        if self.length is not None:
            return min(self.length, self.data['rd_img'].shape[0])
        else:
            return self.data['rd_img'].shape[0]
    

    def __getitem__(self, idx):
        if isinstance(idx, int):
            # (C, D, R)
            data_x = torch.from_numpy(self.data['rd_img'][idx].view(complex).astype(np.csingle))
            data_y = torch.from_numpy(self.data['rd_targ_only'][idx].view(complex).astype(np.csingle))

            # (R, D, C) with channel having smallest stride
            data_x = data_x.permute(2, 1, 0).contiguous().to(self.device)
            data_y = data_y.permute(2, 1, 0).contiguous().to(self.device)

            if not self.x_cplx:
                data_x = view_real(data_x, chan_dim=-1)
            if not self.y_cplx:
                data_y = view_real(data_y, chan_dim=-1)

            cache = torch.tensor([])
            if self.transform:
                data_x, cache = self.transform(data_x)
            if self.target_transform:
                data_y, _ = self.target_transform(data_y, cache)
            
            return data_x, data_y, cache, idx

        elif isinstance(idx, str):
            return self.data[idx]
        
        else:
            raise TypeError('index must be either int or string')

    
    def save_and_get_data(self, model_out, eval_inds, out_file=''):
        '''Saves validation data along with dataset with appropriate indices, dimensions, and ordering. Also returns
        data for further use.
        
        Parameters
        ----------
        model_out : numpy.ndarray
            Output of model inference at test time
        eval_inds : list[int]
            Indices of dataset that correspond to model_out
        out_file : str
            If out_file is nonempty, data will be saved to this filepath
        
        Returns
        -------
        rd_img : numpy.ndarray
            Slice of input data corresponding to rd_pred
        rd_pred : numpy.ndarray
            Model output
        rd_targ_only : numpy.ndarray
            Slice of target labels corresponding to rd_pred
        metadata : dict
            Dataset metadata
        meta_per_image : list[dict]
            Metadata for each dataset image
        '''
        # have to do this because h5py requires indices to be in increasing order for some reason
        ags = np.argsort(eval_inds)
        inds_sorted = [eval_inds[i] for i in ags]
        rd_pred = model_out[ags]

        rd_img = self['rd_img'][inds_sorted].view(complex) # (N, C, D, R)
        rd_targ_only = self['rd_targ_only'][inds_sorted].view(complex)
        metadata = self['metadata']
        meta_per_image = [self['meta_per_image'][i] for i in inds_sorted]
        
        # OUTPUT N, C, R, D
        out_dict = {'rd_img': rd_img.transpose(0, 1, 3, 2),
                    'rd_targ_only': rd_targ_only.transpose(0, 1, 3, 2),
                    'metadata': metadata,
                    'meta_per_image': meta_per_image,
                    }

        sio.savemat(out_file, out_dict)

        return rd_img.transpose(0, 3, 2, 1), rd_pred, rd_targ_only.transpose(0, 3, 2, 1), metadata, meta_per_image

    def close(self):
        pass


class ZScoreStandardize(torch.nn.Module):
    '''Transform for scaling input data
    Computes (x - mean) / std
    '''
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, img, cache=torch.tensor([])):
        '''
        cache : contains a Tensor of shape (2, ...)
        cache[0] == mean
        cache[1] == standard deviation
        '''
        if cache.numel(): # if cache is not empty
            mean, std = cache
        else:
            if self.dim is not None:
                mean = img.mean(dim=self.dim, keepdim=True)
                std = img.std(dim=self.dim, keepdim=True)
            else:
                mean = img.mean()
                std = img.std()
        cache = torch.stack((mean, std), dim=0)
        return (img - mean) / std, cache

    @staticmethod
    def inverse(img, cache):
        mean, std = cache
        return img * std + mean


class MedianNormalize(torch.nn.Module):
    '''Transform for normalizing data by the median
    '''
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, img, cache=torch.tensor([])):
        if self.dim is not None:
            median = torch.median(img.abs(), dim=self.dim, keepdim=True)
        else:
            median = torch.median(img.abs())
        return img / median, cache

class RandomNoiseTransform(torch.nn.Module):
    '''Augmentation that adds Gaussian noise with convolutional taper (blur) at random standard deviations
    '''
    def __init__(self, mag_low=0, mag_high=0):
        super().__init__()
        self.mag_low = mag_low
        self.mag_high = mag_high
    
    def forward(self, img, cache=torch.tensor([])):
        mag = np.random.uniform(self.mag_low, self.mag_high)
        noise = torch.randn(img.shape, dtype=img.dtype) * mag   # (R, D, C)
        noise = noise.unsqueeze(0)  # (1, R, D, C)
        filter = torch.tensor([[0.1186, 0.3443, 0.1186],
                               [0.3443, 1.0000, 0.3443],
                               [0.1186, 0.3443, 0.1186]], dtype=noise.dtype)   # (3, 3)
        filter = filter.unsqueeze(0).unsqueeze(0)   # (1, 1, 3, 3)
        num_chan = img.shape[-1]
        filter = filter.expand(num_chan, 1, 3, 3)    # (6, 1, 3, 3)
        noise = noise.permute(0, 3, 1, 2)   # (1, C, R, D)
        expanded_padding = (1, 1, 1, 1)
        noise_tapered = F.conv2d(F.pad(noise, expanded_padding, mode='circular'), filter, groups=num_chan)
        noise_tapered = noise_tapered.permute(0, 2, 3, 1).squeeze() # (R, D, C)
        return img + noise_tapered, cache


class RollTransform(torch.nn.Module):
    '''Augmentation that randomly shifts data along specified dimension (1=Doppler)
    '''
    def __init__(self, roll_dim=1):
        super().__init__()
        self.roll_dim = roll_dim
        
    def forward(self, img, cache=torch.tensor([])):
        dim_size = img.shape[self.roll_dim]
        if cache.numel():
            shift = cache.item()
        else:
            shift = torch.randint(0, dim_size, (1,)).item()
            cache = torch.tensor(shift)
        return torch.roll(img, shift, dims=self.roll_dim), cache
    
    def inverse(self, img, cache):
        shift = -cache.item()
        return torch.roll(img, shift, dims=self.roll_dim)


def rel_error(x, y):
    """Returns relative error (for debugging)"""
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))