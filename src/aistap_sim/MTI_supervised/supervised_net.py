# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
import torch.nn.functional as F
from aistap_sim.utils.data_utils import view_complex, view_real, gen_tapered_noise_torch


class AISTAP_FC(nn.Module):
    '''Supervised network fully connected in range, implemented as convolutional layers. Only the first
    layer performs convolutions, all subsequent layers are identical to fully connected layers.
    '''
    def __init__(self, input_shape, p=0.1, filter_size=3, norm_output=False):
        '''
        Parameters
        ----------
        input_shape : torch.Tensor.size
            Shape of input tensor. Specified in argument for compatibility with TorchScript compilation
        p : float
            Probability of dropout
        filter_size : int
            Height of convolutional filter in first layer
        norm_output : bool
            If True, each channel vector output is normalized to unit magnitude. Default is False
        '''
        super().__init__()
        self.input_shape = input_shape
        self.norm_output = norm_output
        dop_size = self.input_shape[2]
        in_channels = self.input_shape[-1]
        in_channels *= 2
        mult = dop_size * in_channels // 12
        self.conv1 = nn.Conv2d(in_channels, 12*filter_size*mult, (filter_size, dop_size), stride=1, padding=(filter_size//2,0))
        self.conv2 = nn.Conv2d(12*filter_size*mult, 6*mult, (1, 2), stride=1, padding=0)
        self.conv3 = nn.Conv2d(6*mult, 6*mult, (1, 2), stride=1, padding=0)
        self.conv4 = nn.Conv2d(6*mult, 12*mult, (1, 2), stride=1, padding=0)
        self.drop = nn.Dropout(p)

    def forward(self, x_in, noise_var):
        '''Network forward pass

        Parameters
        ----------
        x_in : torch.Tensor
            4D input tensor of complex data split manually into real 
            and imaginary parts in channel dimension. Shape (N, H, W, C) 
        noise_var : float
            Variance of noise to add to output data (for suppressing sidelobes and other artifacts)
        
        Returns
        -------
        out : torch.Tensor
            Output of neural network
        w : torch.Tensor
            Weight matrix applied to input data x_in
        '''
        # x_in: (N, H, W, C)
        x_orig = x_in
        x = view_real(x_in, chan_dim=-1)
        in_shape_real = x.shape
        x = x.permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = _antirectifier2(x) 
        x = self.drop(x)

        x = self.conv2(x)
        x = _antirectifier2(x)
        x = self.drop(x)

        x = self.conv3(x)
        x = _antirectifier2(x)
        x = self.drop(x)

        x = self.conv4(x)

        x = x.permute(0, 2, 3, 1)
        x = x.view(in_shape_real)
        if self.norm_output:
            x = norm_layer(x, -1)
        w = view_complex(x, chan_dim=-1)
        out = w * x_orig
        if noise_var != 0:
            noise = gen_tapered_noise_torch(out.shape, noise_var, out.device)
            out = out + noise
        return out, w


class AISTAP_CNN(nn.Module):
    '''Fully-convolutional supervised network
    '''
    def __init__(self, input_shape, kernel_size=3, norm_output=False):
        '''
        Parameters
        ----------
        input_shape : torch.Tensor.size
            Shape of input tensor. Specified in argument for compatibility with TorchScript compilation
        kernel_size : int or tuple[int, int]
            Size of convolutional kernel for first 3 layers
        norm_output : bool
            If True, each channel vector output is normalized to unit magnitude. Default is False
        '''
        super().__init__()
        self.input_shape = input_shape
        in_channels = self.input_shape[-1]
        in_channels *= 2

        self.norm_output = norm_output

        mult_out = torch.tensor([2, 4, 8, 1])
        mult_in = mult_out*2

        self.conv1 = nn.Conv2d(in_channels, in_channels*mult_out[0], kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(in_channels*mult_in[0], in_channels*mult_out[1], kernel_size, padding=kernel_size//2)
        self.conv3 = nn.Conv2d(in_channels*mult_in[1], in_channels*mult_out[2], kernel_size, padding=kernel_size//2)

        self.conv1x1_1 = nn.Conv2d(in_channels*mult_in[2], in_channels*mult_out[3], kernel_size=1, padding=0)

        
    def forward(self, x_in, noise_var):
        '''Network forward pass

        Parameters
        ----------
        x_in : torch.Tensor
            4D input tensor of complex data split manually into real 
            and imaginary parts in channel dimension. Shape (N, H, W, C) 
        noise_var : float
            Variance of noise to add to output data (for suppressing sidelobes and other artifacts)
        
        Returns
        -------
        out : torch.Tensor
            Output of neural network
        w : torch.Tensor
            Weight matrix applied to input data x_in
        '''
        x_orig = x_in
        x = view_real(x_in, chan_dim=-1)
        x = x.permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = _antirectifier2(x, cat_dim=1)
        
        x = self.conv2(x)
        x = _antirectifier2(x, cat_dim=1)
        
        x = self.conv3(x)
        x = _antirectifier2(x, cat_dim=1)

        x = self.conv1x1_1(x)

        if self.norm_output:
            x = norm_layer(x, dim=1)
        
        x = x.permute(0, 2, 3, 1)
        w = view_complex(x, chan_dim=-1)
        out = w * x_orig

        if noise_var != 0:
            noise = gen_tapered_noise_torch(out.shape, noise_var, out.device)
            out = out + noise

        return out, w

def _antirectifier(input):
    '''Antirectifier activation function for 2D inputs. Used in place of
    ReLU to preserve positive and negative parts of input. The result is 
    a tensor of samples that are twice as large as the input samples.

    Parameters
    ----------
    input : torch.Tensor
        2D tensor of data of dimension (range, hidden_dim)

    Returns
    -------
    mixed : torch.Tensor
        2D tensor of output data of dimension(range, 2*hidden_dim)
    '''
    scaled_input = input - torch.mean(input, dim=1, keepdim=True)
    normalized = F.normalize(scaled_input, dim=1)
    pos = F.relu(normalized)
    neg = F.relu(-normalized)
    mixed = torch.cat((pos, neg), dim=1)
    return mixed

def _antirectifier2(input, cat_dim=3):
    '''Antirectifier activation function for 4D inputs. Used in place of
    keLU to preserve positive and negative parts of input. The result is 
    a tensor of samples that are twice as large as the input samples. Data
    is concatenated along width dimension for compatibility with convolutional
    layers that mimic fully-connected layers.

    Parameters
    ----------
    input : torch.Tensor
        4D tensor of data of dimension (N, C, H, W)

    Returns
    -------
    mixed : torch.Tensor
        4D tensor of output data of dimension(N, C, H, W*2)
    '''
    scaled_input = input - torch.mean(input, dim=(1,3), keepdim=True)
    normalized = scaled_input / torch.norm(scaled_input, p=2.0, dim=(1,3), keepdim=True).clamp_min(1e-12).expand_as(scaled_input)
    pos = F.relu(normalized)
    neg = F.relu(-normalized)
    mixed = torch.cat((pos, neg), dim=cat_dim)
    return mixed


def norm_layer(input, dim=(1,3)):
    return input / torch.norm(input, p=2.0, dim=dim, keepdim=True).clamp_min(1e-12).expand_as(input)

