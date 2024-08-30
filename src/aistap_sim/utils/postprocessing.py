# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import os
import numpy as np
import pandas as pd
import warnings
import scipy.io as sio
from scipy.signal import convolve as conv
import copy
import glob
import yaml
import matplotlib.pyplot as plt
import pymatreader
from natsort import natsorted


def createDir(dirpath):
    dirpath = os.path.normpath(dirpath)
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except OSError:
            pass
    return dirpath


def db(x):
    return 20*np.log10(np.abs(x))


def calcfftinds(N):
    hw = np.ceil(N/2)
    t = np.reshape(np.arange(-hw, hw), [N, 1])/N
    return t


def get_target_mask(img_shape, targ_locs, range_win, dop_win):
    '''Computes a boolean mask of shape img_shape over targets specified by targ_locs. Regions around
    targets are set as True, everywhere else is False.
    
    Parameters
    ----------
    img_shape : tuple[int]
        Shape of mask in (N, R, D) format
    targ_locs : numpy.ndarray
        Target locations matrix, shape (N, max_num_targs, 2). max_num_targs is the maximum number of targets
        that can appear in a single image. The last dimension contains the (range, Doppler) coordinate. Rows of 
        targ_locs for images that contain fewer targets than max_num_targs should be padded with NaNs.
    range_win : int
        Width of mask window in range dimension
    dop_win : int
        Width of mask window in Doppler dimension

    Returns
    -------
    mask : numpy.ndarray
        Boolean mask of targets, shape (N, R, D)
    '''

    if (range_win * dop_win) % 2 == 0:
        raise ValueError('range_win and dop_win must both be odd integers')

    mask = np.zeros(img_shape, dtype=bool)
    if np.size(targ_locs) == 0:
        return mask
    locs_int = np.round(targ_locs)

    # zero-centered coordinate grid, shape (2, range_win*dop_win)
    grid = np.mgrid[0:range_win, 0:dop_win].reshape(2, -1) - np.array([range_win//2, dop_win//2]).reshape(-1,1)

    locs_int = np.expand_dims(locs_int, axis=2) # (N, max_num_targs, 1, 2)
    grid = np.expand_dims(np.expand_dims(grid.T, 0), 0) # (1, 1, range_win*dop_win, 2)

    coords = locs_int + grid    # broadcast target coordinates with grid mask shape
    coords = coords.reshape(coords.shape[0], -1, coords.shape[-1]) 

    # i = image index
    # j = range coordinate
    # k = doppler coordinate
    coords_i = np.broadcast_to(np.arange(coords.shape[0]).reshape(-1, 1), coords.shape[0:2]).ravel()
    coords_j = coords[:, :, 0].ravel()
    coords_k = coords[:, :, 1].ravel()

    # find and remove coordinates out of range or nans
    to_delete_j = (coords_j < 0) | (coords_j >= img_shape[1]) | np.isnan(coords_j)
    to_delete_k = (coords_k < 0) | (coords_k >= img_shape[2]) | np.isnan(coords_k)
    to_delete = to_delete_j | to_delete_k
    coords_i = coords_i[~to_delete].astype(int)
    coords_j = coords_j[~to_delete].astype(int)
    coords_k = coords_k[~to_delete].astype(int)
    
    mask[coords_i, coords_j, coords_k] = True

    return mask


def get_masked_values(img, targ_locs, range_win, dop_win):
    '''Returns values that would be indexed by boolean mask of get_target_mask, but split across an extra
    target dimension
    
    Parameters
    ----------
    img : numpy.ndarray
        (N, R, D) matrix to be indexed
    targ_locs : numpy.ndarray
        See get_target_mask
    range_win : int
        Width of mask window in range dimension
    dop_win : int
        Width of mask window in Doppler dimension

    Returns
    -------
    vals : numpy.ndarray
        Masked values of img. Each row corresponds to a single target
    '''

    if (range_win * dop_win) % 2 == 0:
        raise ValueError('range_win and dop_win must both be odd integers')
    
    if np.size(targ_locs) == 0:
        return np.array([])

    locs_int = np.round(targ_locs)

    # zero-centered coordinate grid, shape (2, range_win*dop_win)
    grid = np.mgrid[0:range_win, 0:dop_win].reshape(2, -1) - np.array([range_win//2, dop_win//2]).reshape(-1,1)

    locs_int = np.expand_dims(locs_int, axis=2) # (N, max_num_targs, 1, 2)
    grid = np.expand_dims(np.expand_dims(grid.T, 0), 0) # (1, 1, range_win*dop_win, 2)

    coords = locs_int + grid    # broadcast target coordinates with grid mask shape

    # i = image index
    # j = range coordinate
    # k = doppler coordinate
    coords_i = np.broadcast_to(np.arange(coords.shape[0]).reshape(-1, 1, 1), coords.shape[0:3]).reshape(np.prod(coords.shape[0:2]), -1).copy()
    coords_j = coords[:, :, :, 0].reshape(np.prod(coords.shape[0:2]), -1)
    coords_k = coords[:, :, :, 1].reshape(np.prod(coords.shape[0:2]), -1)

    # find and replace coordinates out of range with zeros temporarily, then fill with nans after indexing
    to_delete_j = (coords_j < 0) | (coords_j >= img.shape[1])
    to_delete_k = (coords_k < 0) | (coords_k >= img.shape[2])
    nanmask = to_delete_j | to_delete_k
    coords_i[nanmask] = 0
    coords_j[nanmask] = 0
    coords_k[nanmask] = 0
    
    # remove entire rows of nans from input
    rows_keep = ~np.isnan(coords_j).all(axis=1)
    coords_i = coords_i[rows_keep].astype(int)
    coords_j = coords_j[rows_keep].astype(int)
    coords_k = coords_k[rows_keep].astype(int)
    nanmask = nanmask[rows_keep]

    vals = img[coords_i, coords_j, coords_k]

    vals[nanmask] = np.nan
    
    return vals


def pdf_to_cdf(dist):
    '''Calculates cumulative distribution function from probability density function
    for use with ROC curves
    '''
    tp = np.cumsum(dist)
    tp_sum = tp[-1]
    tp = tp / tp_sum
    tp = 1.0 - tp
    tp[tp < 1e-12] = 0 # threshold for numerical errors
    return tp
    

def calc_probs(fa, det, num_bins):
    '''Calculates probability density function of false alarms and detections
    
    Parameters
    ----------
    fa : numpy.ndarray or list
        False alarm values (all data points not masked by get_target_mask)
    det : numpy.ndarray or list
        Detection (true positive) values from get_masked_values
    num_bins : int
        Number of bins for histogram
        
    Returns
    -------
    pfa : numpy.ndarray
        Probability of false alarm
    pd : numpy.ndarray
        Probability of detection
    center : numpy.ndarray
        Center of each histogram bin
    '''
    all_vals = np.concatenate((fa, det))
    min_val = np.min(all_vals)
    max_val = np.max(all_vals)
    fa_hist, bins = np.histogram(fa, bins=num_bins, range=(min_val, max_val))
    det_hist, _ = np.histogram(det, bins=num_bins, range=(min_val, max_val))
    width = bins[1] - bins[0]
    center = (bins[:-1] + bins[1:]) / 2
    pfa = fa_hist / (np.sum(fa_hist) * width)
    pd = det_hist / (np.sum(det_hist) * width)
    return pfa, pd, center


def calc_sinr_loss(det, targ_locs_dop, targ_locs_xr, Ndop, delta_rr, bin_size):
    '''Calculates SINR loss statistic
    '''
    pix_axis = np.arange(-np.floor(Ndop/2), np.ceil(Ndop/2))

    targ_locs_v = targ_locs_dop - targ_locs_xr
    targ_locs_v = targ_locs_v[~np.isnan(targ_locs_v)]
    bin_i = np.round(targ_locs_v / bin_size).astype(int)

    min_bin_i = int(np.round((pix_axis[0] - 0.5) / bin_size))
    max_bin_i = int(np.round((pix_axis[-1] + 0.5) / bin_size))

    num_bins = max_bin_i - min_bin_i + 1
    rr_axis = np.arange(-np.floor(num_bins/2), np.ceil(num_bins/2)) * bin_size * delta_rr

    indices = bin_i - min_bin_i - 1
    pow_vs_dop = np.bincount(indices, weights=det, minlength=num_bins)
    num_vs_dop = np.bincount(indices, minlength=num_bins) * 1.0
    num_vs_dop[num_vs_dop == 0] = np.nan  # Avoid division by zero

    SINR = 20 * np.log10(pow_vs_dop / num_vs_dop)
    # SINR = pd.Series(SINR).fillna(method='ffill').fillna(method='bfill').rolling(3, min_periods=1).mean().to_numpy()
    fillmissing = pd.Series(SINR).rolling(3, min_periods=1, center=True).mean().to_numpy()
    SINR[np.isnan(SINR)] = fillmissing[np.isnan(SINR)]

    norm_val = np.nanmedian(SINR)

    SINR[np.isnan(SINR)] = norm_val
    SINRloss = SINR - norm_val

    return SINR, SINRloss, rr_axis
    
    
def norm_rms(img, targ_mask, filt_size, guard_size, min_mse=None):
    '''Performs adaptive CFAR normalization. Output data are Z-scores with respect to their
    surroundings, specified by filt_size. Targets are masked (excluded) by targ_mask.
    
    Parameters
    ----------
    img : numpy.ndarray
        (N, R, D) image to be normalized
    targ_mask : np.ndarray
        Target mask from get_target_mask. If None, no mask is used
    filt_size : tuple[int]
        (Range, Doppler) shape of filter. This should be larger than the target mask size.
    guard_size : int
        Square size of guard region of filter set to zero
    min_mse : float
        Enforces minimum mean squared error value to avoid very small denominators

    Returns
    -------
    img_out : numpy.ndarray
        Normalized output image
    '''

    
    if isinstance(filt_size, int):
        filt = np.ones((1, filt_size, filt_size))
    elif isinstance(filt_size, tuple):
        filt = np.ones((1, *filt_size))
    else:
        raise TypeError('filt_size must be either int or tuple')

    if guard_size > 0:
        zero_ind = np.mgrid[:guard_size, :guard_size] - guard_size//2 + np.array(filt.shape[1:], ndmin=3).T//2
        filt[0, zero_ind[0], zero_ind[1]] = 0
    
    if targ_mask is None:
        mask = np.ones_like(img)
        img_use = img
    else:
        mask = 1-targ_mask*1.0
        img_use = img * (1-targ_mask*1.0)
    
    N_meas = conv(mask, filt, mode='same')
    mean_meas = conv(img_use, filt, mode='same') / N_meas
    sq_meas = conv((img_use - mean_meas)**2, filt, 'same')
    mse_meas = np.sqrt(sq_meas / (N_meas - 1))
    if min_mse is not None:
        mse_meas = np.clip(mse_meas, a_min=min_mse, a_max=None)
    img_out = (img-mean_meas) / mse_meas

    return img_out


def do_stap(rd_img, svec_ramp=None, std_thresh=2.8):
    '''Performs STAP (Space-Time Adaptive Processing)
    
    Parameters
    ----------
    rd_img : numpy.ndarray
        Input (N, R, D, C)
    svec_ramp : numpy.ndarray
        Steering vector ramp
    std_thresh : float
        Threshold for exclusion of potential targets from covariance sample
    
    Returns
    -------
    img_null_w : numpy.ndarray
        STAP output
    '''
    
    rd_img_dop = rd_img.transpose(0, 2, 1, 3) # (N, D, R, C)

    # threshold and exclude potential targets
    thresh = np.mean(db(rd_img_dop), axis=(2, 3), keepdims=True) + std_thresh * np.std(db(rd_img_dop), axis=(2, 3), ddof=1, keepdims=True)
    inds_remove_R = np.sum(db(rd_img_dop) > thresh, axis=3) != 0
    rd_img_thresh = copy.copy(rd_img_dop)
    rd_img_thresh[inds_remove_R] = 0

    rd_img_1 = rd_img_thresh.transpose(0, 1, 3, 2) # (N, D, C, R)
    rd_img_2 = rd_img_thresh.conj()

    # per-doppler covariance with diagonal loading for invertibility, (N, D, C, C)
    R_hat = rd_img_1 @ rd_img_2 + np.max(np.abs(rd_img_dop), axis=(2, 3), keepdims=True) * np.eye(rd_img_dop.shape[3]) * 0.01

    if svec_ramp is None:
        svec_ramp = np.zeros((rd_img.shape[0], 1, 1, 1))
    else:
        svec_ramp = np.expand_dims(svec_ramp, (1, 2))
    
    svec = calcfftinds(R_hat.shape[-1]) * R_hat.shape[-1]
    svec = svec.reshape(1, 1, -1, 1) * svec_ramp
    svec = np.exp(1j * svec) # (N, 1, C, sv)
    
    R_inv = np.linalg.inv(R_hat)
    w = R_inv @ svec # (N, D, C, sv)
    rdaimg_nulled = rd_img_dop.conj() @ w # (N, D, R, sv)

    rdaimg_nulled = rdaimg_nulled.transpose(0, 2, 1, 3) # (N, R, D, sv)

    # whiten output across range dim
    rdaimg_null_w = rdaimg_nulled / np.nanmedian(np.abs(rdaimg_nulled), axis=1, keepdims=True)
    
    # optimize steering vectors
    img_null_w = np.nanmax(np.abs(rdaimg_null_w), axis=3)

    return img_null_w
    

def get_targ_coords(meta_per_image, rd_img_shape):
    '''Extracts target coordinates from format in meta_per_image
    '''
    # rd_img_shape: (N, R, D, C)
    targ_dop = [x['targ_pix_dop'] for x in meta_per_image]
    targ_range = [x['targ_pix_range'] for x in meta_per_image]
    targ_xr = [x['targ_pix_xr'] for x in meta_per_image]

    midp_dop = rd_img_shape[2] // 2
    midp_range = rd_img_shape[1] // 2

    max_num_targs = max([len(x) for x in targ_dop])
    true_dop, true_range, true_xr = np.ones((3, rd_img_shape[0], max_num_targs)) * np.nan
    for i in range(rd_img_shape[0]):
        true_dop[i, 0:len(targ_dop[i])] = targ_dop[i] + midp_dop
        true_range[i, 0:len(targ_range[i])] = targ_range[i] + midp_range
        true_xr[i, 0:len(targ_xr[i])] = targ_xr[i] + midp_dop

    return true_range, true_dop, true_xr


def save_sinr_loss_roc(rd_img, rd_pred, metadata, meta_per_image, save_dir='.', roc_mask_shape=(3,3), norm_mask_shape=(3,3), norm_filt_shape=(5,5),
                       norm_guard_size=0, num_bins=1000):
    '''Performs postprocessing to calculate ROC curves and SINR loss statistics and saves data to filesystem as 'eval_statistics.mat'. 
    Calculates:
        white : image with median whitenining only
        pred : output of neural network model
        STAP : STAP output
        STAP_SV : STAP output with steering vector optimizations

    Parameters
    ----------
    rd_img : numpy.ndarray
        Original data for comparison with model output, dimension (N, R, D, C)
    rd_pred : numpy.ndarray
        Output of neural network model, dimension (N, R, D, C)
    metadata : dict
        Dataset metadata
    meta_per_image : list[dict]
        Metadata applicable to each image
    save_dir : str
        Path to directory in which to save output statistics

    Returns
    -------
    out_dict : dict
        Copy of dictionary saved to disk as 'eval_statistics.mat'
    '''

    range_center = np.array([x['range_center'] for x in meta_per_image])
    xrange_step = np.array([x['xrange_step'] for x in meta_per_image])

    antenna_spacing_ang = metadata['channel_spacing'] / range_center
    antenna_spacing_phi = 2*np.pi / metadata['lambda_c'] * antenna_spacing_ang * xrange_step

    ramp_lengths = np.ceil(1/(antenna_spacing_phi*1.2)).astype(int) * 2
    svec_ramp = np.ones((len(antenna_spacing_phi), np.max(ramp_lengths))) * np.nan
    for i in range(len(antenna_spacing_phi)):
        sr = np.arange(0, 1, antenna_spacing_phi[i]*1.2)
        svec_ramp[i, 0:ramp_lengths[i]] = np.concatenate((-np.flip(sr), sr))

    rd_STAP = db(do_stap(rd_img))
    rd_STAP_SV = db(do_stap(rd_img, svec_ramp=svec_ramp))

    rd_img_sum = np.sum(rd_img, axis=3)
    rd_white = db(rd_img_sum / np.median(np.abs(rd_img_sum), axis=1, keepdims=True))

    if len(rd_pred.shape) > 3:
        rd_pred_sum = np.sum(rd_pred, axis=3)
    else:
        rd_pred_sum = rd_pred
    rd_pred_w = db(rd_pred_sum / np.median(np.abs(rd_pred_sum), axis=1, keepdims=True))

    true_range, true_dop, true_xr = get_targ_coords(meta_per_image, rd_img.shape)
    targ_coords = np.stack((true_range, true_dop), axis=-1)
    
    if norm_mask_shape is not None:
        norm_mask = get_target_mask(rd_white.shape, targ_coords, range_win=norm_mask_shape[0], dop_win=norm_mask_shape[1])
    else:
        norm_mask = None

    if norm_filt_shape is not None:
        rd_white_norm = norm_rms(rd_white, norm_mask, norm_filt_shape, norm_guard_size)
        rd_pred_norm = norm_rms(rd_pred_w, norm_mask, norm_filt_shape, norm_guard_size)
        rd_STAP_norm = norm_rms(rd_STAP, norm_mask, norm_filt_shape, norm_guard_size)
        rd_STAP_SV_norm = norm_rms(rd_STAP_SV, norm_mask, norm_filt_shape, norm_guard_size)
    else:
        rd_white_norm = rd_white
        rd_pred_norm = rd_pred_w
        rd_STAP_norm = rd_STAP
        rd_STAP_SV_norm = rd_STAP_SV
    
    roc_mask = get_target_mask(rd_white.shape, targ_coords, range_win=roc_mask_shape[0], dop_win=roc_mask_shape[1])

    fa_white = rd_white_norm[~roc_mask]
    fa_pred = rd_pred_norm[~roc_mask]
    fa_STAP = rd_STAP_norm[~roc_mask]
    fa_STAP_SV = rd_STAP_SV_norm[~roc_mask]

    det_white = np.nanmax(get_masked_values(rd_white_norm, targ_coords, range_win=roc_mask_shape[0], dop_win=roc_mask_shape[1]), axis=1)
    det_pred = np.nanmax(get_masked_values(rd_pred_norm, targ_coords, range_win=roc_mask_shape[0], dop_win=roc_mask_shape[1]), axis=1)
    det_STAP = np.nanmax(get_masked_values(rd_STAP_norm, targ_coords, range_win=roc_mask_shape[0], dop_win=roc_mask_shape[1]), axis=1)
    det_STAP_SV = np.nanmax(get_masked_values(rd_STAP_SV_norm, targ_coords, range_win=roc_mask_shape[0], dop_win=roc_mask_shape[1]), axis=1)

    out_dict = {}
    
    out_dict['pfa_white'], out_dict['pd_white'], out_dict['bins_white'] = calc_probs(fa_white, det_white, num_bins)
    out_dict['pfa_pred'], out_dict['pd_pred'], out_dict['bins_pred'] = calc_probs(fa_pred, det_pred, num_bins)
    out_dict['pfa_STAP'], out_dict['pd_STAP'], out_dict['bins_STAP'] = calc_probs(fa_STAP, det_STAP, num_bins)
    out_dict['pfa_STAP_SV'], out_dict['pd_STAP_SV'], out_dict['bins_STAP_SV'] = calc_probs(fa_STAP_SV, det_STAP_SV, num_bins)

    c=299792458
    # For NOW assume that the CPI is constant so the delta_rr is constant.  IT could vary someday.
    CPI = meta_per_image[0]['CPI']
    delta_rr = c/metadata['fc']/2*(1/CPI)
    delta_pix = 1.2
    out_dict['SINR_white'], out_dict['SINR_loss_white'], out_dict['rr_axis'] = calc_sinr_loss(det_white, true_dop, true_xr, rd_img.shape[2], delta_rr, delta_pix)
    out_dict['SINR_pred'], out_dict['SINR_loss_pred'], _ = calc_sinr_loss(det_pred, true_dop, true_xr, rd_img.shape[2], delta_rr, delta_pix)
    out_dict['SINR_STAP'], out_dict['SINR_loss_STAP'], _ = calc_sinr_loss(det_STAP, true_dop, true_xr, rd_img.shape[2], delta_rr, delta_pix)
    out_dict['SINR_STAP_SV'], out_dict['SINR_loss_STAP_SV'], _ = calc_sinr_loss(det_STAP_SV, true_dop, true_xr, rd_img.shape[2], delta_rr, delta_pix)

    sio.savemat(os.path.join(save_dir, 'eval_statistics.mat'), out_dict)

    return out_dict


def plot_sinr_loss_roc(jobdir, overrides_of_interest, figsize=(8,6), save_dpi=300):
    '''Plots data from save_sinr_loss_roc. Iterates over subdirectories in jobdir, expects
    subdirectory names to be in format job_0, job_1, etc. Saves figures in jobdir.
    
    Parameters
    ----------
    jobdir : str
        Path to root job directory containing multiple subjobs
    overrides_of_interest : list[str]
        List of hydra-zen config override names that will be plotted to identify data
    figsize : tuple[int]
        Size of matplotlib figure
    save_dpi : int
        DPI of saved figures
    '''
    subjobs = natsorted(glob.glob(os.path.join(jobdir, 'job_*')))
    fig_ROC_all, ax_ROC_all = plt.subplots(figsize=figsize)
    fig_loss_all, ax_loss_all = plt.subplots(figsize=figsize)
    fig_pdplot, ax_pdplot = plt.subplots(figsize=figsize, tight_layout=True)
    min_y = 1
    pdplot_data = []
    for subdir in subjobs:
        try:
            plot_data = pymatreader.read_mat(os.path.join(subdir, 'eval_statistics.mat'))
            loss_data = pymatreader.read_mat(os.path.join(subdir, 'loss.mat'))
        except FileNotFoundError:
            job_id = os.path.basename(subdir)
            warnings.warn(f'Skipping {job_id}, missing data')
            continue
            
        with open(os.path.join(subdir, '.hydra', 'overrides.yaml'), 'r') as file:
            overrides = yaml.safe_load(file)
        overrides = {pair.split('=')[0]:pair.split('=')[1] for pair in overrides}
        # Form override string label
        tStr = ''
        for key in overrides:
            if key in overrides_of_interest:
                tStr += key.split('.')[-1] + ': ' + overrides[key] + ', '
        if len(tStr) == 0:
            tStr = 'aistap'
        else:
            tStr = tStr[:-2]
        fa_pred, det_pred, fa_STAP_SV, det_STAP_SV = plot_individual(subdir, plot_data, tStr, figsize, save_dpi)
        dat = ax_ROC_all.semilogx(fa_pred, det_pred, label=tStr)[0]

        xlim = ax_ROC_all.get_xlim()[0]
        ydata = dat.get_ydata()[np.where(dat.get_xdata() > xlim)]
        min_y = np.min([np.min(ydata), min_y])
        min_y_disp = 1 - 1.05*(1-min_y)
        
        ax_loss_all.plot(loss_data['train_loss_x'], loss_data['train_loss_y'], label=tStr)
        
        pdplot_data.append({'ovr': tStr, '1e-2': det_pred[np.argmin(np.abs(fa_pred - 1e-2))],
                            '1e-4': det_pred[np.argmin(np.abs(fa_pred - 1e-4))],
                            '1e-6': det_pred[np.argmin(np.abs(fa_pred - 1e-6))],
                            'stap': det_STAP_SV[np.argmin(np.abs(fa_STAP_SV - 1e-4))]})

    ax_ROC_all.grid()
    ax_ROC_all.set_ylim([min_y_disp, 1])
    ax_ROC_all.set_xlabel('PFA')
    ax_ROC_all.set_ylabel('PD')
    ax_ROC_all.set_title('All ROC curves')
    ax_ROC_all.legend(loc='best')
    fig_ROC_all.savefig(os.path.join(jobdir, 'ROC_all.png'), dpi=save_dpi)
    plt.close(fig_ROC_all) 

    ax_loss_all.grid()
    ax_loss_all.set_title('All loss curves')
    ax_loss_all.set_ylabel('Training Loss')
    ax_loss_all.set_xlabel('Epoch')
    ax_loss_all.legend(loc='best')
    fig_loss_all.savefig(os.path.join(jobdir, 'loss_all.png'), dpi=save_dpi)
    plt.close(fig_loss_all) 

    xticks = [p['ovr'] for p in pdplot_data]
    en2 = np.array([p['1e-2'] for p in pdplot_data])
    en4 = np.array([p['1e-4'] for p in pdplot_data])
    en6 = np.array([p['1e-6'] for p in pdplot_data])
    stap = np.array([p['stap'] for p in pdplot_data])

    # ax_pdplot.bar(range(len(en4)), en4, align='center')
    ax_pdplot.plot(range(len(stap)), stap, linewidth=2, label='STAP SV')
    ax_pdplot.plot(range(len(en4)), en4, linewidth=2, label='AISTAP')
    eb = ax_pdplot.errorbar(range(len(en4)), en4, yerr=(en4-en6, en2-en4), fmt='none', ecolor='black', capsize=5)
    eb[-1][0].set_linestyle('--')
    ax_pdplot.set_xticks(range(len(xticks)))
    ax_pdplot.set_xticklabels(xticks, rotation=45)
    ax_pdplot.set_ylim([0.5, 1])
    ax_pdplot.grid()
    ax_pdplot.set_ylabel('PD at 10^-4 PFA')
    ax_pdplot.set_title('PD vs Job')
    ax_pdplot.legend(loc='best')
    fig_pdplot.savefig(os.path.join(jobdir, 'PD_vs_job.png'), dpi=save_dpi)
    plt.close(fig_pdplot)
        
        
def plot_individual(subdir, plot_data, ovr_string='', figsize=(8, 6), save_dpi=300):
    '''Plots results from individual evaluation runs
    
    Parameters
    ----------
    subdir : str
        Directory in which to save plots
    plot_data : dict
        Evaluation statistics to plot, created from save_sinr_loss_roc
    ovr_string : str
        String of overrides to use as plot titles
    figsize : tuple[int]
        Size of matplotlib figure
    save_dpi : int
        DPI of saved figures

    Returns
    -------
    fa_pred_cdf : numpy.ndarray
        Cumulative distribution of false alarm probability from neural network output
    det_pred_cdf : numpy.ndarray
        Cumulative distribution of detection probability from neural network output
    fa_STAP_SV_cdf : numpy.ndarray
        Cumulative distribution of false alarm probability from STAP with steering vector optimizations
    det_STAP_SV_cdf : numpy.ndarray
        Cumulative distribution of detection probability from STAP with steering vector optimizations
    '''
    fig_pdfs, ax_pdfs = plt.subplots(figsize=figsize)
    fig_cdfs, ax_cdfs = plt.subplots(figsize=figsize)
    fig_ROC, ax_ROC = plt.subplots(figsize=figsize)
    fig_SINR, ax_SINR = plt.subplots(figsize=figsize)
    fig_SINR_loss, ax_SINR_loss = plt.subplots(figsize=figsize)

    if len(ovr_string) > 0:
        ovr_string = ', ' + ovr_string

    # exponential weighted moving average (for smoothing)
    def ewma(data, alpha=0.1):
        return pd.Series(data).ewm(alpha=alpha).mean().to_numpy()

    ax_pdfs.plot(plot_data['bins_white'], plot_data['pfa_white'], '--k', label='Baseline Pfa')
    ax_pdfs.plot(plot_data['bins_white'], ewma(plot_data['pd_white']), '-k', label='Baseline Pd')
    ax_pdfs.plot(plot_data['bins_pred'], plot_data['pfa_pred'], '--C1', label='AISTAP Pfa')
    ax_pdfs.plot(plot_data['bins_pred'], ewma(plot_data['pd_pred']), '-C1', label='AISTAP Pd')
    ax_pdfs.plot(plot_data['bins_STAP'], plot_data['pfa_STAP'], '--C0', label='STAP Pfa')
    ax_pdfs.plot(plot_data['bins_STAP'], ewma(plot_data['pd_STAP']), '-C0', label='STAP Pd')
    ax_pdfs.plot(plot_data['bins_STAP_SV'], plot_data['pfa_STAP_SV'], '--C2', label='STAP SV Pfa')
    ax_pdfs.plot(plot_data['bins_STAP_SV'], ewma(plot_data['pd_STAP_SV']), '-C2', label='STAP SV Pd')

    ax_pdfs.grid()
    ax_pdfs.set_title('Probability Distributions' + ovr_string)
    ax_pdfs.legend(loc='best')
    fig_pdfs.savefig(os.path.join(subdir, 'pdfs.png'), dpi=save_dpi)
    plt.close(fig_pdfs) 

    fa_white_cdf = pdf_to_cdf(plot_data['pfa_white'])
    det_white_cdf = pdf_to_cdf(plot_data['pd_white'])
    fa_pred_cdf = pdf_to_cdf(plot_data['pfa_pred'])
    det_pred_cdf = pdf_to_cdf(plot_data['pd_pred'])
    fa_STAP_cdf = pdf_to_cdf(plot_data['pfa_STAP'])
    det_STAP_cdf = pdf_to_cdf(plot_data['pd_STAP'])
    fa_STAP_SV_cdf = pdf_to_cdf(plot_data['pfa_STAP_SV'])
    det_STAP_SV_cdf = pdf_to_cdf(plot_data['pd_STAP_SV'])

    ax_cdfs_log = ax_cdfs.twinx()
    ax_cdfs_log.set_yscale('log')

    ax_cdfs_log.plot(plot_data['bins_white'], fa_white_cdf, '--k', label='Baseline Pfa')
    ax_cdfs.plot(plot_data['bins_white'], det_white_cdf, '-k', label='Baseline Pd')
    ax_cdfs_log.plot(plot_data['bins_pred'], fa_pred_cdf, '--C1', label='AISTAP Pfa')
    ax_cdfs.plot(plot_data['bins_pred'], det_pred_cdf, '-C1', label='AISTAP Pd')
    ax_cdfs_log.plot(plot_data['bins_STAP'], fa_STAP_cdf, '--C0', label='STAP Pfa')
    ax_cdfs.plot(plot_data['bins_STAP'], det_STAP_cdf, '-C0', label='STAP Pd')
    ax_cdfs_log.plot(plot_data['bins_STAP_SV'], fa_STAP_SV_cdf, '--C2', label='STAP SV Pfa')
    ax_cdfs.plot(plot_data['bins_STAP_SV'], det_STAP_SV_cdf, '-C2', label='STAP SV Pd')

    ax_cdfs.grid()
    ax_cdfs.set_title('Probability Distributions' + ovr_string)
    ax_cdfs.legend(loc='best')
    fig_cdfs.savefig(os.path.join(subdir, 'cdfs.png'), dpi=save_dpi)
    plt.close(fig_cdfs) 
    
    l1 = ax_ROC.semilogx(fa_white_cdf, det_white_cdf, '-k', label='Baseline', linewidth=2)[0]
    l2 = ax_ROC.semilogx(fa_pred_cdf, det_pred_cdf, '-C1', label='AISTAP', linewidth=2)[0]
    l3 = ax_ROC.semilogx(fa_STAP_cdf, det_STAP_cdf, '-C0', label='STAP', linewidth=2)[0]
    l4 = ax_ROC.semilogx(fa_STAP_SV_cdf, det_STAP_SV_cdf, '--C0', label='STAP SV', linewidth=2)[0]

    # find minimum pd plotted for auto-scaling of y axis limit
    xlim = ax_ROC.get_xlim()[0]
    ydata = [l.get_ydata()[np.where(l.get_xdata() > xlim)] for l in [l1, l2, l3, l4]]
    min_y = np.min([np.min(yd) for yd in ydata])
    min_y = 1 - 1.05*(1-min_y)

    ax_ROC.grid()
    ax_ROC.set_ylim([min_y, 1])
    ax_ROC.set_xlabel('PFA')
    ax_ROC.set_ylabel('PD')
    ax_ROC.set_title('ROC Curve' + ovr_string)
    ax_ROC.legend(loc='best')
    fig_ROC.savefig(os.path.join(subdir, 'ROC.png'), dpi=save_dpi)
    plt.close(fig_ROC) 

    ax_SINR.plot(plot_data['rr_axis'], plot_data['SINR_white'], '-k', label='Baseline')
    ax_SINR.plot(plot_data['rr_axis'], plot_data['SINR_pred'], '-C1', label='AISTAP')
    ax_SINR.plot(plot_data['rr_axis'], plot_data['SINR_STAP'], '-C0', label='STAP')
    ax_SINR.plot(plot_data['rr_axis'], plot_data['SINR_STAP_SV'], '--C0', label='STAP SV')

    ax_SINR.grid()
    ax_SINR.set_xlabel('Radial Velocity (m/s)')
    ax_SINR.set_ylabel('SINR (dB)')
    ax_SINR.set_title('SINR' + ovr_string)
    ax_SINR.legend(loc='lower right')
    fig_SINR.savefig(os.path.join(subdir, 'SINR.png'), dpi=save_dpi)
    plt.close(fig_SINR) 

    ax_SINR_loss.plot(plot_data['rr_axis'], plot_data['SINR_loss_white'], '-k', label='Baseline')
    ax_SINR_loss.plot(plot_data['rr_axis'], plot_data['SINR_loss_pred'], '-C1', label='AISTAP')
    ax_SINR_loss.plot(plot_data['rr_axis'], plot_data['SINR_loss_STAP'], '-C0', label='STAP')
    ax_SINR_loss.plot(plot_data['rr_axis'], plot_data['SINR_loss_STAP_SV'], '--C0', label='STAP SV')

    ax_SINR_loss.grid()
    ax_SINR_loss.set_xlabel('Radial Velocity (m/s)')
    ax_SINR_loss.set_ylabel('SINR Loss (dB)')
    ax_SINR_loss.set_title('SINR Loss' + ovr_string)
    ax_SINR_loss.legend(loc='lower right')
    fig_SINR_loss.savefig(os.path.join(subdir, 'SINR_loss.png'), dpi=save_dpi)
    plt.close(fig_SINR_loss) 

    return fa_pred_cdf, det_pred_cdf, fa_STAP_SV_cdf, det_STAP_SV_cdf
