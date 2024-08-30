# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import numpy as np
import torch
import matplotlib.pyplot as plt

def make_color_plots(data, save_path=None, title='', range_window=None, mag_color_range=None, power_color_range=None,
                     normalize_power=False):
    '''Creates four visualizations of range-Doppler chips: absolute magnitude, dB power, sum channel phase, and
    magnitude brightness with color phase

    Parameters
    ----------
    data : numpy.ndarray
        (R, D) or (R, D, C) image to plot
    save_path : str
        Path to save output plot. If None, plt.show() is used and nothing is saved
    title : str
        Plot title
    range_window : int
        Length of range extent to plot, centered at range dim // 2. If None, entire range extent is plotted.
    mag_color_range : tuple[float, float]
        Range of color scale for magnitude plot. If None, automatically determined by matplotlib
    power_color_range : tuple[float, float]
        Range of color scale for power plot. If None, automatically determined by matplotlib
    normalize_power : bool
        If True, power is plotted normalized by its median. Default: False
    '''
    if range_window is None:
        range_window = data.shape[0]
    window_start = data.shape[0]//2 - range_window//2
    range_window_slice = slice(window_start, window_start + range_window)
    
    x_sliced = data[range_window_slice, ...]
    if len(x_sliced.shape) == 3:
        x_sliced = x_sliced.sum(dim=-1)

    fig, ax = plt.subplots(2,2, figsize=(12,10))
    mag = x_sliced.abs()
    if mag_color_range is None:
        mag_color_range = (mag.min(), mag.max())
    im = ax[0,0].imshow(mag, aspect='auto', vmin=mag_color_range[0], vmax=mag_color_range[1])
    plt.colorbar(im, ax=ax[0,0])
    ax[0,0].set_title('Magnitude')

    power = 20*torch.log10(mag)
    if normalize_power:
        power -= power.median()
    if power_color_range is None:
        power_color_range = (power.min(), power.max())
    im = ax[0,1].imshow(power, aspect='auto', vmin=power_color_range[0], vmax=power_color_range[1])
    plt.colorbar(im, ax=ax[0,1])
    ax[0,1].set_title('Power (dB)')

    im = ax[1,0].imshow(x_sliced.angle(), aspect='auto')
    plt.colorbar(im, ax=ax[1,0])
    ax[1,0].set_title('Phase')

    H = _normalize(x_sliced.angle()).numpy()
    L = _normalize(x_sliced.abs(), gain=10).numpy()
    S = L
    RGB = _hls_to_rgb(H, L, S)

    im = ax[1,1].imshow(RGB, aspect='auto')
    plt.colorbar(im, ax=ax[1,1])
    ax[1,1].set_title('Mag Phase Plot')

    fig.suptitle(title)
    plt.subplots_adjust(top=0.93)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

    return mag_color_range, power_color_range


def _normalize(x, gain=1):
    '''Normalizes x to be between 0 and 1, then scales by gain
    '''
    min_ = x.min()
    x_shifted = x - min_
    max_ = x_shifted.max()
    return x_shifted / max_ * gain


def _hls_to_rgb(h, l, s):
    '''Converts HLS (hugh, lightness, and saturation) to RGB colors
    '''
    m2 = np.zeros_like(h)

    m2[l <= 0.5] = (l * (1.0 + s))[l <= 0.5]
    m2[l > 0.5] = (l + s - (l*s))[l > 0.5]

    m1 = 2.0*l - m2
    return np.stack((_v(m1, m2, h+1./3.), _v(m1, m2, h), _v(m1, m2, h-1./3.)), axis=2)


def _v(m1, m2, hue):
    '''Calculates parameter v required for function _hls_to_rgb
    '''
    hue = hue % 1.0
    out = np.zeros_like(hue)
    out[hue < 1./6.] = (m1 + (m2-m1)*hue*6.0)[hue < 1./6.]
    out[(hue >= 1./6.) & (hue < 0.5)] = m2[(hue >= 1./6.) & (hue < 0.5)]
    out[(hue >= 0.5) & (hue < 2./3.)] = (m1 + (m2-m1)*(2./3. - hue)*6.0)[(hue >= 0.5) & (hue < 2./3.)]
    return out