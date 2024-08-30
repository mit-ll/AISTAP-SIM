# AISTAP-SIM

This repository holds the dataset that accompanies our IEEE Radar paper, as well
as coding to minimally load and process the dataset. 

## Citation
Using `AISTAP-SIM` for your research? Please cite the following publication:
```
@inproceedings{vega2024radar,
  author={Vega, Dalton and Newey, Michael and Barrett, David and Axelrod, Alan and Myne, Anu and Wollaber, Allan},
  title={STAP-Informed Neural Network for Radar Moving Target Indicator}, 
  booktitle={Proc. {IEEE} Intl. Radar Conf.},
  year={2024},
  month={May},
}
```

# Getting Started
First, get a working copy of the code and data onto your machine:

1) Clone this repository, e.g., `git clone git@github.com:mit-ll/AISTAP-SIM.git`
2) Download the dataset from the [releases](https://github.com/mit-ll/AISTAP-SIM/releases) 

To get started using the data, we are providing a data reader as a python package and a Matlab file. 

## Matlab Instructions
For Matlab, consult the `scripts/sampleread_matlab.m` file for a few examples of loading
data and plotting.

## Python Instructions
Create a virtual environment (using conda, venv, etc.). We tested in conda using python v3.11.
Then, from the project root directory, run:
`pip install .`
Use the `-e` switch if you would like to edit the code in-place.

Finally, run one of the example scripts in the `scripts` folder. For example, from the
root directory, run 

```sh 
python scripts/run_train_example.py
```

# Datasets
The dataset is distributed as Matlab v7.3 formatted `.mat` files. These are also 
parseable as `HDF5`-formatted files. Each dataset is composed of training set, a test
set, and a small "sample" file to get started. 
| Name | Filenames | Description |
| ---- | -------- | ----------- |
| Ground Clutter train set |`simMed/simMed_train??.mat`  <br> `simMed/simMed_test.mat` <br> `simMed/simMed_sample.mat`  |  Standard GMTI dataset with low wind (train, test, sample files)|
| Windy Ground Clutter |`simWind/simWind_train??.mat` <br> `simWind/simWind_test.mat` <br> `simWind/simWind_sample.mat`  |  Same as ground clutter but much higher wind speeds |
| Noise Only | `simNoise/simNoiseOnly_train??.mat` <br> `simNoise/simNoiseOnly_test.mat` <br> `simNoise/simNoiseOnly_sample.mat`  | No ground clutter, only simulated noise | 

Note that the `??` in the training files indicates a numeral (01-16), as the files had to be broken up to fit onto github. 
The data directory structure with sample files can be downloaded in the 
[releases](https://github.com/mit-ll/AISTAP-SIM/releases) tab as filename `sampledata.zip`. 

## Detailed Description
The radar simulator models a two-meter long Ku band antenna with a 50 ms Ground
Moving Target Indicator (GMTI) coherent processing interval (CPI). The range
resolution is set to 100 meters. We built our model on top of a simple baseline
simulation based on the stochastic features of radar clutter and noise. We
modeled both the noise and the clutter with multiplicative speckle noise that
is circularly complex and normally distributed.  To simulate radar
returns, we employ the far-field approxima-tion and assume a non-squinted
(broadside) radar collection geometry.  We also include a small amount of both
magnitude and phase noise.  To challenge STAP, we implemented heterogeneous
clutter. We created a randomized map that covers the imaged region, and
represents the percentage of trees, grass, roads, and buildings in each pixel.
The power for the speckle in each region is set according to measured values
from the a collection over Joint Base Cape Cod and is also hand-tuned.
Motion of trees in the wind relies on empirical modelling and measurement,
and is hand-tuned to match realistic data.  Clutter power is also randomly
modulated in a randomly selected region of the range extent in a subset of the
images. This results in a bimodal distribution of clutter powers across the
image. Targets were injected into the clutter and noise data as ideal point
scatterers.  We created three datasets for evaluation. The first has a 7 pixel
wide clutter ridge, with 30 dB average target SNR and 15 dB average peak
clutter to noise ratio, which we refer to as "Ground Clutter". In a second
dataset, we increased the level of the wind significantly, providing a much
more pronounced blur effect in the image.  Finally, we have a dataset with
Gaussian noise but no clutter called "No Clutter".

For each simulated image, we vary simulation parameters including:
* radar sensor speed 
* number of targets 
* presence and location of non-homogeneous clutter
* noise level
* clutter level

### Target injection
Doppler coordinates were randomly sampled from one of 2 possible distributions:
from one that covered the entire Doppler range uniformly, or from one that was
highly concentrated on the clutter ridge. We tapered all of the data in range
and Doppler with a 35 dB Taylor window to reduce sidelobes. We specified a
range resolution of 100 meters and a cross-range resolution that varied
randomly per image about 140 meters. We simulate a 2 meter antenna with 6
evenly spaced antenna channels at a 50 km standoff from beam center.


###  Dataset Variables and Dimensions
Dimensions are as shown when loading with LazyMatfileReader. 
If loaded with pymatreader, the dimension order is reversed.  
Additionally, the order is reversed when read into Matlab.

The simulated data are nominally in an `NxCxDxR`-sized arrays in which
* `N` = number of images (2048)
* `C` = channel dimension size (6)
* `D` = Doppler dimension size (64)
* `R` = range dimension size (1024)

There are two primary arrays in each file that contain the raw "imagery" data and
the target-only data:
```python
rd_img : (N, C, D, R), complex128 array
   Radar image input data containing clutter, noise, and injected targets

rd_targ_only : (N, C, D, R), complex128 array
   Training labels containing target response only
```

Additionally, there is rich metadata associated with each of these datasets
that are described in their corresponding dictionaries (or Matlab structures),
including `metadata` for the entire file and `meta_per_image` for each 
of the `N` images.  Below is a brief description of each field.
```
metadata : dict
   Dictionary of dataset metadata containing the following fields:

      lambda_c              Center wavelength of radar
      midp_ch               Mid channel cell
      range_nominal         Nominal range to dwell center
      midp_dop              Mid doppler cell
      num_antenna_channels  Number of antenna channels
      channel_spacing       Meters between antenna channels
      fc                    Center frequency of band
      Vs_nominal            Nominal sensor speed
      range_taper           Taper in the range dimension (applied in fft of range)
      midp_range            Mid range cell
      dop_taper             Taper in the Doppler dimension (applied in fft of Doppler)
      antenna_length        Length of the antenna

meta_per_image : list[dict]
   List of dictionaries for metadata pertanining to each image, containing the following fields

      truth_pix_dop_axis    Conversion between targ_pix values and Doppler pixels in the image (e.g. is the first index 1 or 0)
      Ntrue                 Number of targets in the data
      Vs                    Estimated sensor velocity component perpendicular to the range vector to dwell center (m/s).
      CPI                   Coherent processing interval
      range_step            Range pixel spacing (m)
      rr_axis               Range rate axis values
      targ_pix_dop          Doppler of injected targets (in pixel units)
      targ_pix_xr           Cross-range of injected targets (in pixel units)
      truth_pix_range_axis  Conversion between targ_pix values and range pixels in the image (e.g. is the first index 1 or 0)
      rs_axis               Range axis values
      xrange_step           Cross-range pixel spacing (m)
      xr_axis               Cross-range axis values
      dop_beamwidth_pred    Predicted nominal beamwidth of the radar at dwell center in Doppler pixels
      Vstrue                True sensor velocity component perpendicular to the range vector to dwell center (m/s).
      range_center          Range to dwell center (m)
      targ_pix_range        Range of injected targets (in pixel units)
      dop_axis              Doppler axis values
```

# Disclaimer

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

© 2024 MASSACHUSETTS INSTITUTE OF TECHNOLOGY

Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014)
SPDX-License-Identifier: MIT

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering. Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.

The software/firmware is provided to you on an As-Is basis.
