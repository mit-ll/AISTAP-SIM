# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
import h5py
import pymatreader
from typing import List
from scipy.io.matlab import matfile_version


class LazyMatfileReader(h5py.File):
    """
    This file reader reads Matlab version 7.3 files such that the arrays can
    be interrogated in a "lazy" way and the regular variables are immediately
    available.

    Example
    -------
    >>> from aistap_sim.utils import LazyMatfileReader
    >>> import numpy as np
    >>> lmfr = LazyMatfileReader('tmp.mat', ignore_fields=['params_monte'])
    >>> print(lmfr['rd_img'].shape)
    (6, 64, 128, 16)
    >>> rd_vals = lmfr['rd_img'][:,:,:,1]
    >>> rd_vals = np.array(vals.view(complex))
    >>> print(rd_vals.mean())
    (-0.009556380334237224+0.005226895825978327j)
    """

    def __init__(self, matfile: str,
                 array_names: List[str] = ['rd_img', 'rd_targ_only'],
                 ignore_fields: List[str] = [],
                 pre_load: bool = False) -> None:
        """
        Creates file-like object that instantly returns most variables
        from a version 7.3 Matlab matfile, which is built on HDF5, but
        uses h5py interfaces to allow lazy loading of large arrays.

        Parameters
        ----------
        matfile : str
            The filename of the Matlab .mat file
        array_names : List[str], optional
            The specific array names to defer loading.
            Defaults are ['rd_img', 'rd_targ_only']
        ignore_fields : List[str], optional
            Specific variable names to never load to save memory/time
            Default is an empty list
        pre_load : bool, optional
            If True, load all data into memory immediately (not "lazy")
            Default is False
        """

        # Ensure that we're passed a version 7.3 matfile
        major_version, _ = matfile_version(matfile)
        if major_version != 2:
            raise NotImplementedError("Only Matlab v7.3 files are supported")

        # Remember the array names for later
        self.array_names = array_names.copy()
        self.ignore_fields = ignore_fields.copy()

        # Grab file handles, using h5py for lazy array loading
        self.h5_handle = h5py.File(matfile, 'r')
        pymat_ignored = self.array_names.copy()
        if ignore_fields is not None:
            pymat_ignored += ignore_fields
        self.mat73_handle = pymatreader.read_mat(matfile,
                                                 ignore_fields=pymat_ignored)
        
        self.pre_load = pre_load

    def __getitem__(self, key):
        """
        Return a view into the field from the proper file handle for lazy loading
        """
        if key in self.ignore_fields:
            raise KeyError("This key was ignored; remove it from ignore_fields")
        elif key in self.array_names:
            if self.pre_load:
                return self.h5_handle[key][:]
            else:
                return self.h5_handle[key]
        elif key in self.mat73_handle.keys():  # type: ignore
            return self.mat73_handle[key]  # type: ignore
        else:
            raise KeyError("Unknown key")

    def keys(self):
        """
        Returns all available keys in the matfile
        """
        k1 = set(self.h5_handle.keys())
        if '#refs#' in k1:
            k1.remove('#refs#')
        k2 = set(self.mat73_handle.keys())  # type:ignore
        all_keys = k1.union(k2)
        for field in self.ignore_fields:
            if field in all_keys:
                all_keys.remove(field)

        return all_keys
