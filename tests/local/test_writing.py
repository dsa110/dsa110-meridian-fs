import pkg_resources
from dsamfs.meridian_fringestop import run_fringestopping
from dsacalib.utils import get_autobl_indices
from pyuvdata import UVData
import glob
import numpy as np
from dsamfs.uvh5_utils import uvh5_to_uvfits
from casatasks import importuvfits
import casatools as cc
import os

def test_end2end(tmpdir):
    data_path = pkg_resources.resource_filename('dsamfs', 'data/')
    param_path = '{0}/test_parameters.yaml'.format(data_path)
    header_path = '{0}/test_header.txt'.format(data_path)
    run_fringestopping(param_path, header_file=header_path, outdir=tmpdir)
    fname = glob.glob('{0}/*.hdf5'.format(tmpdir))[0]
    UV = UVData()
    UV.read(fname, file_type='uvh5')
    # Check that the baselines are in the correct order
    nant = UV.Nants_data
    abi = get_autobl_indices(nant, casa=False)
    ant1, ant2 = UV.baseline_to_antnums(UV.baseline_array)
    assert np.all(ant1[abi] == ant2[abi])
    # Check that we can convert to uvfits
    uvh5_to_uvfits(fname)
    assert os.path.exists(fname.replace('hdf5', 'fits'))
    # Check that we can read in the uvfits file
    importuvfits(fname.replace('hdf5', 'fits'), fname.replace('hdf5', 'ms'))
    assert os.path.exists(fname.replace('hdf5', 'ms'))
    ms = cc.ms()
    status = ms.open(fname.replace('hdf5', 'ms'))
    assert status
    ms.close()

    
    
