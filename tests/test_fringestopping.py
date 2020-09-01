import os
import pytest
from antpos import utils
from dsamfs import fringestopping
import dsacalib.utils as du
import numpy as np

def test_gentable(tmpdir):
    fstable = '{0}/fs_table.npz'.format(tmpdir)
    nint = 10
    antenna_order = [24, 10, 3, 66]
    df_bls = utils.get_baselines(antenna_order, autocorrs=True, casa_order=False)
    blen = np.array([df_bls['x_m'], df_bls['y_m'], df_bls['z_m']]).T
    fringestopping.generate_fringestopping_table(blen, nint, outname=fstable)
    assert os.path.exists(fstable)                   

def test_write_fs_delay_table(tmpdir):
    msname = 'test_write'
    source = du.src('TEST', 16*u.hourangle, 37*u.deg, 1.)
    antenna_order = [24, 10, 3, 66]
    df_bls = utils.get_baselines(antenna_order, autocorrs=True, casa_order=False)
    blen = np.array([df_bls['x_m'], df_bls['y_m'], df_bls['z_m']]).T
    
