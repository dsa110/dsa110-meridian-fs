import os
import pytest
from antpos import utils
from dsamfs import fringestopping
import dsacalib.utils as du
import dsacalib.constants as ct
from dsacalib.fringestopping import calc_uvw
import numpy as np
import astropy.units as u

def test_gentable(tmpdir):
    fstable = '{0}/fs_table.npz'.format(tmpdir)
    pt_dec = 0.71094487066
    tsamp = 0.134217728
    nint = 10
    antenna_order = [24, 10, 3, 66]
    outrigger_delays = {24 : 1200, }
    bname = []
    for i, ant1 in enumerate(antenna_order):
        for ant2 in antenna_order[i:]:
            bname += ['{0}-{1}'.format(ant1, ant2)]
    df_bls = utils.get_baselines(antenna_order, autocorrs=True, casa_order=False)
    blen = np.array([df_bls['x_m'], df_bls['y_m'], df_bls['z_m']]).T
    fringestopping.generate_fringestopping_table(
        blen, pt_dec, nint, tsamp, antenna_order, outrigger_delays, bname,
        outname=fstable)
    assert os.path.exists(fstable)

def test_outrigger_lookup():
    bn = '100-101'
    ants = bn.split('-')
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, outrigger_delays = parse_params()
    delay = outrigger_delays.get(int(ants[0]), 0) - outrigger_delays.get(int(ants[1]), 0)
    assert np.abs(delay) > 0
    delay2 = outrigger_delays[int(ants[0])] - outrigger_delays[int(ants[1])]
    assert delay2 == delay
    
def test_write_fs_delay_table():
    msname = 'test_write'
    source = du.src('TEST', 16*u.hourangle, 37*u.deg, 1.)
    antenna_order = [24, 10, 3, 66]
    df_bls = utils.get_baselines(antenna_order, autocorrs=True, casa_order=False)
    blen = np.array([df_bls['x_m'], df_bls['y_m'], df_bls['z_m']]).T
    
def test_calc_uvw():
    nant = 5
    nt = 10
    nbl = (nant*(nant+1))//2
    antenna_order = np.arange(nant)+1
    tobs = 59100.956635023+np.arange(nt)/ct.SECONDS_PER_DAY
    df_bls = utils.get_baselines(antenna_order, autocorrs=True, casa_order=False)
    blen = np.array([df_bls['x_m'], df_bls['y_m'], df_bls['z_m']]).T
    ra = 14.31225787*u.hourangle
    dec = 0.71094487*u.rad
    uvw_blt = fringestopping.calc_uvw_blt(np.tile(blen[np.newaxis, :, :],
                                                  (nt, 1, 1)).reshape(-1, 3),
                                          np.tile(tobs[:, np.newaxis],
                                                  (1, nbl)).flatten(),
                                          'J2000', ra, dec)
    uu, vv, ww = calc_uvw(blen, tobs, 'J2000', ra, dec)
    print(uvw_blt.shape, uu.T.shape)
    assert np.all(np.abs(uvw_blt[:, 0]-uu.T.flatten()) < 1e-6)
    assert np.all(np.abs(uvw_blt[:, 1]-vv.T.flatten()) < 1e-6)
    assert np.all(np.abs(uvw_blt[:, 2]-ww.T.flatten()) < 1e-6)
                                          
    uvw_blt = fringestopping.calc_uvw_blt(np.tile(blen[np.newaxis, :, :],
                                                  (nt, 1, 1)).reshape(-1, 3),
                                          np.tile(tobs[:, np.newaxis],
                                                  (1, nbl)).flatten(),
                                          'HADEC',
                                          np.zeros(nt*nbl)*u.rad,
                                          np.ones(nt*nbl)*dec)
    uu, vv, ww = calc_uvw(blen, tobs, 'HADEC', np.zeros(nt)*u.rad, np.ones(nt)*dec)
    assert np.all(np.abs(uvw_blt[:, 0]-uu.T.flatten()) < 1e-6)
    assert np.all(np.abs(uvw_blt[:, 1]-vv.T.flatten()) < 1e-6)
    assert np.all(np.abs(uvw_blt[:, 2]-ww.T.flatten()) < 1e-6)
