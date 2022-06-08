import os
import numpy as np
import astropy.units as u

from antpos import utils
import dsacalib.constants as ct
from dsacalib.fringestopping import calc_uvw

from dsamfs import fringestopping

from utils import get_config


def test_calc_uvw_blt():
    antenna_order = get_config('antenna_order')
    nant = len(antenna_order)
    nt = get_config('nt')
    nbl = (nant*(nant+1))//2
    tobs = get_config('refmjd') + np.arange(nt) / ct.SECONDS_PER_DAY
    df_bls = utils.get_baselines(antenna_order, autocorrs=True, casa_order=False)
    blen = np.array([df_bls['x_m'], df_bls['y_m'], df_bls['z_m']]).T
    ra = 14.31225787*u.hourangle
    dec = get_config('pt_dec')*u.rad
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


def test_generate_fringestopping_table(tmpdir: str):
    fstable = f"{tmpdir}/fs_table.npz"
    tsamp = get_config('tsamp')
    nint = get_config('nint')
    antenna_order = get_config('antenna_order')
    outrigger_delays = get_config('outrigger_delays')
    pt_dec = get_config('pt_dec')
    mjd0 = get_config('refmjd')

    bname = []
    for i, ant1 in enumerate(antenna_order):
        for ant2 in antenna_order[i:]:
            bname += [f"{ant1}-{ant2}"]

    df_bls = utils.get_baselines(antenna_order, autocorrs=True, casa_order=False)
    blen = np.array([df_bls['x_m'], df_bls['y_m'], df_bls['z_m']]).T

    fringestopping.generate_fringestopping_table(
        blen, pt_dec, nint, tsamp, antenna_order, outrigger_delays, bname, mjd0,
        outname=fstable)

    assert os.path.exists(fstable)

    fsdata = np.load(fstable)
    for key in [
            'dec_rad', 'tsamp_s', 'ha', 'bw', 'bwref', 'antenna_order', 'outrigger_delays',
            'ant_bw']:
        assert key in fsdata.files

    assert fsdata['bw'].shape == (nint, len(bname))


def test_zenith_visibility_mode(tmpdir: str):
    fobs = get_config('f0_GHz') + (
        get_config('deltaf_MHz') / 1e3 * np.arange(get_config('nchan')))
    nint = get_config('nint')
    nant = len(get_config('antenna_order'))
    nbl = (nant*(nant+1))//2
    nchan = get_config('nchan')
    f0_GHz = get_config('f0_GHz')
    deltaf_GHz = get_config('deltaf_MHz')/1e3
    fobs = f0_GHz + deltaf_GHz*np.arange(nchan)

    if not os.path.exists(f"{tmpdir}/fs_table.npz"):
        test_generate_fringestopping_table(tmpdir)

    vismodel = fringestopping.zenith_visibility_model(fobs, f"{tmpdir}/fs_table.npz")
    assert vismodel.shape == (1, nint, nbl, nchan, 1)
    assert isinstance(vismodel.ravel()[0], complex)


def test_fringestop_on_zenith():
    nt = get_config('nt')
    nbl = get_config('nbl')
    nchan = get_config('nchan')
    npol = get_config('npol')
    nint = get_config('nint')

    vis = np.ones((nt, nbl, nchan, npol), np.complex64)
    vis_model = np.tile(0.5 + 0.j, (1, nint, nbl, nchan, 1))
    vis2, nsamples = fringestopping.fringestop_on_zenith(vis, vis_model)
    assert vis2.shape[1:] == vis.shape[1:]
    assert vis2.shape[0] == vis.shape[0] // nint
    assert np.allclose(vis2, 2)
    assert np.all(nsamples == nint)

    vis = np.ones((nt, nbl, nchan, npol), np.complex64)
    vis_model = np.tile(0.5 + 0.j, (1, nint, nbl, nchan, 1))
    vis[-1, ...] = np.nan
    vis2, nsamples = fringestopping.fringestop_on_zenith(vis, vis_model, nans=True)
    assert vis2.shape[1:] == vis.shape[1:]
    assert vis2.shape[0] == vis.shape[0] // nint
    assert np.allclose(vis2, 2)
    assert np.all(nsamples[:-1, ...] == nint)
    assert np.all(nsamples[-1, ...] == nint - 1)
