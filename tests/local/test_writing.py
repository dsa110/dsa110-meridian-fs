import pkg_resources
from dsamfs.meridian_fringestop import run_fringestopping
from dsacalib.utils import get_autobl_indices
from pyuvdata import UVData
import glob
import numpy as np
from dsamfs.uvh5_utils import uvh5_to_ms
from casatasks import importuvfits
import casatools as cc
import os
from astropy.time import Time
from dsamfs.fringestopping import calc_uvw_blt
import astropy.io.fits as pf
from antpos.utils import get_baselines
import astropy.units as u
import astropy.constants as c

def test_end2end(tmpdir):
    data_path = pkg_resources.resource_filename('dsamfs', 'data/')
    param_path = '{0}/test_parameters.yaml'.format(data_path)
    header_path = '{0}/test_header.txt'.format(data_path)
    run_fringestopping(param_path, header_file=header_path, output_dir=tmpdir)
    fname = glob.glob('{0}/*.hdf5'.format(tmpdir))[0]
    UV = UVData()
    UV.read(fname, file_type='uvh5')
    # Check that the baselines are in the correct order
    nant = UV.Nants_data
    abi = get_autobl_indices(nant, casa=False)
    ant1, ant2 = UV.baseline_to_antnums(UV.baseline_array)
    antenna_order = ant2[abi]+1
    print(antenna_order)
    assert np.all(ant1[abi] == ant2[abi])
    print(UV.time_array[:10])
    print(type(UV.time_array))
    print(UV.time_array.dtype)
    # Check that we can convert to uvfits
    uvh5_to_ms(fname, fname.replace('.hdf5', ''))
    assert os.path.exists(fname.replace('hdf5', 'fits'))
    # Check that we can read in the uvfits file
    assert os.path.exists(fname.replace('hdf5', 'ms'))
    ms = cc.ms()
    status = ms.open(fname.replace('hdf5', 'ms'))
    assert status
    uvw_ms = ms.getdata('uvw')['uvw']
    ms.close()
    # Check that the UVW coordinates are right in the fits file
    f = pf.open(fname.replace('hdf5', 'fits'))
    uu = (f['PRIMARY'].data['UU']*u.s*c.c).to_value(u.m)
    vv = (f['PRIMARY'].data['VV']*u.s*c.c).to_value(u.m)
    ww = (f['PRIMARY'].data['WW']*u.s*c.c).to_value(u.m)
    ant1 = f['PRIMARY'].data['ANTENNA1']
    ant2 = f['PRIMARY'].data['ANTENNA2']
    assert ant2[1]==ant2[0] # Check that ant1 and ant2 are defined properly
    bnames = ['{0}-{1}'.format(int(ant1[i]), int(ant2[i])) for i in range(len(ant1))]
    df = get_baselines(antenna_order, casa_order=False, autocorrs=True)
    idx = [np.where(df['bname']==bn)[0][0] for bn in bnames]
    blen = np.array([df['x_m'][idx], df['y_m'][idx], df['z_m'][idx]]).T
    time = Time(f['PRIMARY'].data['DATE'],format='jd').mjd
    for i in range(10):
        try:
            if f['PRIMARY'].header['CTYPE{0}'.format(i)] == 'RA':
                ra = f['PRIMARY'].header['CRVAL{0}'.format(i)]*u.deg
            elif f['PRIMARY'].header['CTYPE{0}'.format(i)] == 'DEC':
                dec = f['PRIMARY'].header['CRVAL{0}'.format(i)]*u.deg
        except KeyError:
            continue
    assert ra is not None
    assert dec is not None
    uvw = calc_uvw_blt(blen, time, 'J2000', ra, dec)
    assert np.all(np.abs(uvw[:, 0] - uu) < 1e-4)
    assert np.all(np.abs(uvw[:, 1] - vv) < 1e-4)
    assert np.all(np.abs(uvw[:, 2] - ww) < 1e-4)
    assert np.all(np.abs(uvw-uvw_ms.T) < 1e-4)
    UV = UVData()
    UV.read(fname.replace('hdf5', 'ms'), file_type='ms')
    assert np.all(np.abs(UV.antenna_diameters-4.65) < 1e-4)
