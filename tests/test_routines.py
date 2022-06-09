import socket
import os
import glob

import pkg_resources
import numpy as np
import casatools as cc
from pyuvdata import UVData
from astropy.time import Time
import astropy.units as u
import astropy.constants as c
import astropy.io.fits as pf

from antpos.utils import get_itrf
from dsacalib.utils import get_autobl_indices
from dsacalib.ms_io import uvh5_to_ms

from dsamfs.fringestopping import calc_uvw_blt
from utils import get_config


def test_end2end(tmpdir):

    try:
        from dsamfs.routines import run_fringestopping
    except ModuleNotFoundError:
        print('No psrdada installed.  Skipping test.')
        return

    if socket.gethostname() == get_config('localhost'):
        # Access data stored on disk
        data_path = pkg_resources.resource_filename('dsamfs', 'data/')
        param_path = f"{data_path}/test_parameters.yaml"
        header_path = f"{data_path}/test_header.txt"

        # Run fringestopping as a test
        run_fringestopping(param_file=param_path, header_file=header_path, output_dir=tmpdir)

        # Check that the hdf5 file metadata is correct
        fname = glob.glob(f"{tmpdir}/*.hdf5")[0]
        UV = UVData()
        UV.read(fname, file_type='uvh5')
        # Check that the baselines are in the correct order
        nant = UV.Nants_data
        abi = get_autobl_indices(nant, casa=False)
        ant1, ant2 = UV.baseline_to_antnums(UV.baseline_array)
        assert np.all(ant1[abi] == ant2[abi])

        # Check that we can convert to uvfits and ms
        uvh5_to_ms(fname, fname.replace('.hdf5', ''))
        ms = cc.ms()
        status = ms.open(fname.replace('hdf5', 'ms'))
        assert status
        uvw_ms = ms.getdata('uvw')['uvw']
        ms.close()

        # Check that the UVW coordinates are right in the fits file
        # f = pf.open(fname.replace('hdf5', 'fits'))
        # uu = (f['PRIMARY'].data['UU']*u.s*c.c).to_value(u.m)
        # vv = (f['PRIMARY'].data['VV']*u.s*c.c).to_value(u.m)
        # ww = (f['PRIMARY'].data['WW']*u.s*c.c).to_value(u.m)
        # ant1_array = f['PRIMARY'].data['ANTENNA1']
        # ant2_array = f['PRIMARY'].data['ANTENNA2']

        # df_itrf = get_itrf()
        # antenna_positions = np.array([df_itrf['x_m'], df_itrf['y_m'],
        #                                  df_itrf['z_m']]).T-UV.telescope_location
        # blen = np.zeros((ant1_array.shape[0], 3))
        # for i, ant1 in enumerate(ant1_array):
        #     ant2 = ant2_array[i]
        #     blen[i, ...] = antenna_positions[int(ant2)-1, :] - \
        #                    antenna_positions[int(ant1)-1, :]

        # assert ant1_array[1]==ant1_array[0] # Check that ant1 and ant2 are defined properly
        # time = Time(f['PRIMARY'].data['DATE'],format='jd').mjd
        # for i in range(10):
        #     try:
        #         if f['PRIMARY'].header["CTYPE{i}"] == 'RA':
        #             ra = f['PRIMARY'].header[f"CRVAL{i}"]*u.deg
        #         elif f['PRIMARY'].header[f"CTYPE{i}"] == 'DEC':
        #             dec = f['PRIMARY'].header[f"CRVAL{i}"]*u.deg
        #     except KeyError:
        #         continue
        # assert ra is not None
        # assert dec is not None

        # uvw = calc_uvw_blt(blen, time, 'J2000', ra, dec)
        # uvw = -1*uvw

        # assert np.all(np.abs(uvw[:, 0] - uu) < 1e-1)
        # assert np.all(np.abs(uvw[:, 1] - vv) < 1e-1)
        # assert np.all(np.abs(uvw[:, 2] - ww) < 1e-1)
        # assert np.all(np.abs(uvw-uvw_ms.T) < 1e-2)

        # Check that we can open the ms
        UV = UVData()
        UV.read(fname.replace('hdf5', 'ms'), file_type='ms')
        assert np.all(np.abs(UV.antenna_diameters-4.65) < 1e-4)
