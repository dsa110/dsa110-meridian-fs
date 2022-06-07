import os
import h5py
import numpy as np
from pyuvdata import UVData
from astropy.time import Time
import dsacalib.constants as ct

from dsamfs import io as mfsio
from utils import get_config


def test_initialize_uvh5_file(tmpdir: str):
    nfreq = get_config('nchan')
    npol = get_config('npol')
    pt_dec = get_config('pt_dec')
    antenna_order = get_config('antenna_order')
    nant = len(antenna_order)
    fobs = get_config('f0_GHz') + np.arange(nfreq) * get_config('deltaf_MHz') / 1e3
    fs_table = 'fs_table.npz'

    with h5py.File(f"{tmpdir}/test.hdf5", "w") as fhdf5:
        mfsio.initialize_uvh5_file(fhdf5, nfreq, npol, pt_dec, antenna_order, fobs, fs_table)

    with h5py.File(f"{tmpdir}/test.hdf5", "r") as fhdf5:
        assert "Header" in fhdf5
        for key in [
                "latitude", "longitude", "altitude", "telescope_name", "instrument", "object_name",
                "history", "phase_type", "phase_center_app_dec", "Nants_data", "Nants_telescope",
                "antenna_diameters", "ant_1_array", "ant_2_array", "antenna_names",
                "antenna_numbers", "Nbls", "Nblts", "Nfreqs", "Npols", "Ntimes", "Nspws",
                "uvw_array", "time_array", "integration_time", "freq_array",
                "channel_width", "spw_array", "polarization_array", "antenna_positions",
                "extra_keywords"]:
            assert key in fhdf5["Header"]
        nants_data = fhdf5['Header']['Nants_data'][()]
        assert nants_data == nant
        assert "Data" in fhdf5
        for key in ["visdata", "flags", "nsamples"]:
            assert key in fhdf5["Data"]


def test_update_uvh5_file(tmpdir: str):
    if not os.path.exists(f"{tmpdir}/test.hdf5"):
        test_initialize_uvh5_file(tmpdir)

    nt = get_config('nt')
    tsamp = get_config('tsamp')
    obstime = Time.now().mjd + tsamp / ct.SECONDS_PER_DAY * np.arange(nt)

    bname = get_config('bname')
    nbl = len(bname)
    uvw = get_config('uvw')
    nint = get_config('nint')
    nchan = get_config('nchan')
    npol = get_config('npol')

    nsamples = np.tile(nint, (nt, nbl, nchan, npol))
    data = np.ones((nt, nbl, nchan, npol), dtype=np.complex64)

    with h5py.File(f"{tmpdir}/test.hdf5", "r+") as fhdf5:
        mfsio.update_uvh5_file(fhdf5, data, obstime, tsamp, bname, uvw, nsamples)

    with h5py.File(f"{tmpdir}/test.hdf5", "r") as fhdf5:
        assert fhdf5["Data"]["visdata"].shape == (nt * nbl, 1, nchan, npol)
        assert fhdf5["Header"]["time_array"].shape[0] == nt * nbl
        assert fhdf5["Header"]["ant_1_array"].shape[0] == nt * nbl

    UV = UVData()
    UV.read(f"{tmpdir}/test.hdf5", file_type='uvh5')
    data = UV.data_array
    assert data.shape == (nt * nbl, 1, nchan, npol)


def test_dada_to_uvh5(tmpdir: str):
    pass
    # This one requires a reader - do locally only
