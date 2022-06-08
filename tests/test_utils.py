import socket
import numpy as np
from astropy.time import Time
from astropy.units import Quantity
import astropy.units as u

import dsacalib.constants as ct
import dsamfs.utils as du
from utils import get_config


def test_get_delays():
    nant = get_config('nant')
    antenna_order = range(1, nant + 1)

    delays = du.get_delays(antenna_order, nant)

    int(delays.ravel()[0])
    assert delays.shape == (nant, 2)


def test_get_time():
    time = du.get_time()
    assert time > 55499.0

    if socket.gethostname() == get_config('localhost'):
        assert time > (Time.now() - 30 * u.d).mjd


def test_read_header():
    pass
    # Need to set up a reader
    # Do locally
    # tstart, tsamp = du.read_header(reader)
    # assert isinstance(tstart, float)
    # assert isinstance(tsamp, float)


def test_read_buffer():
    pass
    # Need to set up a reader
    # Do locally
    # nbl = get_config('nbl')
    # nchan = get_config('nchan')
    # npol = get_config('npol')
    # data = du.read_buffer(reader, nbl, nchan, npol)
    # assert data.shape[1] == nbl
    # assert data.shape[2] == nchan
    # assert data.shape[3] == npol


def test_update_time():
    tstart = Time.now().mjd * ct.SECONDS_PER_DAY
    samples_per_frame = 16
    sample_rate = 1.

    t, tstart2 = du.update_time(tstart, samples_per_frame, sample_rate)

    assert tstart2 > t[-1]
    assert abs((tstart2 - t[-1]) - 1 / ct.SECONDS_PER_DAY) < 1e-3
    assert abs((tstart - t[0])) < 1e-3


def test_integrate():
    nt = 16
    nint = 2
    nbl = get_config('nbl')
    nchan = get_config('nchan')
    npol = get_config('npol')

    data = np.ones((nt, nbl, nchan, npol), dtype=complex)
    outdata = du.integrate(data, nint)
    assert np.allclose(outdata, 1. + 0j)
    assert outdata.shape == (nt // nint, nbl, nchan, npol)


def test_load_visibility_model(tmpdir: str):
    pass
    # fs_table = f"{tmpdir}/fs_table.npz"
    # blen
    # nint =
    # fobs
    # pt_dec
    # tsamp
    # antenna_order
    # outrigger_delays
    # bname
    # refmjd

    # vis_model = du.load_visibility_model(
    #     fs_table, blen, nint, fobs, pt_dec, tsamp, antenna_order, outrigger_delays, bname,
    #     refmjd)

    # assert vismodel.shape == ()


def load_antenna_delays(tmpdir: str):
    pass
    # ant_delay_table =
    # nant =
    # npol =
    # bl_delays = du.load_antenna_delays(ant_delay_table, nant, npol)
    # assert bl_delays.shape ==


def test_baseline_uvw():
    antenna_order = get_config('antenna_order')
    pt_dec = get_config('pt_dec')
    refmjd = get_config('refmjd')

    expected_bname = get_config('bname')
    expected_blen = get_config('blen')
    expected_uvw = get_config('uvw')

    bname, blen, uvw = du.baseline_uvw(antenna_order, pt_dec, refmjd)

    assert list(bname) == expected_bname
    assert np.allclose(np.array(blen, dtype=float), expected_blen)
    assert np.allclose(uvw, expected_uvw)


def test_parse_params():
    if socket.gethostname() == get_config('localhost'):
        params = du.parse_params()
        param_types = (
            bool, str, int, int, int, np.ndarray, int, int, int, int, list, float, float, bool,
            int, dict, float)
        for i, param in enumerate(params):
            assert isinstance(param, param_types[i])


def test_get_pointing_declination():
    if socket.gethostname() == get_config('localhost'):
        pointing = du.get_pointing_declination()
        assert isinstance(pointing, Quantity)
        assert -90 * u.deg <= pointing <= 90 * u.deg


def test_put_outrigger_delays():
    # No test because we don't have an etcd sandbox
    pass


def put_refmjd():
    # No test because we don't have an etcd sandbox
    pass
