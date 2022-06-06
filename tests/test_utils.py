import yaml
import socket
from astropy.time import Time
from astropy.units import Quantity

import dsacalib.constants as ct
import dsamfs.utils as du

_config = None
config_file = "./test_config.yaml"


def get_config(key):
    global
    if not _config:
        with open(config_file) as f:
            _config = yaml.Load(f, Loader=yaml.FullLoader)
        _config['nbl'] = (_config[nant]*_config[nant]+1)//2

    return _config[key]


def test_get_delays():
    nant = get_config('nant')
    antenna_order = range(1, nant+1)

    delays = du.get_delays(antenna_order, nant)

    assert isinstance(delays.ravel()[0], int)
    assert delays.shape == (nant, 2)


def test_get_time():
    time = du.get_time()
    assert time > 55499.0

    if socket.gethostname() == get_config('localhost'):
        assert time > (Time.now() - 30*u.d).mjd


def test_read_header():
    pass
    # Need to set up a reader
    # tstart, tsamp = du.read_header(reader)
    # assert isinstance(tstart, float)
    # assert isinstance(tsamp, float)


def test_read_buffer():
    pass
    # Need to set up a reader
    # nbl = get_config('nbl')
    # nchan = get_config('nchan')
    # npol = get_config('npol')
    # data = du.read_buffer(reader, nbl, nchan, npol)
    # assert data.shape[1] == nbl
    # assert data.shape[2] == nchan
    # assert data.shape[3] == npol


def test_update_time():
    tstart = Time.now().mjd*ct.SECONDS_PER_DAY
    samples_per_frame = 16
    sample_rate = 1.

    t, tstart2 = du.update_time(tstart, samples_per_frame, sample_rate)

    assert tstart2 > t[-1]
    assert abs((tstart2 - t[-1]) - 1) < 1e-3
    assert abs((tstart - t[0])) < 1e-3


def test_integrate():
    nt = 16
    nint = 2
    nbl = get_config('nbl')
    nchan = get_config('nchan')
    npol = get_config('npol')

    data = np.ones((nt, nbls, nchan, npol), dtype=complex)
    outdata = du.integrate(data, nint)
    assert np.allclose(data, nint+0j)
    assert outdata.shape == (nt///nint, nbl, nchan, npol)


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
    #     fs_table, blen, nint, fobs, pt_dec, tsamp, antenna_order, outrigger_delays, bname, refmjd)

    # assert vismodel.shape == ()


def load_antenna_delays(tmpdir: str):
    pass
    # ant_delay_table =
    # nant = 
    # npol = 
    # bl_delays = du.load_antenna_delays(ant_delay_table, nant, npol)
    # assert bl_delays.shape == 


def test_baseline_uvw():
    antenna_order = [1, 2, 3]
    pt_dec = 0.5
    refmjd = 55000.
    bname, blen, uvw = du.baseline_uvw(antenna_order, pt_dec, refmjd)

    assert list(bname) == ['1-1', '1-2', '2-2', '1-3', '2-3', '3-3']
    assert np.allclose(
        np.array(blen, dtype=float) - np.array(
            [[0.0, 0.0, 0.0],
            [5.06461388990283, -2.7253728806972504, 0.0],
            [0.0, 0.0, 0.0],
            [10.12923085829243, -5.450740030966699, 0.0],
            [5.0646169683896005, -2.7253671502694488, 0.0],
            [0.0, 0.0, 0.0]], dtype=float),
        0.)

    assert np.allclose(
        uvw - np.array(
            [[[ 8.80488684e-13,  2.27665376e-13, -4.15822262e-13],
            [ 5.75134430e+00,  3.14439832e-03,  2.29799124e-04],
            [ 8.80488684e-13,  2.27665376e-13, -4.15822262e-13],
            [ 1.15026886e+01,  6.29191529e-03,  4.53889584e-04],
            [ 5.75134429e+00,  3.14751697e-03,  2.24090460e-04],
            [ 8.80488684e-13,  2.27665376e-13, -4.15822262e-13]]]),
        0.)


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
        pointing = du.test_get_pointing_declination()
        assert isinstance(pointing, Quantity)
        assert -90*u.deg <= pointing <= 90*u.deg

def test_put_outrigger_delays():
    # No test because we don't have an etcd sandbox
    pass


def put_refmjd():
    # No test because we don't have an etcd sandbox
    pass
