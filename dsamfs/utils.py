"""
DSAMFS/PSRSDADA_UTILS.PY

Dana Simard, dana.simard@astro.caltech.edu, 02/2020

Utilities to interact with the psrdada buffer written to
by the DSA-110 correlator
"""

import socket
from datetime import datetime
from collections import OrderedDict
import numpy as np
import yaml
import astropy.units as u
from antpos.utils import get_baselines
import scipy #pylint: disable=unused-import
import casatools as cc
from astropy.time import Time

from dsautils import dsa_store
import dsautils.dsa_syslog as dsl
from dsautils import cnf
import dsacalib.constants as ct
from dsacalib.fringestopping import calc_uvw

from dsamfs.fringestopping import generate_fringestopping_table
from dsamfs.fringestopping import zenith_visibility_model

# Logger
LOGGER = dsl.DsaSyslogger()
LOGGER.subsystem("software")
LOGGER.app("dsamfs")

ETCD = dsa_store.DsaStore()

def get_delays(antenna_order, nants):
    """Gets the delays currently set in the sanps.

    Parameters
    ----------
    antenna_order : array
        The order of antennas in the snaps.
    nants : int
        The total number of antennas in the array.

    Returns
    -------
    ndarray
        The delays for each antenna/pol in the array.
    """
    delays = np.zeros((nants, 2), dtype=np.int)
    d = dsa_store.DsaStore()
    nant_snap = 3
    nsnaps = len(antenna_order)//nant_snap
    nant_lastsnap = len(antenna_order)%nant_snap
    if nant_lastsnap != 0:
        nsnaps += 1
    else:
        nant_lastsnap = nant_snap
    for i in range(0, nsnaps):
        LOGGER.info('getting delays for snap {0} of {1}'.format(i+1, nsnaps))
        try:
            snap_delays = np.array(
                d.get_dict(
                    '/mon/snap/{0}/delays'.format(i+1)
                )['delays']
            )*2
            snap_delays = snap_delays.reshape(3, 2)[
                :nant_snap if i<nsnaps-1 else nant_lastsnap, :]
            delays[(antenna_order-1)[i*3:(i+1)*3], :] = snap_delays
        except (AttributeError, TypeError) as e:
            LOGGER.error('delays not set for snap{0}'.format(i+1))
    return delays

def get_time():
    """
    Gets the start time of the first spectrum from etcd.
    """
    try:
        ret_time = (ETCD.get_dict('/mon/snap/1/armed_mjd')['armed_mjd']
                    +float(ETCD.get_dict('/mon/snap/1/utc_start')['utc_start'])
                    *4.*8.192e-6/86400.)
    except:
        ret_time = 55000.0

    return ret_time

def read_header(reader):
    """
    Reads a psrdada header.

    Parameters
    ----------
    reader : psrdada reader instance
        The reader instance connected to the psrdada buffer.

    Returns
    -------
    tstart : float
        The start time in mjd seconds.
    tsamp : float
        The sample time in seconds.
    """
    header = reader.getHeader()
    tsamp = float(header['TSAMP'])
    tstart = float(header['MJD_START'])*ct.SECONDS_PER_DAY
    return tstart, tsamp

def read_buffer(reader, nbls, nchan, npol):
    """
    Reads a psrdada buffer as float32 and returns the visibilities.

    Parameters
    ----------
    reader : psrdada Reader instance
        An instance of the Reader class for the psrdada buffer to read.
    nbls : int
        The number of baselines.
    nchan : int
        The number of frequency channels.
    npol : int
        The number of polarizations.

    Returns
    -------
    ndarray
        The data. Dimensions (time, baselines, channels, polarization).
    """
    page = reader.getNextPage()
    reader.markCleared()

    data = np.asarray(page)
    data = data.view(np.float32)
    data = data.reshape(-1, 2).view(np.complex64).squeeze(axis=-1)
    try:
        data = data.reshape(-1, nbls, nchan, npol)
    except ValueError:
        print('incomplete data: {0} out of {1} samples'.format(
            data.shape[0]%(nbls*nchan*npol), nbls*nchan*npol))
        data = data[
            :data.shape[0]//(nbls*nchan*npol)*(nbls*nchan*npol)
            ].reshape(-1, nbls, nchan, npol)
    return data

def update_time(tstart, samples_per_frame, sample_rate):
    """
    Update the start time and the array of sample times for a dataframe.

    Parameters
    ----------
    tstart : float
        The start time of the frame in mjd seconds.
    samples_per_frame : int
        The number of time samples in the frame.
    sample_rate : float
        The sampling rate in samples per second.

    Returns
    -------
    t : array(float)
         The center of the time bin for each sample in mjd seconds.
    tstart : float
          The start time of the next dataframe in seconds.
    """
    t = tstart+np.arange(samples_per_frame)/sample_rate/ct.SECONDS_PER_DAY
    tstart += samples_per_frame/sample_rate/ct.SECONDS_PER_DAY
    return t, tstart

def integrate(data, nint):
    """
    A simple integration for testing and benchmarking.

    Integrates along the time axis.

    Parameters
    ----------
    data : ndarray
        The data to integrate. Dimensions (time, baseline, channel,
        polarization).
    nint : int
         The number of consecutive time samples to combine.

    Returns
    -------
    ndarray
        The integrated data. Dimensions (time, baseline, channel,
        polarization).
    """
    (_nt, nbls, nchan, npol) = data.shape
    data = data.reshape(-1, nint, nbls, nchan, npol).mean(1)
    return data

def load_visibility_model(
    fs_table, blen, nint, fobs, pt_dec, tsamp, antenna_order,
    outrigger_delays, bname, refmjd
):
    """
    Load the visibility model for fringestopping.

    If the path to the file does not exist or if the model is for a different
    number of integrations or baselines a new model will be created and saved
    to the file path. TODO: Order  may not be correct! Need to verify the
    antenna order that the correlator uses.

    Parameters
    ----------
    fs_table : str
         The full path to the .npz file containing the fringestopping model.
    antenna_order : array
         The order of the antennas in the correlator.
    nint : int
         The number of time samples to integrate.
    nbls : int
         The number of baselines.

    Returns
    -------
    ndarray
        The visibility model to use for fringestopping.
    """
    try:
        fs_data = np.load(fs_table, allow_pickle=True)
        assert fs_data['bw'].shape == (nint, blen.shape[0])
        assert np.abs(fs_data['dec_rad']-pt_dec) < 1e-6
        assert np.abs(fs_data['tsamp_s']-tsamp) < 1e-6
        assert np.all(fs_data['antenna_order']==antenna_order)
        assert fs_data['outrigger_delays']==outrigger_delays
        assert fs_data['refmjd']==refmjd
    except (FileNotFoundError, AssertionError, KeyError):
        print('Creating new fringestopping table.')
        generate_fringestopping_table(
            blen, pt_dec, nint, tsamp, antenna_order, outrigger_delays,
            bname, refmjd, outname=fs_table
        )

    vis_model = zenith_visibility_model(fobs, fs_table)

    return vis_model

def load_antenna_delays(ant_delay_table, nant, npol=2):
    """Load antenna delays from a CASA calibration table.

    Parameters
    ----------
    ant_delay_table : str
        The full path to the calibration table.
    nant : int
        The number of antennas.
    npol : int
        The number of polarizations.

    Returns
    -------
    ndarray
        The relative delay per baseline in nanoseconds. Baselines are in
        anti-casa order. Dimensions (nbaselines, npol).
    """
    error = 0
    tb = cc.table()
    error += not tb.open(ant_delay_table)
    antenna_delays = tb.getcol('FPARAM')
    npol = antenna_delays.shape[0]
    antenna_delays = antenna_delays.reshape(npol, -1, nant)
    error += not tb.close()

    bl_delays = np.zeros(((nant*(nant+1))//2, npol))
    idx = 0
    for i in np.arange(nant):
        for j in np.arange(i+1):
            #j-i or i-j ?
            bl_delays[idx, :] = antenna_delays[:, 0, j]-antenna_delays[:, 0, i]

    return bl_delays

def baseline_uvw(antenna_order, pt_dec, refmjd, autocorrs=True, casa_order=False):
    """Calculates the antenna positions and baseline coordinates.

    Parameters
    ----------
    antenna_order : list
        The names of the antennas in correct order.
    pt_dec : float
        The pointing declination in radians.
    autocorrs : bool
        Whether to consider only cross-correlations or both cross-correlations
        and auto-correlations when constructing baselines. Defaults True.
    casa_order : bool
        Whether the baselines are organized in casa order (e.g. [1-1, 1-2, 1-3,
        2-2, 2-3, 3-3]) or the reverse. Defaults False.

    Returns
    -------
    bname : list
        The names of the baselines, e.g. ['1-1', '1-2', '2-2'].
    blen : ndarray
        The itrf coordinates of the baselines, dimensions (nbaselines, 3).
    uvw : ndarray
        The uvw coordinates of the baselines for a phase reference at meridian.
        Dimensions (nbaselines, 3).
    """
    df_bls = get_baselines(antenna_order, autocorrs=autocorrs, casa_order=casa_order)
    bname = df_bls['bname']
    blen = np.array([df_bls['x_m'], df_bls['y_m'], df_bls['z_m']]).T
    bu, bv, bw = calc_uvw(blen, refmjd, 'HADEC', 0.*u.deg, (pt_dec*u.rad).to(u.deg))
    uvw = np.array([bu, bv, bw]).T
    return bname, blen, uvw

def parse_params(param_file=None):
    """Parses parameter file.

    Parameters
    ----------
    param_file : str
        The full path to the yaml parameter file.
    """
    if param_file is not None:
        with open(param_file) as fhand:
            corr_cnf = yaml.safe_load(fhand)
        mfs_cnf = corr_cnf

    else:
        my_cnf = cnf.Conf(use_etcd=True)
        corr_cnf = my_cnf.get('corr')
        mfs_cnf = my_cnf.get('fringe')

    test = mfs_cnf['test']
    key_string = mfs_cnf['key_string']
    nant = corr_cnf['nant']
    nchan = corr_cnf['nchan']
    npol = corr_cnf['npol']
    samples_per_frame = mfs_cnf['samples_per_frame']
    samples_per_frame_out = mfs_cnf['samples_per_frame_out']
    nint = mfs_cnf['nint']
    fringestop = mfs_cnf['fringestop']
    nfreq_int = mfs_cnf['nfreq_int']
    ant_od = OrderedDict(sorted(corr_cnf['antenna_order'].items()))
    antenna_order = list(ant_od.values())
    dfreq = corr_cnf['bw_GHz']/nchan
    if corr_cnf['chan_ascending']:
        fobs = corr_cnf['f0_GHz']+np.arange(nchan)*dfreq
    else:
        fobs = corr_cnf['f0_GHz']-np.arange(nchan)*dfreq
    pt_dec = get_pointing_declination().to_value(u.rad)
    tsamp = corr_cnf['tsamp'] # in seconds

    hname = socket.gethostname()
    try:
        ch0 = corr_cnf['ch0'][hname]
    except KeyError:
        ch0 = 3400
        LOGGER.error('host {0} not in correlator'.format(hname))
    nchan_spw = corr_cnf['nchan_spw']
    fobs = fobs[ch0:ch0+nchan_spw]
    filelength_minutes = mfs_cnf['filelength_minutes']
    outrigger_delays = mfs_cnf['outrigger_delays']

    refmjd = mfs_cnf['refmjd']

    assert (samples_per_frame_out*nint)%samples_per_frame == 0, \
        "Each frame out must contain an integer number of frames in."

    return test, key_string, nant, nchan_spw, npol, fobs, \
        samples_per_frame, samples_per_frame_out, nint, \
        nfreq_int, antenna_order, pt_dec, tsamp, fringestop, \
        filelength_minutes, outrigger_delays, refmjd

def get_pointing_declination():
    """Gets the pointing declination from etcd."""
    return ETCD.get_dict("/mon/array/dec")['dec_deg']*u.deg

def put_outrigger_delays(outrigger_delays):
    """Store the current outrigger delays in etcd."""
    current_mjd = Time(datetime.utcnow()).mjd

    for ant in range(1, 118):
        payload = {'time': current_mjd, 'ant_num': ant, 'delay': outrigger_delays.get(str(ant), 0)}
        ETCD.put_dict(f"/mon/fringe/{ant}", payload)

def put_refmjd(refmjd):
    """Store the current reference time in etcd."""
    current_mjd = Time(datetime.utcnow()).mjd
    payload = {'time': current_mjd, 'ant_num': 0, 'refmjd': refmjd}
    ETCD.put_dict("/mon/fringe/0", payload)
