"""
DSAMFS/PSRSDADA_UTILS.PY

Dana Simard, dana.simard@astro.caltech.edu, 02/2020

Utilities to interact with the psrdada buffer written to
by the DSA-110 correlator
"""

import os
from datetime import datetime
import numpy as np
import dsacalib.constants as ct
from dsamfs.fringestopping import generate_fringestopping_table
from dsamfs.fringestopping import zenith_visibility_model
from antpos.utils import get_baselines
import scipy #pylint: disable=unused-import
import casatools as cc

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
    t = tstart+np.arange(samples_per_frame)/sample_rate
    tstart += samples_per_frame/sample_rate
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

def load_visibility_model(fs_table, blen, nint, fobs, pt_dec,
                          ant_delay_tbl=None):
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
        fs_data = np.load(fs_table)
        assert fs_data['bw'].shape == (nint, blen.shape[0])
    except (FileNotFoundError, AssertionError):
        print('Creating new fringestopping table.')
        generate_fringestopping_table(blen, pt_dec, nint, tsamp,
                                      outname=fs_table)
        os.link(fs_table,
                '{0}_{1}.npz'.format(
                    fs_table.strip('.npz'),
                    datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')))

    vis_model = zenith_visibility_model(fobs, fs_table)

    if ant_delay_tbl is not None:
        bl_delays = load_antenna_delays(ant_delay_tbl, len(antenna_order))
        vis_model /= np.exp(2j*np.pi*
                            fobs[:, np.newaxis]*bl_delays[:, np.newaxis, :])

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

def antenna_positions(antenna_order, pt_dec, autocorrs=True, casa_order=False):
    raise NotImplementedError
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
    ant_itrf : ndarray
        The ITRF coordinates of the antenna positions relative to the center of
        the array.  Dimensions (nants, 3).
    """
    df_bls = get_baselines(antenna_order, autocorrs, casa_order)
    bname = df_bls['bname']
    blen = np.array([df_bls['x_m'], df_bls['y_m'], df_bls['z_m']]).T
    bu, bv, bw = calc_uvw(blen, 58849.0, 'HADEC', 0.*u.deg,
                          (pt_dec*u.rad).to(u.deg))
    uvw = np.array([bu, bv, bw]).T
    # also need the itrf coordinates of the antennas
    
    return bname, blen, uvw, ant_itrf