"""
HDF5_UTILS.PY

Dana Simard, dana.simard@astro.caltech.edu, 02/2020

Routines to interact w/ hdf5 files used for the 24-hr visibility buffer
"""

# To do:
# Replace to_deg w/ astropy versions

import numpy as np
import h5py
from dsacalib import constants as ct
from antpos.utils import get_baselines
import astropy.units as u
from astropy.utils import iers
iers.conf.iers_auto_url_mirror = ct.IERS_TABLE
from astropy.time import Time #pylint: disable=wrong-import-position

def read_hdf5_file(fl, source=None, dur=50*u.min, autocorrs=False,
                   badants=None, quiet=True):
    """Reads in an hdf5 visibility file.

    Parameters
    ----------
    fl : str
        The full path to the hdf5 file.
    source : src class instance
        The source to extract from the hdf5 file. If None, the entire file is
        read. Defaults None.
    dur : astropy quantity
        The amount (in time) to read from the file. Defaults ``50*u.min``.
    autocorrs : boolean
        If True, extracts both autocorrelations and crosscorrelations.  If
        False, extracts only crosscorrelations. Defaults False.
    badants : list
        A list of bad antennas. Baselines with antennas in this list are not
        extracted from the file. Defaults None.
    quiet : boolean
        If False, prints additional information about the file.  Defaults True.
    """
    if source is not None:
        lstmid = source.ra.to_value(u.rad)
        seg_len = (dur/2*(15*u.deg/u.h)).to_value(u.rad)

    with h5py.File(fl, 'r') as f:
        antenna_order = list(f['antenna_order'][...])
        nant = len(antenna_order)
        fobs = f['fobs_GHz'][...]
        mjd = f['time_mjd_seconds'][...]/ct.SECONDS_PER_DAY
        nt = len(f['time_mjd_seconds'])
        tsamp = (mjd[0]-mjd[-1])/(nt-1)*ct.SECONDS_PER_DAY
        lst0 = Time(mjd[0], format='mjd').sidereal_time(
            'apparent', longitude=ct.OVRO_LON*u.rad).radian
        lst = np.angle(np.exp(1j*(lst0+2*np.pi/ct.SECONDS_PER_SIDEREAL_DAY*
                                  np.arange(nt)*tsamp)))

        if source is not None:
            if not quiet:
                print("\n-------------EXTRACT DATA--------------------")
                print("Extracting data around {0}".format(lstmid*180/np.pi))
                print("{0} Time samples in data".format(nt))
                print("LST range: {0:.1f} --- ({1:.1f}-{2:.1f}) --- {3:.1f}deg"
                      .format(lst[0]*180./np.pi, (lstmid-seg_len)*180./np.pi,
                              (lstmid+seg_len)*180./np.pi, lst[-1]*180./np.pi))

            idx1 = np.argmax(np.absolute(
                np.exp(1j*lst)+np.exp(1j*lstmid)*np.exp(-1j*seg_len)))
            idx2 = np.argmax(np.absolute(
                np.exp(1j*lst)+np.exp(1j*lstmid)*np.exp(1j*seg_len)))
            transit_idx = np.argmax(np.absolute(np.exp(1j*lst)+
                                                np.exp(1j*(lstmid))))

            mjd = mjd[idx1:idx2]
            lst = lst[idx1:idx2]
            dat = f['vis'][idx1:idx2, ...]
            if not quiet:
                print("Extract: {0} ----> {1} sample; transit at {2}".format(
                    idx1, idx2, transit_idx))
                print("----------------------------------------------")
        else:
            dat = f['vis'][...]
            transit_idx = None

    # Now we have to extract the correct baselines

    if not autocorrs:
        basels = list(range(int(nant/2*(nant+1))))
        i = -1
        for j in range(1, nant+1):
            i += j
            basels.remove(i)
        # Fancy indexing can have downfalls and may change in future numpy
        # versions. See issue here https://github.com/numpy/numpy/issues/9450
        vis = dat[:, basels, ...]
        assert vis.shape[0] == len(mjd)
        assert vis.shape[1] == len(basels)

#     if autocorrs:
#         bname = np.array([[a, a] for a in antenna_order])
#         blen  = np.zeros((len(antenna_order), 3))
#         if badants is not None:
#             badants = [str(ba) for ba in badants]
#             good_idx = list(range(len(antenna_order)))
#             for badant in badants:
#                 good_idx.remove(antenna_order.index(badant))
#             vis = vis[:, good_idx,...]
#             bname = bname[good_idx, ...]
#             blen = blen[good_idx, ...]
#
#    if not autocorrs:
    df_bls = get_baselines(antenna_order, autocorrs=autocorrs, casa_order=True)
    blen = np.array([df_bls['x_m'], df_bls['y_m'], df_bls['z_m']]).T
    bname = [bn.split('-') for bn in df_bls['bname']]
    if badants is not None:
        bname = np.array(bname)
        blen = np.array(blen)
        good_idx = list(range(len(bname)))
        for i, bn in enumerate(bname):
            if (bn[0] in badants) or (bn[1] in badants):
                good_idx.remove(i)
        vis = vis[:, good_idx, ...]
        blen = blen[good_idx, ...]
        bname = bname[good_idx, ...]

        #badants = [str(ba) for ba in badants]
        for badant in badants:
            antenna_order.remove(badant)

    assert vis.shape[0] == len(mjd)
    vis = vis.swapaxes(0, 1)
    dt = np.median(np.diff(mjd))
    if len(mjd) > 0:
        tstart = mjd[0]-dt/2
        tstop = mjd[-1]+dt/2
    else:
        tstart = None
        tstop = None

    if not isinstance(bname, list):
        bname = bname.tolist()

    return fobs, blen, bname, tstart, tstop, vis, mjd, transit_idx, \
        antenna_order

def initialize_hdf5_file(fhdf, fobs, antenna_order, t0, nbls, nchan, npol,
                         nant):
    """Initializes the hdf5 file.

    Parameters
    ----------
    fhdf : hdf5 file handler
        The file to initialize.
    fobs : array
        The center frequency of each channel in GHz.
    antenna_order : array
        The order of the antennas in the correlator. Antenna names should be
        expressed as ints.
    t0 : float
        The time of the first sample in mjd seconds.
    nbls : int
        The number of baselines.
    nchan : int
        The number of channels.
    npol : int
        The number of polarizations.
    nant : int
        The number of antennas.

    Returns
    -------
    vis_ds : hdf5 dataset
        The dataset for the visibilities
    t_ds : hdf5 dataset
        The dataset for the times.
    """
    _ds_fobs = fhdf.create_dataset(
        "fobs_GHz", (nchan, ), dtype=np.float32, data=fobs)
    _ds_ants = fhdf.create_dataset(
        "antenna_order", (nant, ), dtype=np.int, data=antenna_order)
    _t_st = fhdf.create_dataset(
        "tstart_mjd_seconds", (1, ), maxshape=(1, ), dtype=int, data=t0)
    vis_ds = fhdf.create_dataset(
        "vis", (0, nbls, nchan, npol), maxshape=(None, nbls, nchan, npol),
        dtype=np.complex64, chunks=True, data=None)
    t_ds = fhdf.create_dataset(
        "time_seconds", (0, ), maxshape=(None, ), dtype=np.float32,
        chunks=True, data=None)
    return vis_ds, t_ds
