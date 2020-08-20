"""
DSAMFS/FRINGESTOPPING.PY

Dana Simard, dana.simard@astro.caltech.edu 11/2019

Casa-based routines for calculating and applying fringe-stopping phases
to visibilities
"""
import sys
import numpy as np
import scipy # pylint: disable=unused-import
import casatools as cc
import astropy.units as u
from dsacalib import constants as ct
from dsacalib.fringestopping import calc_uvw

def generate_fringestopping_table(blen, pt_dec, nint=ct.NINT, tsamp=ct.TSAMP,
                                  outname='fringestopping_table',
                                  mjd0=58849.0):
    """Generates a table of the w vectors towards a source.

    Generates a table for use in fringestopping and writes it to a numpy
    pickle file named fringestopping_table.npz

    Parameters
    ----------
    blen : array
        The lengths of the baselines in ITRF coordinates, in m. Dimensions
        (nbaselines, 3).
    pt_dec : float
        The pointing declination in radians.
    nint : int
        The number of time integrations to calculate the table for.
    tsamp : float
        The sampling time in seconds.
    outname : str
        The prefix to use for the table to which to save the w vectors. Will
        save the output to `outname`.npy Defaults ``fringestopping_table``.
    mjd0 : float
        The start time in MJD. Defaults 58849.0.
    """
    dt = np.arange(nint)*tsamp
    dt = dt-np.median(dt)
    hangle = dt*360/ct.SECONDS_PER_SIDEREAL_DAY
    _bu, _bv, bw = calc_uvw(blen, mjd0+dt/ct.SECONDS_PER_DAY, 'HADEC',
                            hangle*u.deg,
                            np.ones(hangle.shape)*(pt_dec*u.rad).to(u.deg))
    if nint%2 == 1:
        bwref = bw[:, (nint-1)//2] #pylint: disable=unsubscriptable-object
    else:
        _bu, _bv, bwref = calc_uvw(blen, mjd0, 'HADEC', 0.*u.deg,
                                   (pt_dec*u.rad).to(u.deg))
        bwref = bwref.squeeze()
    bw = bw-bwref[:, np.newaxis]
    bw = bw.T
    bwref = bwref.T
    np.savez(outname, dec=pt_dec, ha=hangle, bw=bw, bwref=bwref)

def zenith_visibility_model(fobs, fstable='fringestopping_table.npz'):
    """Creates the visibility model from the fringestopping table.

    Parameters
    ----------
    fobs : array
        The observing frequency of each channel in GHz.
    fstable : str
        The full path to the fringestopping table. Defaults
        ``fringestopping_table.npz``.

    Returns
    -------
    ndarray
        The visibility model, dimensions (1, time, baseline, frequency,
        polarization).
    """
    data = np.load(fstable)
    bws = data['bw']
    vis_model = np.exp(2j*np.pi/ct.C_GHZ_M*fobs[:, np.newaxis]*
                       bws[np.newaxis, :, :, np.newaxis, np.newaxis])
    return vis_model

def fringestop_on_zenith(vis, vis_model, nans=False):
    """Performs meridian fringestopping.

    Fringestops on hour angle 0, declination pointing declination for the
    midpoint of each integration, then integrates the data.  The number of
    samples to integrate by is set by the length of the second axis of
    `vis_model`.

    Parameters
    ----------
    vis : ndarray
        The input visibilities, dimensions (time, baseline, frequency, pol).
    vis_model : ndarray
        The visibility model, dimensions (1, nint, baseline, frequency, pol).
    nans : boolean
        Whether the visibility array is nan-padded. Defaults False.

    Returns
    -------
    narray
        The fringe-stopped and integrated visibilities. Dimensions (time,
        baseline, frequency, pol).
    """
    nint = vis_model.shape[1]
    nt, nbl, nchan, npol = vis.shape
    assert nt%nint == 0, ('Number of times in the visibility file must be '
                          'divisible by nint')
    vis = vis.reshape(-1, nint, nbl, nchan, npol)
    vis /= vis_model
    if nans:
        vis = np.nanmean(vis, axis=1)
    else:
        vis = np.mean(vis, axis=1)
    return vis

def write_fs_delay_table(msname, source, blen, tobs, nant):
    """Writes the fringestopping delays to a delay calibration table.

    Not tested. Table is written to `msname`_`cal.name`_fscal

    Parameters
    ----------
    msname : str
        The prefix of the ms for which this table is generated.
    source : src class instance
        The source (or location) to fringestop on.
    blen : ndarray
        The ITRF coordinates of the baselines. Dimensions (baselines, 3).
    tobs : array
        The observation time of each time bin in mjd.
    nant : int
        The number of antennas in the array.
    """
    nt = tobs.shape[0]
    _bu, _bv, bw = calc_uvw(blen, tobs, source.epoch, source.ra, source.dec)

    ant_delay = np.zeros((nt, nant))
    ant_delay[:, 1:] = bw[:nant-1, :].T/ct.C_GHZ_M #pylint: disable=unsubscriptable-object

    error = 0
    tb = cc.table()
    error += not tb.open('{0}/templatekcal'.format(ct.PKG_DATA_PATH))
    error += not tb.copy('{0}_{1}_fscal'.format(msname, source.name))
    error += not tb.close()

    error += not tb.open('{0}_{1}_fscal'.format(msname, source.name),
                         nomodify=False)
    error += not tb.addrows(nant*nt-tb.nrows())
    error += not tb.flush()
    assert tb.nrows() == nant*nt
    error += not tb.putcol(
        'TIME', np.tile((tobs*u.d).to_value(u.s).reshape(-1, 1),
                        (1, nant)).flatten())
    error += not tb.putcol('FIELD_ID', np.zeros(nt*nant, dtype=np.int32))
    error += not tb.putcol(
        'SPECTRAL_WINDOW_ID', np.zeros(nt*nant, dtype=np.int32))
    error += not tb.putcol(
        'ANTENNA1', np.tile(np.arange(nant, dtype=np.int32).reshape(1, nant),
                            (nt, 1)).flatten())
    error += not tb.putcol('ANTENNA2', -1*np.ones(nt*nant, dtype=np.int32))
    error += not tb.putcol('INTERVAL', np.zeros(nt*nant, dtype=np.int32))
    error += not tb.putcol('SCAN_NUMBER', np.ones(nt*nant, dtype=np.int32))
    error += not tb.putcol('OBSERVATION_ID', np.zeros(nt*nant, dtype=np.int32))
    error += not tb.putcol(
        'FPARAM', np.tile(ant_delay.reshape(1, -1), (2, 1)).reshape(2, 1, -1))
    error += not tb.putcol(
        'PARAMERR', np.zeros((2, 1, nt*nant), dtype=np.float32))
    error += not tb.putcol('FLAG', np.zeros((2, 1, nt*nant), dtype=bool))
    error += not tb.putcol('SNR', np.zeros((2, 1, nt*nant), dtype=np.float64))
    error += not tb.flush()
    error += not tb.close()

    if error > 0:
        sys.stderr.write('{0} errors occured during calibration'.format(error))
