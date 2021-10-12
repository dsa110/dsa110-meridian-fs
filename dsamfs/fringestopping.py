"""
DSAMFS/FRINGESTOPPING.PY

Dana Simard, dana.simard@astro.caltech.edu 11/2019

Casa-based routines for calculating and applying fringe-stopping phases
to visibilities
"""
import sys
import os
import numpy as np
import scipy # pylint: disable=unused-import
import casatools as cc
import astropy.units as u
from dsacalib import constants as ct
from dsacalib.fringestopping import calc_uvw

def calc_uvw_blt(blen, tobs, src_epoch, src_lon, src_lat, obs='OVRO_MMA'):
    """Calculates uvw coordinates.

    Uses CASA to calculate the u,v,w coordinates of the baselines `b` towards a
    source or phase center (specified by `src_epoch`, `src_lon` and `src_lat`)
    at the specified time and observatory.

    Parameters
    ----------
    blen : ndarray
        The ITRF coordinates of the baselines.  Type float, shape (nblt,
        3), units of meters.
    tobs : ndarray
        An array of floats, the times in MJD for which to calculate the uvw
        coordinates, shape (nblt).
    src_epoch : str
        The epoch of the source or phase-center, as a CASA-recognized string
        e.g. ``'J2000'`` or ``'HADEC'``
    src_lon : astropy quantity
        The longitude of the source or phase-center, in degrees or an
        equivalent unit.
    src_lat : astropy quantity
        The latitude of the source or phase-center, in degrees or an equivalent
        unit.

    Returns
    -------
    bu : ndarray
        The u-value for each time and baseline, in meters. Shape is
        ``(len(b), len(tobs))``.
    bv : ndarray
        The v-value for each time and baseline, in meters. Shape is
        ``(len(b), len(tobs))``.
    bw : ndarray
        The w-value for each time and baseline, in meters. Shape is
        ``(len(b), len(tobs))``.
    """
    nblt = tobs.shape[0]
    buvw = np.zeros((nblt, 3))
    # Define the reference frame
    me = cc.measures()
    qa = cc.quanta()
    if obs is not None:
        me.doframe(me.observatory(obs))
    if not isinstance(src_lon.ndim, float) and src_lon.ndim > 0:
        assert src_lon.ndim == 1
        assert src_lon.shape[0] == nblt
        assert src_lat.shape[0] == nblt
        direction_set = False
    else:
        if (src_epoch == 'HADEC') and (nblt > 1):
            raise TypeError('HA and DEC must be specified at each '
                           'baseline-time in tobs.')
        me.doframe(me.direction(src_epoch,
                                qa.quantity(src_lon.to_value(u.deg), 'deg'),
                                qa.quantity(src_lat.to_value(u.deg), 'deg')))
        direction_set = True
    contains_nans = False
    for i in range(nblt):
        me.doframe(me.epoch('UTC', qa.quantity(tobs[i], 'd')))
        if not direction_set:
            me.doframe(me.direction(src_epoch,
                                    qa.quantity(src_lon[i].to_value(u.deg),
                                                'deg'),
                                    qa.quantity(src_lat[i].to_value(u.deg),
                                                'deg')))
        bl = me.baseline('itrf', qa.quantity(blen[i, 0], 'm'),
                          qa.quantity(blen[i, 1], 'm'),
                          qa.quantity(blen[i, 2], 'm'))
        # Get the uvw coordinates
        try:
            buvw[i, :] = me.touvw(bl)[1]['value']
        except KeyError:
            contains_nans = True
            buvw[i, :] = np.ones(3)*np.nan
    if contains_nans:
        print('Warning: some solutions not found for u, v, w coordinates')
    return buvw


def generate_fringestopping_table(
    blen,
    pt_dec,
    nint,
    tsamp,      
    antenna_order,
    outrigger_delays,
    bname,
    outname='fringestopping_table',
    mjd0=58849.0
):
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
    antenna_order : list
        The order of the antennas.
    outrigger_delays : dict
        The outrigger delays in ns.
    bname : list
        The names of each baseline. Length nbaselines. Names are strings.
    outname : str
        The prefix to use for the table to which to save the w vectors. Will
        save the output to `outname`.npy Defaults ``fringestopping_table``.
    mjd0 : float
        The start time in MJD. Defaults 58849.0.
    """
    # Get the indices that correspond to baselines with the refant
    # Use the first antenna as the refant so that the baselines are in
    # the same order as the antennas
    refidxs = []
    refant = str(antenna_order[0])
    for i, bn in enumerate(bname):
        if refant in bn:
            refidxs += [i]

    # Get the geometric delays at the "source" position and meridian
    dt = np.arange(nint)*tsamp
    dt = dt-np.median(dt)
    hangle = dt*360/ct.SECONDS_PER_SIDEREAL_DAY
    _bu, _bv, bw = calc_uvw(
        blen,
        mjd0+dt/ct.SECONDS_PER_DAY,
        'HADEC',
        hangle*u.deg,        
        np.ones(hangle.shape)*(pt_dec*u.rad).to(u.deg)
    )
    _bu, _bv, bwref = calc_uvw(
        blen,
        mjd0,
        'HADEC',
        0.*u.deg,
        (pt_dec*u.rad).to(u.deg)
    )
    ant_bw = bwref[refidxs] 
    bw = bw-bwref
    bw = bw.T
    bwref = bwref.T
    
    # Add in per-antenna delays for each baseline
    for i, bn in enumerate(bname):
        ant1, ant2 = bn.split('-')
        # Add back in bw at the meridian calculated per antenna
        bw[:, i] += ant_bw[antenna_order.index(int(ant2)), :] - \
            ant_bw[antenna_order.index(int(ant1)), :]
        # Add in outrigger delays
        bw[:, i] += (outrigger_delays.get(int(ant1), 0) - \
            outrigger_delays.get(int(ant2), 0))*0.29979245800000004

    # Save the fringestopping table
    if os.path.exists(outname):
        os.unlink(outname)
    np.savez(outname, dec_rad=pt_dec, tsamp_s=tsamp, ha=hangle, bw=bw,
             bwref=bwref, antenna_order=antenna_order, outrigger_delays=outrigger_delays, ant_bw=ant_bw)

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
    print(vis.shape, vis_model.shape)
    vis /= vis_model
    if nans:
        nsamples = np.count_nonzero(~np.isnan(vis), axis=1)
        vis = np.nanmean(vis, axis=1)

    else:
        vis = np.mean(vis, axis=1)
        nsamples = np.ones(vis.shape)*nint
    return vis, nsamples

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
