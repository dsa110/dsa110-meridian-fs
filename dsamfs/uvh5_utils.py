"""
HDF5_UTILS.PY

Dana Simard, dana.simard@astro.caltech.edu, 02/2020

Routines to interact w/ hdf5 files used for the 24-hr visibility buffer
"""
from datetime import datetime
import numpy as np
import traceback
import h5py
from dsacalib import constants as ct
from antpos.utils import get_itrf, get_baselines
import astropy.units as u
import dsamfs.psrdada_utils as pu
from dsamfs.fringestopping import fringestop_on_zenith, calc_uvw_blt
import dsautils.dsa_syslog as dsl
from pyuvdata import UVData
from astropy.time import Time
import astropy.constants as c
from dsacalib.utils import get_autobl_indices

logger = dsl.DsaSyslogger()
logger.subsystem("software")
logger.app("dsacalib")

def initialize_uvh5_file(fhdf, nfreq, npol, pt_dec, antenna_order, fobs,
                         fs_table=None):
    """Initializes an HDF5 file according to the UVH5 specification.

    For details on the specification of the UVH5 file format, see the pyuvdata
    memo "UVH5 file format" from November 28, 2018.

    Parameters
    ----------
    fhdf5 : file handler
        The hdf5 file to initialize.
    nbls : int
        The number of baselines in the correlated data.
    nfreq : int
        The number of frequency channels in the correlated data.
    npol : int
        The number of polarizations in the correlated data.
    pt_dec : float
        The declination at which the visbilities are phased, in radians.
    antenna_order : array
        The order of the antennas.  The antennas should be specified as
        integers between 1 and 117 inclusive.  (E.g. DSA-24 should be
        identified as 24.)
    fs_table : str
        The full path to the table used in fringestopping.  Defaults None.
    """
    # also need the itrf coordinates of the antennas
    df = get_itrf(height=ct.OVRO_ALT*u.m, latlon_center=(ct.OVRO_LAT*u.rad,
                                                         ct.OVRO_LON*u.rad))
    ant_itrf = np.array([df['dx_m'], df['dy_m'], df['dz_m']]).T
    nants_telescope = max(df.index)
    # have to have some way of calculating the ant_1_array and
    # ant_2_array order and uvw array.  The uvw array should be constant but
    # still has to have dimensions (nblts, 3)

    # Header parameters
    header = fhdf.create_group("Header")
    data = fhdf.create_group("Data")
    # The following must be defined - to add in
    header["latitude"] = (ct.OVRO_LAT*u.rad).to_value(u.deg)
    header["longitude"] = (ct.OVRO_LON*u.rad).to_value(u.deg)
    header["altitude"] = ct.OVRO_ALT
    header["telescope_name"] = np.string_("OVRO_MMA")
    header["instrument"] = np.string_("DSA")
    header["object_name"] = np.string_("search")
    header["history"] = np.string_("written by dsa110-meridian-fringestopping "
                                   "on {0}".format(datetime.now().strftime(
                                       '%Y-%m-%dT%H:%M:%S')))
    header["phase_type"] = np.string_("drift")
    header["Nants_data"] = len(antenna_order)
    header["Nants_telescope"] = nants_telescope
    # ant_1_array and ant_2_array have ot be updated
    header.create_dataset(
        "ant_1_array", (0, ), maxshape=(None, ), dtype=np.int,
        chunks=True, data=None)
    header.create_dataset(
        "ant_2_array", (0, ), maxshape=(None, ), dtype=np.int,
        chunks=True, data=None)
    antenna_names = np.array(['{0}'.format(ant_no+1) for ant_no in
                              range(nants_telescope)], dtype="S4")
    header.create_dataset("antenna_names", (nants_telescope, ), dtype="S4",
                          data=antenna_names)
    header["antenna_numbers"] = np.arange(nants_telescope)
    header["Nbls"] = ((header["Nants_data"][()]+1)*
                      header["Nants_data"][()])//2
    header["Nblts"] = 0
    header["Nfreqs"] = nfreq
    header["Npols"] = npol
    header["Ntimes"] = 0
    header["Nspws"] = 1
    header.create_dataset(
        "uvw_array", (0, 3), maxshape=(None, 3), dtype=np.float32,
        chunks=True, data=None)
    header.create_dataset(
        "time_array", (0, ), maxshape=(None, ), dtype=np.float64,
        chunks=True, data=None)
    header.create_dataset(
        "integration_time", (0, ), maxshape=(None, ), dtype=np.float64,
        chunks=True, data=None)
    header["freq_array"] = fobs[np.newaxis, :]*1e9
    header["channel_width"] = np.median(np.diff(fobs))*1e9
    header["spw_array"] = np.array([1])
    # Polarization array is defined at the top of page 8 of
    # AIPS memo 117:
    # Values of 1 through 4 are assiged to Stokes I, Q, U, V
    # Values of -5 through -8 to XX, YY, XY, YX
    header["polarization_array"] = np.array([-5, -6])
    header["antenna_positions"] = ant_itrf

    # Optional parameters
    extra = header.create_group("extra_keywords")
    extra["phase_center_dec"] = pt_dec
    extra["ha_phase_center"] = 0.
    extra["phase_center_epoch"] = 2000
    if fs_table is not None:
        extra["fs_table"] = np.string_(fs_table)
        # Should link to a perm vrsn

    # Data sets
    data.create_dataset(
        "visdata", (0, 1, nfreq, npol), maxshape=(None, 1, nfreq, npol),
        dtype=np.complex64, chunks=True, data=None)
    data.create_dataset(
        "flags", (0, 1, nfreq, npol), maxshape=(None, 1, nfreq, npol),
        dtype=np.bool, chunks=True, data=None)
    # likely set flags_dataset all to 1?
    data.create_dataset(
        "nsamples", (0, 1, nfreq, npol), maxshape=(None, 1, nfreq, npol),
        dtype=np.float32)
    # nsamples tells us how many samples went into each integration

def update_uvh5_file(fhdf5, data, t, tsamp, bname, uvw, nsamples):
    """Appends new data to the uvh5 file.

    Currently assumes phasing at the meridian. To account for tracking, need to
    update to allow the passed uvw to also have time axis.

    Parameters
    ----------
    fhdf5 : file handler
        The open and initialized hdf5 file.
    data : ndarray
        The data to append to the file. Dimensions (time, baseline, channel,
        polarization).
    t : array
        The central time of each timebin in `data`, in MJD.
    tsamp : float
        The sampling time of the data before integration.
    bname : array
        The name of each baseline. E.g. ['1-1', '1-2', '1-3', '2-2', '2-3',
        '3-3'].
    uvw : ndarray
        The UVW coordinates at the phase center. Dimensions (nbls, 3).
    nsamples : ndarray
        The number of samples (unflagged) samples that have been integrated for
        each bin of `data`.  Same dimensions as `data`.
    """
    (nt, nbls, nchan, npol) = data.shape
    assert t.shape[0] == nt
    assert data.shape == nsamples.shape
    assert uvw.shape[1] == nbls
    assert uvw.shape[2] == 3
    antenna_order = fhdf5["Header"]["antenna_names"][:]
    ant_1_array = np.array(
        [np.where(antenna_order == np.string_(bn.split('-')[0]))
         for bn in bname], dtype=np.int).squeeze()
    ant_2_array = np.array(
        [np.where(antenna_order == np.string_(bn.split('-')[1]))
         for bn in bname], dtype=np.int).squeeze()
    old_size = fhdf5["Header"]["time_array"].shape[0]
    new_size = old_size+nt*nbls
    fhdf5["Header"]["time_array"].resize(new_size, axis=0)
    fhdf5["Header"]["time_array"][old_size:] = np.tile(
        t[:, np.newaxis], (1, nbls)).flatten()
    fhdf5["Header"]["integration_time"].resize(new_size, axis=0)
    fhdf5["Header"]["integration_time"][old_size:] = np.ones(
        (nt*nbls, ), dtype=np.float32)*tsamp
    # Note that the uvw convention for pyuvdata is -1*the standard convention
    fhdf5["Header"]["uvw_array"].resize(new_size, axis=0)
    if uvw.shape[0] == 1:
        fhdf5["Header"]["uvw_array"][old_size:, :] = -1*np.tile(
            uvw, (nt, 1, 1)).reshape(-1, 3)
    else:
        assert uvw.shape[0] == nt
        fhdf5["Header"]["uvw_array"][old_size:, :] = -1*uvw.reshape(-1, 3)
    fhdf5["Header"]["Ntimes"][()] = new_size//nbls
    fhdf5["Header"]["Nblts"][()] = new_size
    fhdf5["Header"]["ant_1_array"].resize(new_size, axis=0)
    fhdf5["Header"]["ant_1_array"][old_size:] = np.tile(
        ant_1_array[np.newaxis, :], (nt, 1)).flatten()
    fhdf5["Header"]["ant_2_array"].resize(new_size, axis=0)
    fhdf5["Header"]["ant_2_array"][old_size:] = np.tile(
        ant_2_array[np.newaxis, :], (nt, 1)).flatten()
    fhdf5["Data"]["visdata"].resize(new_size, axis=0)
    fhdf5["Data"]["visdata"][old_size:, ...] = data.reshape(
        nt*nbls, 1, nchan, npol)
    fhdf5["Data"]["flags"].resize(new_size, axis=0)
    fhdf5["Data"]["flags"][old_size:, ...] = np.zeros(
        (nt*nbls, 1, nchan, npol), dtype=np.bool)
    fhdf5["Data"]["nsamples"].resize(new_size, axis=0)
    fhdf5["Data"]["nsamples"][old_size:, ...] = nsamples.reshape(
        nt*nbls, 1, nchan, npol)

def dada_to_uvh5(reader, fout, nbls, nchan, npol, nint, samples_per_frame_out,
                 sample_rate_out, pt_dec, antenna_order, fs_table,
                 tsamp, bname, uvw, fobs, vis_model):
    """
    Reads dada buffer and writes to uvh5 file.
    """
    print('Opening output file {0}.hdf5'.format(fout))
    with h5py.File('{0}.hdf5'.format(fout), 'w') as fhdf5:
        initialize_uvh5_file(fhdf5, nchan, npol, pt_dec, antenna_order, fobs,
                             fs_table)

        idx_frame_out = 0
        nans = False
        tstart = pu.get_time()
        tstart += (nint*tsamp/2)/ct.SECONDS_PER_DAY + 2400000.5
        while not nans:
            data_in = np.ones((samples_per_frame_out*nint, nbls, nchan, npol),
                              dtype=np.complex64)*np.nan
            for i in range(data_in.shape[0]):
                try:
                    assert reader.isConnected
                    data_in[i, ...] = pu.read_buffer(reader, nbls, nchan, npol)
                except (AssertionError, ValueError) as e:
                    print('Last integration has {0} timesamples'.format(i))
                    logger.info('Disconnected from buffer with message {0}:\n{1}'.
                                format(type(e).__name__, ''.join(
                                    traceback.format_tb(e.__traceback__))))
                    nans = True
                    break

            data, nsamples = fringestop_on_zenith(data_in, vis_model, nans)
            t, tstart = pu.update_time(tstart, samples_per_frame_out,
                                       sample_rate_out)
            update_uvh5_file(fhdf5, data, t, tsamp, bname, uvw, nsamples)

            idx_frame_out += 1
            print('Integration {0} done'.format(idx_frame_out))

        reader.disconnect()

def uvh5_to_uvfits(fname, ra=None, dec=None, dt=None):
    """
    Converts a uvh5 data to a uvfits file.
    
    Parameters
    ----------
    fname : str
        The full path to the uvh5 data file.
    ra : astropy quantity
        The RA at which to phase the data. If None, will phase at the meridian
        of the center of the uvh5 file.
    dec : astropy quantity
        The DEC at which to phase the data. If None, will phase at the pointing
        declination.
    dt : astropy quantity
        Duration of data to extract. If None, will extract the entire file.
    """
    UV = UVData()
    UV.read(fname, file_type='uvh5')
    time = Time(UV.time_array, format='jd')
    pt_dec = UV.extra_keywords['phase_center_dec']*u.rad

    if ra is None:
        ra = UV.lst_array[UV.Nblts//2]*u.rad
    if dec is None:
        dec = pt_dec
    # Extract only the times that we are interested in
    if dt is not None:
        lst_min = (ra - (dt*2*np.pi*u.rad/(ct.SECONDS_PER_SIDEREAL_DAY*u.s))/2).to_value(u.rad)%(2*np.pi)
        lst_max = (ra + (dt*2*np.pi*u.rad/(ct.SECONDS_PER_SIDEREAL_DAY*u.s))/2).to_value(u.rad)%(2*np.pi)
        if lst_min < lst_max:
            idx_to_extract = np.where(UV.lst_array >= lst_min and UV.lst_array <= lst_max)
        else:
            idx_to_extract = np.where(UV.lst_array >= lst_min or UV.lst_array <= lst_max )
        tmin = time[min(idx_to_extract)]
        tmax = time[max(idx_to_extract)]
        UV.read(fname, file_type='uvh5', time_range=[tmin.jd, tmax.jd])
        time = Time(UV.time_array, format='jd')
    lamb = c.c/(UV.freq_array*u.Hz)
    # Get the baselines in itrf coordinates
    ant1, ant2 = UV.baseline_to_antnums(UV.baseline_array)
    antenna_order = [int(UV.antenna_names[np.where(UV.antenna_numbers==ant1[abi])[0][0]]) for
                                          abi in get_autobl_indices(UV.Nants_data, casa=False)]
    df = get_baselines(antenna_order, casa_order=False, autocorrs=True)
    ant1_names = np.array(UV.antenna_names)[UV.antenna_numbers[UV.ant_1_array]]
    ant2_names = np.array(UV.antenna_names)[UV.antenna_numbers[UV.ant_2_array]]
    bnames = ['{0}-{1}'.format(ant1_names[i], ant2_names[i]) for i in range(len(ant1_names))]
    idx = [np.where(df['bname']==bn)[0][0] for bn in bnames]
    blen = np.array([df['x_m'][idx], df['y_m'][idx], df['z_m'][idx]]).T
    # UVW coordinates for drift scan
    uvwd = calc_uvw_blt(blen, time.mjd, 'HADEC', np.zeros(UV.Nblts)*u.rad, np.ones(UV.Nblts)*pt_dec)
    uvwp = calc_uvw_blt(blen, time.mjd, 'J2000', ra, dec)
    dw = (uvwp[:, -1] - uvwd[:, -1])*u.m
    # Not sure about sign - double check
    phase_model = np.exp((2*np.pi/lamb*-1*dw[:, np.newaxis, np.newaxis]).to_value(
        u.dimensionless_unscaled))
    UVout = UV.copy()
    UVout.uvw_array = -1*uvwp # Miriad convention
    UVout.data_array = UV.data_array/phase_model[..., np.newaxis]
    UVout.phase_type = 'phased'
    UVout.phase_center_ra = ra.to_value(u.rad)
    UVout.phase_center_dec = dec.to_value(u.rad)
    UVout.phase_center_epoch = 2000.
    UVout.write_uvfits('{0}fits'.format(fname.strip('hdf5')), spoof_nonessential=True)
