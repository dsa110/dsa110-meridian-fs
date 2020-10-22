"""DSAMFS/IO.PY

Routines to read and manipulate the correlator-data psrdada buffer stream and
write the correlated data to a uvh5 file.

Dana Simard, dana.simard@astro.caltech.edu, 2020
"""

from datetime import datetime
import os
import traceback
import numpy as np
import h5py
import astropy.units as u
from psrdada.exceptions import PSRDadaError
from antpos.utils import get_itrf
import dsautils.dsa_syslog as dsl
import dsacalib.constants as ct
import dsamfs.utils as pu
from dsamfs.fringestopping import fringestop_on_zenith

logger = dsl.DsaSyslogger()
logger.subsystem("software")
logger.app("dsamfs")

def dada_to_uvh5(reader, outdir, nbls, nchan, npol, nint, nfreq_int,
                 samples_per_frame_out, sample_rate_out, pt_dec, antenna_order,
                 fs_table, tsamp, bname, uvw, fobs, vis_model, test):
    """
    Reads dada buffer and writes to uvh5 file.
    """
    if nfreq_int > 1:
        assert nchan%nfreq_int == 0, ("Number of channels must be an integer "
                                      "number of output channels.")
        fobs = np.median(fobs.reshape(-1, nfreq_int), axis=1)
        nchan = len(fobs)

    nans = False
    idx_frame_out = 0 # total number of fsed frames, for timekeeping
    max_frames_per_file = int(np.ceil(60*60*sample_rate_out))
    while not nans:
        now = datetime.utcnow()
        fout = now.strftime("%Y-%m-%dT%H:%M:%S")
        if outdir is not None:
            fout = '{0}/{1}'.format(outdir, fout)
        print('Opening output file {0}.hdf5'.format(fout))
        with h5py.File('{0}_incomplete.hdf5'.format(fout), 'w') as fhdf5:
            initialize_uvh5_file(fhdf5, nchan, npol, pt_dec, antenna_order,
                                 fobs, fs_table)

            idx_frame_file = 0 # number of fsed frames write to curent file
            while (idx_frame_file < max_frames_per_file) and (not nans):
                data_in = np.ones(
                    (samples_per_frame_out*nint, nbls, nchan*nfreq_int, npol),
                    dtype=np.complex64)*np.nan
                for i in range(data_in.shape[0]):
                    try:
                        assert reader.isConnected
                        data_in[i, ...] = pu.read_buffer(
                            reader, nbls, nchan*nfreq_int, npol)
                    except (AssertionError, ValueError, PSRDadaError) as e:
                        print('Last integration has {0} timesamples'.format(i))
                        logger.info('Disconnected from buffer with message'
                                    '{0}:\n{1}'.
                                    format(type(e).__name__, ''.join(
                                        traceback.format_tb(e.__traceback__))))
                        nans = True
                        break

                if idx_frame_out == 0:
                    if test:
                        tstart = 59000.5
                    else:
                        tstart = pu.get_time()
                    tstart += (nint*tsamp/2)/ct.SECONDS_PER_DAY+2400000.5

                data, nsamples = fringestop_on_zenith(data_in, vis_model, nans)
                t, tstart = pu.update_time(tstart, samples_per_frame_out,
                                           sample_rate_out)
                if nfreq_int > 1:
                    if not nans:
                        data = np.mean(data.reshape(
                            data.shape[0], data.shape[1], nchan, nfreq_int,
                            npol), axis=3)
                        nsamples = np.mean(nsamples.reshape(
                            nsamples.shape[0], nsamples.shape[1], nchan,
                            nfreq_int, npol), axis=3)
                    else:
                        data = np.nanmean(
                            data.reshape(data.shape[0], data.shape[1], nchan,
                                         nfreq_int, npol),
                            axis=3
                        )
                        nsamples = np.nanmean(nsamples.reshape(
                            nsamples.shape[0], nsamples.shape[1], nchan,
                            nfreq_int, npol), axis=3)

                update_uvh5_file(fhdf5, data, t, tsamp, bname, uvw, nsamples)

                idx_frame_out += 1
                idx_frame_file += 1
                print('Integration {0} done'.format(idx_frame_out))
        os.rename('{0}_incomplete.hdf5'.format(fout), '{0}.hdf5'.format(fout))
    try:
        reader.disconnect()
    except PSRDadaError:
        pass

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
    # The following must be defined
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
    header["antenna_diameters"] = np.ones(nants_telescope)*4.65
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
    header["channel_width"] = np.abs(np.median(np.diff(fobs))*1e9)
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
         for bn in bname], dtype=np.int
    ).squeeze()
    ant_2_array = np.array(
        [np.where(antenna_order == np.string_(bn.split('-')[1]))
         for bn in bname], dtype=np.int
    ).squeeze()

    old_size = fhdf5["Header"]["time_array"].shape[0]
    new_size = old_size+nt*nbls

    # TIME_ARRAY
    fhdf5["Header"]["time_array"].resize(new_size, axis=0)
    fhdf5["Header"]["time_array"][old_size:] = np.tile(
        t[:, np.newaxis],
        (1, nbls)
    ).flatten()

    # INTEGRATION_TIME
    fhdf5["Header"]["integration_time"].resize(new_size, axis=0)
    fhdf5["Header"]["integration_time"][old_size:] = np.ones(
        (nt*nbls, ),
        dtype=np.float32
    )*tsamp

    # UVW_ARRAY
    # Note that the uvw and baseline convention for pyuvdata is B-A,
    # where vis=A^* B
    fhdf5["Header"]["uvw_array"].resize(new_size, axis=0)
    if uvw.shape[0] == 1:
        fhdf5["Header"]["uvw_array"][old_size:, :] = np.tile(
            uvw,
            (nt, 1, 1)
        ).reshape(-1, 3)
    else:
        assert uvw.shape[0] == nt
        fhdf5["Header"]["uvw_array"][old_size:, :] = uvw.reshape(-1, 3)

    # Ntimes and Nblts
    fhdf5["Header"]["Ntimes"][()] = new_size//nbls
    fhdf5["Header"]["Nblts"][()] = new_size

    # ANT_1_ARRAY
    fhdf5["Header"]["ant_1_array"].resize(new_size, axis=0)
    fhdf5["Header"]["ant_1_array"][old_size:] = np.tile(
        ant_1_array[np.newaxis, :],
        (nt, 1)
    ).flatten()

    # ANT_2_ARRAY
    fhdf5["Header"]["ant_2_array"].resize(new_size, axis=0)
    fhdf5["Header"]["ant_2_array"][old_size:] = np.tile(
        ant_2_array[np.newaxis, :],
        (nt, 1)
    ).flatten()

    # VISDATA
    fhdf5["Data"]["visdata"].resize(new_size, axis=0)
    fhdf5["Data"]["visdata"][old_size:, ...] = data.reshape(
        nt*nbls, 1, nchan, npol)

    # FLAGS
    fhdf5["Data"]["flags"].resize(new_size, axis=0)
    fhdf5["Data"]["flags"][old_size:, ...] = np.zeros(
        (nt*nbls, 1, nchan, npol), dtype=np.bool)

    # NSAMPLES
    fhdf5["Data"]["nsamples"].resize(new_size, axis=0)
    fhdf5["Data"]["nsamples"][old_size:, ...] = nsamples.reshape(
        nt*nbls, 1, nchan, npol)

def dada_to_uvh5(reader, outdir, nbls, nchan, npol, nint, nfreq_int,
                 samples_per_frame_out, sample_rate_out, pt_dec, antenna_order,
                 fs_table, tsamp, bname, uvw, fobs, vis_model, test):
    """
    Reads dada buffer and writes to uvh5 file.
    """
    if nfreq_int > 1:
        assert nchan%nfreq_int == 0, ("Number of channels must be an integer "
                                      "number of output channels.")
        fobs = np.median(fobs.reshape(-1, nfreq_int), axis=1)
        nchan = len(fobs)

    nans = False
    idx_frame_out = 0 # total number of fsed frames, for timekeeping
    max_frames_per_file = int(np.ceil(15*60*sample_rate_out))
    while not nans:
        now = datetime.utcnow()
        fout = now.strftime("%Y-%m-%dT%H:%M:%S")
        if outdir is not None:
            fout = '{0}/{1}'.format(outdir, fout)
        print('Opening output file {0}.hdf5'.format(fout))
        with h5py.File('{0}_incomplete.hdf5'.format(fout), 'w') as fhdf5:
            initialize_uvh5_file(fhdf5, nchan, npol, pt_dec, antenna_order,
                                 fobs, fs_table)

            idx_frame_file = 0 # number of fsed frames write to curent file
            while (idx_frame_file < max_frames_per_file) and (not nans):
                data_in = np.ones(
                    (samples_per_frame_out*nint, nbls, nchan*nfreq_int, npol),
                    dtype=np.complex64)*np.nan
                for i in range(data_in.shape[0]):
                    try:
                        assert reader.isConnected
                        data_in[i, ...] = pu.read_buffer(
                            reader, nbls, nchan*nfreq_int, npol)
                    except (AssertionError, ValueError, PSRDadaError) as e:
                        print('Last integration has {0} timesamples'.format(i))
                        logger.info('Disconnected from buffer with message'
                                    '{0}:\n{1}'.
                                    format(type(e).__name__, ''.join(
                                        traceback.format_tb(e.__traceback__))))
                        nans = True
                        break

                if idx_frame_out == 0:
                    if test:
                        tstart = 59000.5
                    else:
                        tstart = pu.get_time()
                    tstart += (nint*tsamp/2)/ct.SECONDS_PER_DAY+2400000.5

                data, nsamples = fringestop_on_zenith(data_in, vis_model, nans)
                t, tstart = pu.update_time(tstart, samples_per_frame_out,
                                           sample_rate_out)
                if nfreq_int > 1:
                    if not nans:
                        data = np.mean(data.reshape(
                            data.shape[0], data.shape[1], nchan, nfreq_int,
                            npol), axis=3)
                        nsamples = np.mean(nsamples.reshape(
                            nsamples.shape[0], nsamples.shape[1], nchan,
                            nfreq_int, npol), axis=3)
                    else:
                        data = np.nanmean(data.reshape(
                            data.shape[0], data.shape[1], nchan, nfreq_int, npol),
                                         axis=3)
                        nsamples = np.nanmean(nsamples.reshape(
                            nsamples.shape[0], nsamples.shape[1], nchan,
                            nfreq_int, npol), axis=3)

                update_uvh5_file(fhdf5, data, t, tsamp, bname, uvw, nsamples)

                idx_frame_out += 1
                idx_frame_file += 1
                print('Integration {0} done'.format(idx_frame_out))
        os.rename('{0}_incomplete.hdf5'.format(fout), '{0}.hdf5'.format(fout))
    try:
        reader.disconnect()
    except PSRDadaError:
        pass

def uvh5_to_ms(fname, msname, ra=None, dec=None, dt=None, antenna_list=None,
               flux=None):
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
    zenith_dec = 0.6503903199825691*u.rad
    UV = UVData()
    # This next section contains the list 
    if dt is not None:
        if isinstance(fname, list):
            fname_init = fname[0]
        else:
            fname_init = fname
        if antenna_list is not None:
            UV.read(fname_init, file_type='uvh5', antenna_names=antenna_list)
        else:
            UV.read(fname_init, file_type='uvh5')
        time = Time(UV.time_array, format='jd')

        pt_dec = UV.extra_keywords['phase_center_dec']*u.rad
        if ra is None:
            ra = UV.lst_array[UV.Nblts//2]*u.rad
        if dec is None:
            dec = pt_dec

        lst_min = (ra - (dt*2*np.pi*u.rad/(ct.SECONDS_PER_SIDEREAL_DAY*u.s))/2
                  ).to_value(u.rad)%(2*np.pi)
        lst_max = (ra + (dt*2*np.pi*u.rad/(ct.SECONDS_PER_SIDEREAL_DAY*u.s))/2
                  ).to_value(u.rad)%(2*np.pi)
        if lst_min < lst_max:
            idx_to_extract = np.where((UV.lst_array >= lst_min) &
                                      (UV.lst_array <= lst_max))[0]
        else:
            idx_to_extract = np.where((UV.lst_array >= lst_min) |
                                      (UV.lst_array <= lst_max))[0]
        tmin = time[min(idx_to_extract)]
        tmax = time[max(idx_to_extract)]
        UV = UVData()
        if antenna_list is not None:
            UV.read(fname, file_type='uvh5', antenna_names=antenna_list,
                    time_range=[tmin.jd, tmax.jd])
        else:
            UV.read(fname, file_type='uvh5', time_range=[tmin.jd, tmax.jd])
        time = Time(UV.time_array, format='jd')
    else:
        if antenna_list is not None:
            UV.read(fname, file_type='uvh5', antenna_names=antenna_list)
        else:
            UV.read(fname, file_type='uvh5')
        time = Time(UV.time_array, format='jd')

        pt_dec = UV.extra_keywords['phase_center_dec']*u.rad
        if ra is None:
            ra = UV.lst_array[UV.Nblts//2]*u.rad
        if dec is None:
            dec = pt_dec

    lamb = c.c/(UV.freq_array*u.Hz)
    # Get the baselines in itrf coordinates
    ant1, _ant2 = UV.baseline_to_antnums(UV.baseline_array)
    antenna_order = [int(UV.antenna_names[np.where(
        UV.antenna_numbers==ant1[abi])[0][0]]) for abi in
                     get_autobl_indices(UV.Nants_data, casa=False)]
    df = get_baselines(antenna_order, casa_order=False, autocorrs=True)
    df_itrf = get_itrf()
    blen = np.array([df['x_m'], df['y_m'], df['z_m']]).T
    blen = np.tile(blen[np.newaxis, ...], (UV.Ntimes, 1, 1)).reshape(-1, 3)
    UV.antenna_positions = np.array([df_itrf['x_m'], df_itrf['y_m'],
                                     df_itrf['z_m']]).T-UV.telescope_location
    uvw_m = calc_uvw_blt(blen[:UV.Nbls], time[:UV.Nbls].mjd, 'HADEC',
                         np.zeros(UV.Nbls)*u.rad, np.ones(UV.Nbls)*pt_dec)
    uvw_z = calc_uvw_blt(blen[:UV.Nbls], time[:UV.Nbls].mjd, 'HADEC',
                         np.zeros(UV.Nbls)*u.rad, np.ones(UV.Nbls)*zenith_dec)
    dw = (uvw_z[:, -1] - uvw_m[:, -1])*u.m
    # Not sure about sign - double check
    phase_model = np.exp((2j*np.pi/lamb*dw[:, np.newaxis, np.newaxis])
                         .to_value(u.dimensionless_unscaled))
    UV.uvw_array = np.tile(uvw_z[np.newaxis, :, :], (UV.Ntimes, 1, 1)
                       ).reshape(-1, 3)
    UV.data_array = (UV.data_array.reshape(UV.Ntimes, UV.Nbls, UV.Nspws,
                                           UV.Nfreqs, UV.Npols)
                     /phase_model[np.newaxis, ..., np.newaxis]).reshape(
        UV.Nblts, UV.Nspws, UV.Nfreqs, UV.Npols)

    UV.phase(ra.to_value(u.rad), dec.to_value(u.rad), use_ant_pos=False)
    # Below is the manual calibration which can be used instead if needed.
    #uvw = calc_uvw_blt(blen, time.mjd, 'RADEC', ra.to(u.rad), dec.to(u.rad))
    #dw = (uvw[:, -1] - np.tile(uvw_m[np.newaxis, :, -1], (UV.Ntimes, 1)
    #                          ).reshape(-1))*u.m
    #phase_model = np.exp((2j*np.pi/lamb*dw[:, np.newaxis, np.newaxis])
    #                      .to_value(u.dimensionless_unscaled))
    #UV.uvw_array = uvw
    #UV.data_array = UV.data_array/phase_model[..., np.newaxis]
    #UV.phase_type = 'phased'
    #UV.phase_center_dec = dec.to_value(u.rad)
    #UV.phase_center_ra = ra.to_value(u.rad)
    #UV.phase_center_epoch = 2000.
    # Look for missing channels
    freq = UV.freq_array.squeeze()
    # The channels may have been reordered by pyuvdata so check that the
    # parameter UV.channel_width makes sense now.
    ascending = np.median(np.diff(freq)) > 0
    if ascending:
        assert np.all(np.diff(freq) > 0)
        if UV.channel_width < 0:
            UV.channel_width *= -1
    else:
        assert np.all(np.diff(freq) < 0)
        if UV.channel_width > 0:
            UV.channel_width *= -1
    # Are there missing channels?
    if not np.all(np.diff(freq)-UV.channel_width < 1e-5):
        # There are missing channels!
        nfreq = int(np.rint(np.abs(freq[-1]-freq[0])/UV.channel_width+1))
        freq_out = freq[0] + np.arange(nfreq)*UV.channel_width
        existing_idxs = np.rint((freq-freq[0])/UV.channel_width).astype(int)
        data_out = np.zeros((UV.Nblts, UV.Nspws, nfreq, UV.Npols),
                            dtype=UV.data_array.dtype)
        nsample_out = np.zeros((UV.Nblts, UV.Nspws, nfreq, UV.Npols),
                                dtype=UV.nsample_array.dtype)
        flag_out = np.zeros((UV.Nblts, UV.Nspws, nfreq, UV.Npols),
                             dtype=UV.flag_array.dtype)
        # Ah, fancy indexing, be careful with numpy reshaping bugs!
        data_out[:, :, existing_idxs, :] = UV.data_array
        nsample_out[:, :, existing_idxs, :] = UV.nsample_array
        flag_out[:, :, existing_idxs, :] = UV.flag_array
        # Now write everything
        UV.Nfreqs = nfreq
        UV.freq_array = freq_out[np.newaxis, :]
        UV.data_array = data_out
        UV.nsample_array = nsample_out
        UV.flag_array = flag_out

    if os.path.exists('{0}.fits'.format(msname)):
        os.remove('{0}.fits'.format(msname))

    UV.write_uvfits('{0}.fits'.format(msname),
                    spoof_nonessential=True)
    # Get the model to write to the data
    if flux is not None:
        fobs = UV.freq_array.squeeze()/1e9
        lst = UV.lst_array
        model = amplitude_sky_model(du.src('cal', ra, dec, flux),
                                    lst, pt_dec, fobs).T
        model = np.tile(model[:, :, np.newaxis], (1, 1, UV.Npols))
    else:
        model = np.ones((UV.Nblts, UV.Nfreqs, UV.Npols), dtype=np.complex64)

    if os.path.exists('{0}.ms'.format(msname)):
        shutil.rmtree('{0}.ms'.format(msname))
    importuvfits('{0}.fits'.format(msname),
                 '{0}.ms'.format(msname))

    # Changes these to use casacore instead
    with table('{0}.ms/ANTENNA'.format(msname), readonly=False) as tb:
        tb.putcol('POSITION',
              np.array([df_itrf['x_m'], df_itrf['y_m'], df_itrf['z_m']]).T)

    addImagingColumns('{0}.ms'.format(msname))
    with table('{0}.ms'.format(msname), readonly=False) as tb:
        tb.putcol('MODEL_DATA', model)

