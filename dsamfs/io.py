"""DSAMFS/IO.PY

Routines to read and manipulate the correlator-data psrdada buffer stream and
write the correlated data to a uvh5 file.

Dana Simard, dana.simard@astro.caltech.edu, 2020
"""

from datetime import datetime
import os
import traceback
import socket
import numpy as np
import h5py
import astropy.units as u
from psrdada.exceptions import PSRDadaError
from antpos.utils import get_itrf
import dsautils.dsa_syslog as dsl
import dsautils.dsa_store as ds
import dsacalib.constants as ct
import dsamfs.utils as pu
from dsamfs.fringestopping import fringestop_on_zenith

etcd = ds.DsaStore()

logger = dsl.DsaSyslogger()
logger.subsystem("software")
logger.app("dsamfs")

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
    df = get_itrf(
        latlon_center=(ct.OVRO_LAT*u.rad, ct.OVRO_LON*u.rad, ct.OVRO_ALT*u.m)
    )
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
    snapdelays = pu.get_delays(np.array(antenna_order), nants_telescope)
    extra["applied_delays_ns"] = np.string_(
        ' '.join([str(d) for d in snapdelays.flatten()])
    )
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
    bname : list(str)
        The name of each baseline.
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
                 fs_table, tsamp, bname, uvw, fobs,
                 vis_model, test, nmins):
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
    max_frames_per_file = int(np.ceil(nmins*60*sample_rate_out))
    hostname = socket.gethostname()
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
                            data.shape[0], data.shape[1], nchan,
                            nfreq_int, npol),
                                         axis=3)
                        nsamples = np.nanmean(nsamples.reshape(
                            nsamples.shape[0], nsamples.shape[1], nchan,
                            nfreq_int, npol), axis=3)

                update_uvh5_file(
                    fhdf5, data, t, tsamp, bname, uvw,
                    nsamples
                )

                idx_frame_out += 1
                idx_frame_file += 1
                print('Integration {0} done'.format(idx_frame_out))
        os.rename('{0}_incomplete.hdf5'.format(fout), '{0}.hdf5'.format(fout))
        try:
            etcd.put_dict(
                '/cmd/cal',
                {
                    'cmd': 'rsync',
                    'val':
                    {
                        'hostname': hostname,
                        'filename': '{0}.hdf5'.format(fout)
                    }
                }
            )
        except:
            logger.info('Could not reach ETCD to transfer {0} from {1}'.format(fout, hostname))
    try:
        reader.disconnect()
    except PSRDadaError:
        pass
