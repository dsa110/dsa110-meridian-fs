"""Routines for running meridian fringestopping of DSA-110 data.

Dana Simard, dana.simard@astro.caltech.edu, 2020
"""
import socket
import subprocess
import numpy as np
import astropy.units as u
from psrdada import Reader
import dsautils.dsa_syslog as dsl
from dsautils import cnf
import dsamfs.utils as pu
from dsamfs.io import dada_to_uvh5

logger = dsl.DsaSyslogger()
logger.subsystem("software")
logger.app("dsamfs")

def run_fringestopping(param_file=None, header_file=None, output_dir=None, working_dir=None):
    """Read in data, fringestop on zenith, and write to hdf5 file.
    Parameters
    ----------
    param_file : str
        The full path to the json parameter file. Defaults to the file
        meridian_fringestopping_parameters.py in the package directory.
    """
    if working_dir is None:
        working_dir = "/home/ubuntu/data/"

    # Read in parameter file
    test, key_string, nant, nchan, npol, fobs, samples_per_frame, \
        samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, \
        tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd = \
        pu.parse_params(param_file)
    nbls = (nant * (nant + 1)) // 2
    key = int(f"0x{key_string}", 16)

    hostname = socket.gethostname()
    conf = cnf.Conf()
    subband = list(conf.get("corr")['ch0'].keys()).index(hostname)

    # Update outrigger delays and refmjd in etcd
    pu.put_outrigger_delays(outrigger_delays)
    pu.put_refmjd(refmjd)

    # Generate fringestopping table and load visibility model
    fs_table = (
        f"fringestopping_table_dec{(pt_dec*u.rad).to_value(u.deg):.1f}deg_"
        f"{len(antenna_order)}ant.npz")
    fs_table = f"{working_dir}/{fs_table}"
    bname, blen, uvw = pu.baseline_uvw(antenna_order, pt_dec, refmjd, casa_order=False)
    vis_model = pu.load_visibility_model(
        fs_table, blen, nint, fobs, pt_dec, tsamp, antenna_order, outrigger_delays, bname, refmjd)
    if not fringestop:
        vis_model = np.ones(vis_model.shape, vis_model.dtype)

    logger.info(
        f"Started fringestopping of dada buffer {key_string} with {nint} "
        f"integrations and {len(blen)} baselines.")

    if test:
        # Setup the dada buffer for testing
        sample_rate = 1 / 0.134217728
        header_size = 4096
        buffer_size = int(4 * nbls * npol * nchan * samples_per_frame * 2)
        data_rate = buffer_size * (sample_rate / samples_per_frame) / 1e6
        with subprocess.Popen(
                ["dada_db", "-a", str(header_size), "-b", str(buffer_size), "-k", key_string],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
        ) as p_create:
            outs, errs = p_create.communicate(timeout=15)

            if p_create.returncode != 0:
                print(errs.decode("utf-8"))
                logger.info(errs.decode("utf-8"))
                raise RuntimeError("Dada buffer could not be created.")

        print(outs.decode("utf-8"))

    print(f"Initializing reader: {key_string}")
    reader = Reader(key)

    if test:
        p_write = subprocess.Popen(
            ["dada_junkdb", "-r", str(data_rate), "-t", "60", "-k", key_string, header_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)

    # Get the start time and the sample time from the reader
    sample_rate_out = 1 / (tsamp * nint)

    # Read in psrdada buffer, fringestop, and write to uvh5
    dada_to_uvh5(
        reader, output_dir, working_dir, nbls, nchan, npol, nint, nfreq_int,
        samples_per_frame_out, sample_rate_out, pt_dec, antenna_order,
        fs_table, tsamp, bname, uvw, fobs,
        vis_model, test, filelength_minutes, subband
    )

    if test:
        outs, errs = p_write.communicate(timeout=15)
        if p_write.returncode != 0:
            logger.info(errs.decode("utf-8"))
            print(errs.decode("utf-8"))
            raise RuntimeError("Error in writing to dada buffer.")

        print(outs.decode("utf-8"))
        print(errs.decode("utf-8"))

        with subprocess.Popen(
                ["dada_db", "-d", "-k", key_string],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
        ) as p_kill:
            outs, errs = p_kill.communicate(timeout=15)
            if p_kill.returncode != 0:
                logger.info(errs.decode("utf-8"))
                print(errs.decode("utf-8"))
            else:
                print(outs.decode("utf-8"))

    logger.info(f"Disconnected from psrdada buffer {key_string}")
