"""
meridian_fringestopping.py
dana.simard@astro.caltech.edu, Feb 2020

Reads correlated data from a psrdada
ringbuffer.  Fringestops on the meridian for each integrated
sample, before integrating the data and writing it to a hdf5 file.
"""
import sys
import subprocess
from psrdada import Reader
import dsacalib.constants as ct
import dsautils.dsa_syslog as dsl
import dsamfs.psrdada_utils as pu
from dsamfs.uvh5_utils import dada_to_uvh5
import dsamfs

logger = dsl.DsaSyslogger()
logger.subsystem("software")
logger.app("dsamfs")

def run_fringestopping(param_file, header_file=None, outdir=None):
    """Read in data, fringestop on zenith, and write to hdf5 file.
    Parameters
    ----------
    param_file : str
        The full path to the json parameter file. Defaults to the file
        meridian_fringestopping_parameters.py in the package directory.
    """
    if param_file is None:
        param_file = '{0}/meridian_fringestopping_parameters.json'.format(
            dsamfs.__path__[0])
    test, key_string, nant, nchan, npol, fobs, fout, samples_per_frame, \
        samples_per_frame_out, nint, fs_table, antenna_order, pt_dec, tsamp = \
        pu.parse_param_file(param_file)
    if outdir is not None:
        fout = '{0}/{1}'.format(outdir, fout)
        fs_table = '{0}/{1}'.format(outdir, fs_table)
    nbls = (nant*(nant+1))//2
    key = int('0x{0}'.format(key_string), 16)
    bname, blen, uvw = pu.baseline_uvw(antenna_order, pt_dec)

    logger.info("Started fringestopping of dada buffer {0} with {1} "
                "integrations and {2} baselines. Fringestopped data written "
                "to {3}.hdf5".format(key_string, nint, nbls, fout))

    if test:
        sample_rate = 1/0.134217728
        header_size = 4096
        buffer_size = int(4*nbls*npol*nchan*samples_per_frame*2)
        data_rate = buffer_size*(sample_rate/samples_per_frame)/1e6
        p_create = subprocess.Popen(
            ["dada_db", "-a", str(header_size), "-b", str(buffer_size), "-k",
             key_string], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        outs, errs = p_create.communicate(timeout=15)
        if p_create.returncode != 0:
            print(errs.decode("utf-8"))
            logger.info(errs.decode("utf-8"))
            raise RuntimeError('Dada buffer could not be created.')
        print(outs.decode("utf-8"))

    print('Initializing reader: {0}'.format(key_string))
    reader = Reader(key)

    if test:
        p_write = subprocess.Popen(
            ["dada_junkdb", "-r", str(data_rate), "-t", "60", "-k", key_string,
             header_file], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)

    # Get the start time and the sample time from the reader
    # tstart, tsamp = pu.read_header(reader)
    # tstart += nint*tsamp/2
    tstart = 58871.66878472222*ct.SECONDS_PER_DAY
    #tsamp = 0.134217728
    tstart += nint*tsamp/2
    t0 = int(tstart)
    tstart -= t0
    sample_rate_out = 1/(tsamp*nint)

    vis_model = pu.load_visibility_model(fs_table, blen, nant, nint, fobs,
                                         pt_dec, tsamp)
    dada_to_uvh5(reader, fout, nbls, nchan, npol, nint, samples_per_frame_out,
                 sample_rate_out, pt_dec, antenna_order, fs_table, tstart,
                 tsamp, bname, uvw, fobs, vis_model)

    if test:
        outs, errs = p_write.communicate(timeout=15)
        if p_write.returncode != 0:
            logger.info(errs.decode("utf-8"))
            print(errs.decode("utf-8"))
            raise RuntimeError('Error in writing to dada buffer.')
        print(outs.decode("utf-8"))
        p_kill = subprocess.Popen(
            ["dada_db", "-d", "-k", key_string], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        outs, errs = p_kill.communicate(timeout=15)
        if p_kill.returncode != 0:
            logger.info(errs.decode("utf-8"))
            print(errs.decode("utf-8"))
        else:
            print(outs.decode("utf-8"))

    logger.info("Disconnected from psrdada buffer {0} and closed file "
                "{1}.hdf5".format(key_string, fout))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        PARAM_FILE = sys.argv[1]
        if len(sys.argv) > 2:
            outdir = sys.argv[2]
        else:
            outdir = None
    else:
        PARAM_FILE = None
    run_fringestopping(PARAM_FILE, outdir=outdir)

