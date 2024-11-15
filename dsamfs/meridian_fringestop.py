"""
meridian_fringestopping.py
dana.simard@astro.caltech.edu, Feb 2020

Reads correlated data from a psrdada
ringbuffer.  Fringestops on the meridian for each integrated
sample, before integrating the data and writing it to a hdf5 file.
"""
import sys
from dsamfs.routines import run_fringestopping

if len(sys.argv) > 1:
    OUTDIR = sys.argv[1]
else:
    OUTDIR = None

if len(sys.argv) > 2:
    WORKING_DIR = sys.argv[2]
else:
    WORKING_DIR = None

if len(sys.argv) > 3:
    PARAM_FILE = sys.argv[3]
else:
    PARAM_FILE = None

if len(sys.argv) > 4:
    HEADER_FILE = sys.argv[4]
else:
    HEADER_FILE = None

run_fringestopping(
    PARAM_FILE, header_file=HEADER_FILE, output_dir=OUTDIR, working_dir=WORKING_DIR)
