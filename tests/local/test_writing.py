import pkg_resources
from dsamfs.meridian_fringestop import run_fringestopping
from pyuvdata import UVData
import json
import glob

def test_end2end(tmpdir):
    data_path = pkg_resources.resource_filename('dsamfs', 'data/')
    param_path = '{0}/test_parameters.yaml'.format(data_path)
    header_path = '{0}/test_header.txt'.format(data_path)
    run_fringestopping(param_path, header_file=header_path, outdir=tmpdir)
    fname = glob.glob('{0}/*.hdf5'.format(tmpdir))[0]
    UV = UVData()
    UV.read(fname, file_type='uvh5')
