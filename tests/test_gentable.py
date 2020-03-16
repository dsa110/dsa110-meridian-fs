import pytest
from dsamfs import fringestopping
from antpos import utils


def test_gentable():
    df_bls = fringestopping.get_baselines(antenna_order,autocorrs=True,casa_order=False)
    blen = np.array([df_bls['x_m'],df_bls['y_m'],df_bls['z_m']]).T
    utils.generate_fringestopping_table(blen,nint,outname=fs_table)
    assert os.path.exists(fs_table)                   
