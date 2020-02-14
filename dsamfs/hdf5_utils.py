"""
HDF5_UTILS.PY

Dana Simard, dana.simard@astro.caltech.edu, 02/2020

Routines to interact w/ hdf5 files used for the 24-hr 
visibility buffer
"""

# To do:
# Replace to_deg w/ astropy versions

import __casac__ as cc
import astropy.io.fits as pf
import astropy.units as u
import numpy as np
from astropy.time import Time
from dsacalib import constants as ct
from antpos.utils import *
import h5py
from dsacalib.utils import convert_to_ms, get_antpos_itrf

from astropy.utils import iers
iers.conf.iers_auto_url_mirror = ct.iers_table

def read_hdf5_file(fl,source=None,dur=50*u.min,autocorrs=False,
                   badants=None,quiet=True):
    if source is not None:
        stmid = source.ra.to_value(u.rad)
        seg_len = (dur/2*(15*u.deg/u.h)).to_value(u.rad)
        
    with h5py.File(fl, 'r') as f:
        antenna_order = list(f['antenna_order'][...])
        nant = len(antenna_order)
        fobs = f['fobs_GHz'][...]
        mjd = f['time_mjd_seconds'][...]/ct.seconds_per_day
        nt = len(f['time_mjd_seconds'])
        tsamp  = (mjd[0] - mjd[-1])/(nt-1)*ct.seconds_per_day


        st0 = Time(mjd[0], format='mjd').sidereal_time(
            'apparent',longitude=ct.ovro_lon*u.rad).radian 
    
        st  = np.angle(np.exp(1j*(st0 + 2*np.pi/ct.seconds_per_sidereal_day*
                          np.arange(nt)*tsamp)))
    
        st1 = Time(mjd[-1],format='mjd').sidereal_time(
            'apparent',longitude=ct.ovro_lon).radian

        if source is not None:
            if not quiet:
                print("\n-------------EXTRACT DATA--------------------")
                print("Extracting data around {0}".format(stmid*180/np.pi))
                print("{0} Time samples in data".format(nt))
                print("LST range: {0:.1f} --- ({1:.1f}-{2:.1f}) --- {3:.1f}deg".format(st[0]*180./np.pi,(stmid-seg_len)*180./np.pi, (stmid+seg_len)*180./np.pi,st[-1]*180./np.pi))

            I1 = np.argmax(np.absolute(np.exp(1j*st)+
                           np.exp(1j*stmid)*np.exp(-1j*seg_len)))
            I2 = np.argmax(np.absolute(np.exp(1j*st)+
                           np.exp(1j*stmid)*np.exp(1j*seg_len)))
            transit_idx = np.argmax(np.absolute(np.exp(1j*st)+
                           np.exp(1j*(stmid))))
            
            mjd = mjd[I1:I2]
            st  = st[I1:I2]
            dat = f['vis'][I1:I2,...]
            if not quiet:
                print("Extract: {0} ----> {1} sample; transit at {2}".format(I1,I2,I0))
                print("----------------------------------------------")

        else:
            dat = f['vis'][...]
            transit_idx = None

    # Now we have to extract the correct baselines
    auto_bls = []
    cross_bls = list(range(int(nant/2*(nant+1))))
    i=-1
    for j in range(1,nant+1):
        i += j
        auto_bls += [i]
        cross_bls.remove(i)

    basels = auto_bls if autocorrs else cross_bls
    
    # Fancy indexing can have downfalls and may change in future numpy versions
    # See issue here https://github.com/numpy/numpy/issues/9450
    vis = dat[:,basels,...]
    assert vis.shape[0] == len(mjd)
    assert vis.shape[1] == len(basels)

    
    if autocorrs:
        bname = np.array([[a,a] for a in antenna_order])
        blen  = np.zeros((len(antenna_order),3))
        if badants is not None:
            badants = [str(ba) for ba in badants]
            good_idx = list(range(len(antenna_order)))
            for badant in badants:
                good_idx.remove(antenna_order.index(badant))
            vis = vis[:,good_idx,...]
            bname = bname[good_idx,...]
            blen = blen[good_idx,...]
    
    if not autocorrs:
        df_bls = get_baselines(antenna_order,autocorrs=False,casa_order=True)
        blen   = np.array([df_bls['x_m'],df_bls['y_m'],df_bls['z_m']]).T
        bname  = [bn.split('-') for bn in df_bls['bname']]
        if badants is not None:
            bname = np.array(bname)
            blen = np.array(blen)
            good_idx = list(range(len(bname)))
            for i,bn in enumerate(bname):
                if (bn[0] in badants) or (bn[1] in badants):
                    good_idx.remove(i)
            vis = vis[:,good_idx,...]
            blen = blen[good_idx,...]
            bname = bname[good_idx,...]
        
    if badants is not None:
        #badants = [str(ba) for ba in badants]
        for badant in badants:
            antenna_order.remove(badant)
            
    assert vis.shape[0] == len(mjd)
    vis = vis.swapaxes(0,1)
    
    dt = np.median(np.diff(mjd))
    if len(mjd)>0:
        tstart = mjd[0]-dt/2
        tstop = mjd[-1]+dt/2
    else:
        tstart = None
        tstop = None
    
    if type(bname) is not list:
        bname = bname.tolist()
        
    return fobs, blen, bname, tstart, tstop, vis, mjd, transit_idx, antenna_order


def initialize_hdf5_file(f,fobs,antenna_order,t0,nbls,nchan,npol,nant):
    """Initialize the hdf5 file.
    
    Args:
        f: hdf5 file handler
            the file
        fobs: array(float)
            the center frequency of each channel
        antenna_order: array(int)
            the order of the antennas in the correlator
        t0: float
            the time of the first sample in mjd seconds
        nbls: int
            the number of baselines
        nchan: int
            the number of channels
        npol: int
            the number of polarizations
        nant: int
            the number of antennas
        
    Returns:
        vis_ds: hdf5 dataset
            the dataset for the visibilities
        t_ds: hdf5 dataset
            the dataset for the times
    """
    ds_fobs = f.create_dataset("fobs_GHz",(nchan,),dtype=np.float32,data=fobs)
    ds_ants = f.create_dataset("antenna_order",(nant,),dtype=np.int,data=antenna_order)
    t_st = f.create_dataset("tstart_mjd_seconds",
                           (1,),maxshape=(1,),
                           dtype=int,data=t0)

    vis_ds = f.create_dataset("vis", 
                            (0,nbls,nchan,npol), 
                            maxshape=(None,nbls,nchan,npol),
                            dtype=np.complex64,chunks=True,
                            data = None)
    t_ds = f.create_dataset("time_seconds",
                           (0,),maxshape=(None,),
                           dtype=np.float32,chunks=True,
                           data = None)
    return vis_ds, t_ds