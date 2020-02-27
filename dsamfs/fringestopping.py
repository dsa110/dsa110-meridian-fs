"""
DSAMFS/FRINGESTOPPING.PY

Dana Simard, dana.simard@astro.caltech.edu 11/2019

Casa-based routines for calculating and applying fringe-stopping phases
to visibilities
"""
import __casac__ as cc
import numpy as np
import astropy.units as u
from dsacalib import constants as ct
from dsacalib.utils import to_deg
from numba import jit
from astropy.time import Time
from scipy.special import j1
from astropy.coordinates.angle_utilities import angular_separation
from dsacalib.fringestopping import calc_uvw

def generate_fringestopping_table(b,nint=ct.nint,tsamp=ct.tsamp,pt_dec=ct.pt_dec,
                                  outname='fringestopping_table',mjd0=58849.0):
    """Generates a table of the w vectors towards a source to use in fringe-
    stopping.  And writes it to a numpy pickle file named 
    fringestopping_table.npz
    
    Args:
      b: array or list
        shape (nbaselines, 3), the lengths of the baselines in ITRF coordinates
      nint: int
        the number of time integrations to calculate the table for
      tsamp: float
        the sampling time in seconds
      pt_dec: str
        the pointing declination, eg '+73d00m00.0000s'
      method: str
        either 'A' (reference w to ha=0 dec=0 at midpoint of observation), 
        'B' (reference w to ha=0, dec=pt_dec at midpoint of observation)
      mjd0: float
        the mjd of the start time
        
    Returns:
      Nothing
    """
    dt = np.arange(nint)*tsamp
    dt = dt - np.median(dt)
    ha = dt * 360/ct.seconds_per_sidereal_day
    bu,bv,bw = calc_uvw(b,mjd0+dt/ct.seconds_per_day,'HADEC',
                       ha*u.deg, np.ones(ha.shape)*to_deg(ct.pt_dec))
    if nint%2 == 1:
        bwref = bw[:,(nint-1)//2]
    else:
        bu,bv,bwref = calc_uvw(b,mjd0,'HADEC',
                              0.*u.deg,to_deg(ct.pt_dec))
        bwref = bwref.squeeze()
    bw = bw - bwref[:,np.newaxis]
    bw = bw.T
    bwref = bwref.T
    np.savez(outname,
             dec=pt_dec,ha=ha,bw=bw,bwref=bwref)
    return

def fringestop_on_zenith_worker(vis,vis_model,nint,nbl,nchan,npol):
    vis.shape=(-1,nint,nbl,nchan,npol)
    vis /= vis_model
    return vis.mean(axis=1)

def zenith_visibility_model(fobs,fstable='fringestopping_table.npz'):
    data = np.load(fstable)
    bws = data['bw']
    vis_model = np.exp(2j*np.pi/ct.c_GHz_m * fobs[:,np.newaxis] *
                       bws[np.newaxis,:,:,np.newaxis,np.newaxis])
    return vis_model

def fringestop_on_zenith(vis,vis_model,nint,nans=False):
    """Fringestops on HA=0, DEC=pointing declination for the midpoint of each   
    integration and then integrates the data.  The number of samples to integrate 
    by is set by the length of the bw array in bw_file.
    
    Args:
      vis: complex array
        The input visibilities, dimensions (baselines,time,freq,pol)
      fobs: real array
        The central frequency of each channel, in GHz
      bw_file: str
        The path to the .npz file containing the bw array (the length of the
        w-vector in m)
      nint: int
        the number of samples (in time) to integrate
      
    Returns:
      vis: complex array
        The fringe-stopped and integrated visibilities
    """
    # Remove the padding to save time - assume the user has done this correctly
#     if vis.shape[0]%nint != 0:
#         npad = nint - vis.shape[0]%nint
#         print('Warning: Padding array to integrate.  Last bin contains only {0}% real data.'.format((nint-npad)/nint*100))
#         vis = np.pad(vis,((0,npad),(0,0),(0,0),(0,0)),mode='constant',
#                  constant_values=(np.nan,))
#         nans = True
    nt,nbl,nchan,npol = vis.shape
    if not nans:
        vis = fringestop_on_zenith_worker(vis,vis_model,nint,nbl,nchan,npol)
    else:
        vis = vis.reshape(-1,nint,nbl,nchan,npol)/vis_model
        vis = np.nanmean(vis,axis=1)
    return vis

def write_fs_delay_table(msname,source,blen,tobs,nant):
    """Writes the delays needed to fringestop on a source to a delay calibration
    table in casa format. 
    
    Not tested.
    
    Args:
      msname: str
        The prefix of the ms for which this table is generated
        Note: doesn't open the ms
      source: src class
        The source (or location) to fringestop on
      blen: array, float
        The ITRF coordinates of the baselines
      tobs: array, float
        The observation time of each time bin in mjd
      nant: int
        The number of antennas in the array
        
    Returns:
    """
    nt = tobs.shape[0]
    bu,bw,bw = calc_uvw(blen, tobs, source.epoch, source.ra, source.dec)
    
    ant_delay = np.zeros((nt,nant))
    ant_delay[:,1:] = bw[:nant-1,:].T/ct.c_GHz_m
    
    error = 0
    tb = cc.table.table()
    error += not tb.open('{0}/templatekcal'.format(ct.pkg_data_path))
    error += not tb.copy('{0}_{1}_fscal'.format(msname,source.name))
    error += not tb.close()
    
    error += not tb.open('{0}_{1}_fscal'.format(msname,source.name),nomodify=False)
    error += not tb.addrows(nant*nt - tb.nrows())
    error += not tb.flush()
    assert(tb.nrows() == nant*nt)
    error += not tb.putcol('TIME',np.tile((tobs*u.d).to_value(u.s).reshape(-1,1),(1,nant)).flatten())
    error += not tb.putcol('FIELD_ID',np.zeros(nt*nant,dtype=np.int32))
    error += not tb.putcol('SPECTRAL_WINDOW_ID',np.zeros(nt*nant,dtype=np.int32))
    error += not tb.putcol('ANTENNA1',np.tile(np.arange(nant,dtype=np.int32).reshape(1,nant),(nt,1)).flatten())
    error += not tb.putcol('ANTENNA2',-1*np.ones(nt*nant,dtype=np.int32))
    error += not tb.putcol('INTERVAL',np.zeros(nt*nant,dtype=np.int32))
    error += not tb.putcol('SCAN_NUMBER',np.ones(nt*nant,dtype=np.int32))
    error += not tb.putcol('OBSERVATION_ID',np.zeros(nt*nant,dtype=np.int32))
    error += not tb.putcol('FPARAM',np.tile(ant_delay.reshape(1,-1),(2,1)).reshape(2,1,-1))
    error += not tb.putcol('PARAMERR',np.zeros((2,1,nt*nant),dtype=np.float32))
    error += not tb.putcol('FLAG',np.zeros((2,1,nt*nant),dtype=bool))
    error += not tb.putcol('SNR',np.zeros((2,1,nt*nant),dtype=np.float64))
    #error += not tb.putcol('WEIGHT',np.ones((nt*nant),dtype=np.float64)) 
    # For some reason, WEIGHT is empty in the template ms, so we don't need to
    # modify it
    error += not tb.flush()
    error += not tb.close()
    
    if error > 0:
        print('{0} errors occured during calibration'.format(error))
    
    return