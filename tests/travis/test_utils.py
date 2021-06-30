from dsamfs.utils import parse_params

def test_parse_params():
    test, key_string, nant, nchan_spw, npol, fobs, \
    samples_per_frame, samples_per_frame_out, nint, \
    nfreq_int, antenna_order, pt_dec, tsamp, fringestop, \
    filelength_minutes, outrigger_delays = parse_params()
    assert nant == len(antenna_order)
    
