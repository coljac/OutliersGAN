import os
import glob
import numpy as np
from astropy.io import fits

import matplotlib.pyplot as plt

def to_fits(y, name, params=(None,None,None), x=None, labels=('wave','flux')):
    """
    Saves 2D data in the FITS format. 
    If 'x' is provided the spectra are saved into a Table, otherwise it encodes the x dimension using WCS. 
    All spectra will be stored to the same file under a ImageHDU each. 
    If you wish to store one spectra per FITS file, just call this function multiple times!
    
    Arguments:
    -> y: y data (list of lists, where each inner list corresponds to one spectrum)
    -> name: name of the fits file without including the extension
    -> params: wcs conversion values in Angstroms (crpix, cdelt, crval)
    -> x: x data (list of lists, where each inner list corresponds to one spectrum)
    -> labels: name of the x and y variable (xlabel, ylabel)

    Returns:
    -> nothing
    """    
    if x is not None:
        from astropy.table import Table
        t = Table([x, y], names=(labels[0], labels[1]))
        t.write(name, format='fits')

    else:
        from astropy import wcs
        w = wcs.WCS(naxis=1)
        w.wcs.crpix = [params[0]]
        w.wcs.cdelt = np.array([params[1]])
        w.wcs.crval = [params[2]]
        w.wcs.ctype = ["WAVE"]
        w.wcs.cunit = ["Angstrom"]    

        header = w.to_header()
        hdu1 = fits.PrimaryHDU()

        hdu_im = []
        for i in range(len(y)):
            hdu_im.append(fits.ImageHDU(header=header))
            hdu_im[-1].data = y[i]

        hdul = fits.HDUList([hdu1, *hdu_im])
        hdul.writeto(name + '.fits')
