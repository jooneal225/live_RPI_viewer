import numpy as np


class radialDat:
    """Empty object container."""

    def __init__(self):
        self.mean = None
        self.std = None
        self.median = None
        self.numel = None
        self.max = None
        self.min = None
        self.r = None


def radial_data(data,annulus_width=1,working_mask=None,x=None,y=None,rmax=None):
    """
    r = radial_data(data,annulus_width,working_mask,x,y)

    A function to reduce an image to a radial cross-section.

    :INPUT:
      data   - whatever data you are radially averaging.  Data is
              binned into a series of annuli of width 'annulus_width'
              pixels.

      annulus_width - width of each annulus.  Default is 1.

      working_mask - array of same size as 'data', with zeros at
                        whichever 'data' points you don't want included
                        in the radial data computations.

      x,y - coordinate system in which the data exists (used to set
               the center of the data).  By default, these are set to
               integer meshgrids

      rmax -- maximum radial value over which to compute statistics

    :OUTPUT:
        r - a data structure containing the following
                   statistics, computed across each annulus:

          .r      - the radial coordinate used (outer edge of annulus)

          .mean   - mean of the data in the annulus

          .std    - standard deviation of the data in the annulus

          .median - median value in the annulus

          .max    - maximum value in the annulus

          .min    - minimum value in the annulus

          .numel  - number of elements in the annulus
    """

    #---------------------
    # Set up input parameters
    #---------------------
    data = np.array(data)

    if working_mask is None:
        working_mask = np.ones(data.shape,bool)

    npix, npiy = data.shape
    if x is None or y is None:
        x1 = np.arange(-npix/2.,npix/2.)
        y1 = np.arange(-npiy/2.,npiy/2.)
        x,y = np.meshgrid(y1,x1)

    r = abs(x+1j*y)

    if rmax is None:
        rmax = r[working_mask].max()

    #---------------------
    # Prepare the data container
    #---------------------
    dr = np.abs(x[0,0] - x[0,1]) * annulus_width
    radial = np.arange(rmax/dr)*dr + dr/2.
    nrad = len(radial)
    radialdata = radialDat()
    radialdata.mean = np.zeros(nrad,complex)
    radialdata.std = np.zeros(nrad,complex)
    radialdata.median = np.zeros(nrad,complex)
    radialdata.numel = np.zeros(nrad,complex)
    radialdata.max = np.zeros(nrad,complex)
    radialdata.min = np.zeros(nrad,complex)
    radialdata.r = radial

    #---------------------
    # Bin once, then walk the bins
    #---------------------
    # The annuli are [irad*dr, (irad+1)*dr), so the bin a pixel lands in is
    # just floor(r/dr) -- no need to re-scan the whole array once per annulus.
    # Sorting by that index makes each annulus a contiguous slice, which turns
    # an O(npix * nrad) sweep into one O(npix log npix) sort plus nrad cheap
    # reductions over small slices.
    bin_index = np.floor(np.asarray(r, dtype=float) / dr).astype(np.int64)
    keep = np.asarray(working_mask, dtype=bool) & (bin_index >= 0) & (bin_index < nrad)

    flat_bins = bin_index[keep]
    flat_data = data[keep]
    order = np.argsort(flat_bins, kind='stable')
    sorted_bins = flat_bins[order]
    sorted_data = flat_data[order]
    # Slice boundaries for every annulus in one pass
    edges = np.searchsorted(sorted_bins, np.arange(nrad + 1))

    for irad in range(nrad):
      chunk = sorted_data[edges[irad]:edges[irad + 1]]
      if chunk.size == 0:
        radialdata.mean[irad] = np.nan
        radialdata.std[irad]  = np.nan
        radialdata.median[irad] = np.nan
        radialdata.numel[irad] = np.nan
        radialdata.max[irad] = np.nan
        radialdata.min[irad] = np.nan
      else:
        radialdata.mean[irad] = chunk.mean()
        radialdata.std[irad]  = chunk.std()
        radialdata.median[irad] = np.median(chunk)
        radialdata.numel[irad] = chunk.size
        radialdata.max[irad] = chunk.max()
        radialdata.min[irad] = chunk.min()

    #---------------------
    # Return with data
    #---------------------

    return radialdata
