# [FSC T] = fourier_shell_corr(img1,img2,dispfsc,SNRt,fig,hold_on)
# Computes the Fourier shell correlation between img1 and img2. It can also
# compute the threshold function T. Images can be complex-valued.
# dispfsc = 1;  % Display results
# SNRt          % Power SNR for threshold, popular options:
#               % SNRt = 0.5; 1 bit threshold for average
#               % SNRt = 0.2071; 1/2 bit threshold for average
# M. van Heela, and M. Schatzb, "Fourier shell correlation threshold criteria," Journal of Structural Biology 151, 250-262 (2005)
# Junjing Deng 29-Apr-2014


import numpy as np
from scipy.fft import fft2, fftshift

from .azimuthal import radial_data

#: Power SNR for the 1-bit threshold criterion.
SNRT_1BIT = 0.5
#: Power SNR for the 1/2-bit threshold criterion.
SNRT_HALF_BIT = 0.2071


def tukeywin(window_length, alpha=0.5):
    '''The Tukey window, also known as the tapered cosine window, can be regarded as a cosine lobe of width \alpha * N / 2 that is convolved with a rectangle window of width (1 - \alpha / 2). At \alpha = 1 it becomes rectangular, and at \alpha = 0 it becomes a Hann window.
    We use the same reference as MATLAB to provide the same results in case users compare a MATLAB output to this function
    output
    Reference
    ---------
    http://www.mathworks.com/access/helpdesk/help/toolbox/signal/tukeywin.html
    '''
    # Special cases
    if alpha <= 0:
        return np.ones(window_length) #rectangular window
    elif alpha >= 1:
        return np.hanning(window_length)
    # Normal case
    x = np.linspace(0, 1, window_length)
    w = np.ones(x.shape)
    # first condition 0 <= x < alpha/2
    first_condition = x<alpha/2
    w[first_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[first_condition] - alpha/2) ))
    # second condition already taken care of
    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = x>=(1 - alpha/2)
    w[third_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[third_condition] - 1 + alpha/2)))
    return w


def threshold_curve(n_half, SNRt=SNRT_HALF_BIT):
    """
    The van Heel & Schatz threshold T(r) for shells r = 0 .. n_half-1.

    Split out of :func:`fourier_shell_corr` so a caller that has already paid
    for the FFTs can switch between the 1-bit and 1/2-bit criteria without
    recomputing the correlation, which does not depend on ``SNRt`` at all.
    """
    r = np.arange(n_half, dtype=float)
    n = 2*np.pi*r
    n[0] = 1
    return ( SNRt + 2*np.sqrt(SNRt)/np.sqrt(n) + 1./np.sqrt(n)  ) /( SNRt + 2*np.sqrt(SNRt)/np.sqrt(n) + 1  )


def fourier_shell_corr(img1,img2,SNRt=SNRT_HALF_BIT,alpha=0.5,to_print=True):
    """
    (FSC, T, C) for two square images of the same shape.

    ``FSC`` is binned out to the corner of Fourier space and so is longer than
    the threshold ``T``, which stops at Nyquist -- compare them as
    ``FSC[:T.size]``. The frequency axis is not returned: shell ``i`` of the
    truncated curve sits at ``i / T.size`` in units of Nyquist.

    ``to_print`` is a plot switch, not a verbosity flag, and it blocks on
    matplotlib's ``show()``. Qt callers want ``to_print=False``.
    """
    t=tukeywin(img1.shape[0],alpha) #Tukey Window, L/8=0.125 for one side
    TW_2d=np.sqrt(np.outer(t,t))
    img1=img1*TW_2d
    img2=img2*TW_2d

    F1 = fftshift(fft2(img1))
    F2 = fftshift(fft2(img2))

    C = radial_data(F1 * np.conj(F2)).mean
    C1 =radial_data(np.abs(F1)**2).mean
    C2 =radial_data(np.abs(F2)**2).mean
    FSC = np.real(C)/(np.sqrt(np.abs(C1*C2)))

    r = np.arange(F1.shape[0]/2, dtype=float)
    T = threshold_curve(len(r), SNRt)

    if to_print:
        # Imported here so the module stays usable -- and importable -- without
        # matplotlib, which nothing but this branch needs.
        import matplotlib.pyplot as plt

        a,=plt.plot(r/(F1.shape[0]/2),FSC[:len(r)],'g',linewidth=1.)
        b,=plt.plot(r/(F1.shape[0]/2),T,'r--',linewidth=1.)

        if SNRt == SNRT_HALF_BIT:
            plt.legend([a,b],['FSC','1/2 bit threshold'])
        elif SNRt == SNRT_1BIT:
            plt.legend([a,b],['FSC','1 bit threshold'])
        else:
            plt.legend([a,b],['FSC','Threshold SNR = ' +str(SNRt)])
        plt.xlabel('Spatial frequency/Nyquist')
        plt.ylabel('Fourier Ring Correlation')
        plt.ylim(-0.1,1.1)
        plt.xlim(0,1.0)
        plt.show()
    return FSC, T, C
