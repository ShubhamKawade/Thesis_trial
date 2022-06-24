import matplotlib.pyplot as plt
import numpy as np
import math
import bces.bces
from scipy import stats
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def cleanup(data):
    data = data[data.Cluster != 'A1631']

    data['Cluster'] = [x.strip('+') for x in data.Cluster]
    data['Cluster'] = [x.strip('*') for x in data.Cluster]
    data['Cluster'] = [x.strip('!') for x in data.Cluster]
    
    data = data.reset_index(drop=True)

    return data


def plot_bestfit(X,scale_X, scale_Y, a, b):
    a_new = np.log10( np.multiply( 10**a, scale_Y) )
    def Y_theo_loglog(X, b, a_new):
        return 10**(b*X + a_new)
    x_linspace_new = np.log10( X / scale_X) 
    z = Y_theo_loglog( x_linspace_new, b, a_new) 
    return z 


def removing_galaxy_groups(data):
# =============================================================================
#     groups = ['A0168','A0189','A0262','A0400','A0407','A0957','A1060','A1142','A1185','A1991','A2064','A2151a','A2634','A2717','A2877','A3341','A3392','A3526+','A3558C','A3581','A3744','AWM4','AWM5','CAN010','CID28','HCG62','IC1262','IVZw038','MKW11','MKW4','MKW8','NGC1132','NGC1550','NGC4325','NGC4636','NGC5044+','NGC5846','NGC6338i','RBS0540','RXCJ1252.5-3116','RXCJ1304.2-3030','RXJ0123.2+3327','RXJ0123.6+3315','RXJ0228.2+2811','RXJ0341.3+1524','RXJ1205.1+3920','S0301','S0540','S0753','S0805','S0851','UGC03957','USGCS152','Zw0959.6+3257','ZwCl1665','ZwCl8338','A1035','A1648','A2399','NGC7556','PegasusII','RXCJ0340.6m0239','S0384','S0868','A2622','A3570','A3733','RXCJ1353.4m2753','RXJ1740.5p3539','RXCJ2104.9m5149','S0987','S0555','S1136','Zw1420.2p4952','NGC1650','CID36','RXCJ1337.4m4120','RXCJ1742.8p3900','A0194','RXCJ2124.3m7446','RXCJ1926.9m5342','S0112','A0076']
#     for i in range(len(groups)):
#         data.drop(data[data['Cluster']==f'{groups[i]}'].index, inplace=True)
# =============================================================================
    data = data[data.M_weighted_1e14 > 1]
    return data

def calculate_asymm_err(a_arr):
    a_med = np.median(a_arr)
    a_low = []
    a_high = []

    for val in a_arr:
        if val < a_med:
            a_low.append(val)
        elif val > a_med:
            a_high.append(val)
        

    a_low.sort(reverse=True)
    a_high.sort()

    si_fac = 0.683  # 0.683 1sigma, 0.95.. 2sigma......
    sig_index_a_low = math.ceil(len(a_low) * si_fac)  # OVERESTIMATES THE REAL UNCERTAINTY! a bit
    sig_index_a_high = math.ceil(len(a_high) * si_fac)

    sig_a_minus = np.abs(a_low[sig_index_a_low] - a_med)
    sig_a_plus = np.abs(a_high[sig_index_a_high] - a_med)

    return -sig_a_minus, sig_a_plus


def correct_psf(array):
    return -2.0027 + 1.6224 * array - 0.07727 * (array**2) + 0.00321 * (array**3)


def calculate_bestfit(x_array, x_err, y_array, y_err,cov=0):
    a, b, a_err, b_err, cov_ab = bces.bces.bces(x_array, x_err, y_array, y_err, cov)
    scatter = calculate_sig_intr_yx(x_array, y_array, x_err, y_err, a[0], b[0])

    return b[0], 10 ** b[0], a[0], scatter


def calculate_sig_intr_yx(data_x, data_y, sig_x, sig_y, b, a):
    # b=slope, a=norm
    sig_proj_sq_i = np.square(sig_y) + (b ** 2. * np.square(sig_x))
    # weights_i = (1. / sig_proj_sq_i) / ((1. / (len(data_x))) * np.sum(1. / sig_proj_sq_i))
    sig_raw_sq = (1. / (len(data_x) - 2.)) * np.sum(((data_y - b * data_x - a) ** 2.))  # Y-direction
    sig_int_i = (np.abs(sig_proj_sq_i - sig_raw_sq)) ** (1. / 2.)

    return np.mean(sig_int_i)


def calculate_chi_red(y_data, exp, y_err, x_err, scatter, slope):
    err = y_err ** 2 + (slope ** 2) * x_err ** 2 + scatter ** 2
    chisq = np.sum(((y_data-exp)**2)/err)

    return chisq / (len(y_data) - 2)


def calculate_sigma_dev(mean_distribution_1, error_distribution_1, mean_distribution_2, error_distribution_2):
    if mean_distribution_1 < mean_distribution_2:
        delta_1 = error_distribution_1[1]
        delta_2 = error_distribution_2[0]
    else:
        delta_1 = error_distribution_1[0]
        delta_2 = error_distribution_2[1]

    return np.round(_stat_significance(mean_distribution_1, delta_1, mean_distribution_2, delta_2), 2)


def _stat_significance(mean_distribution_1, delta_1, mean_distribution_2, delta_2):
    return np.abs(mean_distribution_1 - mean_distribution_2) / (delta_1 ** 2 + delta_2 ** 2) ** 0.5


def percent_diff(value_1, err_1, value_2, err_2, bestfit, err_bestfit):
    """ This function evaluates the percentage difference between statistics obtained from the 
        subsamples. The value is evalueated as :
            (value1 - value2)/ bestfit
            and the error is evaluated based on uncertainety propagation for (A-B/C)
    """
    if value_1 < value_2:
        delta1 = err_1[1]
        delta2 = err_2[0]
    else:
        delta1 = err_1[0]
        delta2 = err_2[1]
    uncertainty = ((bestfit**2)*(delta1**2+delta2**2) + ((err_bestfit**2)*(value_1 - value_2)**2)/bestfit**4)**0.5

    return np.round((value_1 - value_2)/bestfit, 4), np.round(uncertainty, 4)


def calculate_cdf(array, bins):
    count, bins_count = np.histogram(array, bins)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    return bins_count, cdf

def calculate_ks_stat(array1, array2):
    D, p = stats.ks_2samp(array1, array2)
    D_threshold = 1.358 * np.sqrt((len(array1) + len(array2))/ (len(array2) * len(array1)) )
    return np.round(D,5), np.round(p,5) , np.round(D_threshold,5)



def confidence_ellipse(xdata, ydata, x_bestfit, y_bestfit, ax, n_std=3, facecolor='none', **kwargs):
        """
        Create a plot of the covariance confidence ellipse of *x* and *y*.
    
        Parameters
        ----------
        x, y : array-like, shape (n, )
            Input data.
    
        ax : matplotlib.axes.Axes
            The axes object to draw the ellipse into.
    
        n_std : float
            The number of standard deviations to determine the ellipse's radiuses.
    
        **kwargs
            Forwarded to `~matplotlib.patches.Ellipse`
    
        Returns
        -------
        matplotlib.patches.Ellipse
        """
        if xdata.size != ydata.size:
            raise ValueError("x and y must be the same size")
    
        cov = np.cov(xdata, ydata)
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensionl dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                          facecolor=facecolor, **kwargs)
    
        # Calculating the stdandard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = x_bestfit
    
        # calculating the stdandard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = y_bestfit
    
        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)
    
        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)


def plot_confidence_ellipse(xdata ,ydata ,x_bestfit ,y_bestfit,n_std,ax ):
    
    #fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xdata,ydata)
    
    
    
    
    
    
    confidence_ellipse(xdata, ydata,x_bestfit,y_bestfit, ax, n_std=1,
                       label=r'$1\sigma$ contour', edgecolor='firebrick', lw = 2)
    #plt.show()
# =============================================================================
#     #confidence_ellipse(x, y, ax_nstd, n_std=2,
#                       # label=r'$2\sigma$', edgecolor='green', lw = 2)
#     confidence_ellipse(xdata, ydata, ax_nstd, n_std=3,
#                        label=r'$3\sigma$ contour', edgecolor='blue', lw = 2)
#     #plt.plot(np.mean(x),np.mean(y),'ro',label = f'Slope = {np.round(np.mean(x),3)}, Norm = {np.round(np.mean(y),3)}')
#     plt.plot(x_bestfit,y_bestfit,'ro',label = f'Best fit ({x_bestfit,y_bestfit})')
#     
#     #ax_nstd.scatter(mu[0], mu[1], c='red', s=3)
#     ax_nstd.set_title('')
#     plt.xlim(1.7,2.7)
#     plt.ylim(1.0,1.6)
#     ax_nstd.legend()
#     plt.xlabel('Slope')
#     plt.ylabel('Normalization')
#     plt.title('Contours for disturbed clusters (500 iterations)')
#     
#     plt.savefig('LT_normVslope_relaxed.png',dpi = 400)
# =============================================================================
    
def conf_band_lim(x_data, Norm, Slope, err_Norm, err_Slope):
    def lin_fit(X, Slope, Norm):
        return Norm * X ** Slope

    cb_plus = lin_fit(x_data, Slope + err_Slope[1], Norm + err_Norm[1])
    cb_minus = lin_fit(x_data, Slope-err_Slope[0], Norm-err_Norm[0])
    cb_plus_minus = lin_fit(x_data, Slope+err_Slope[1], Norm-err_Norm[0])
    cb_minus_plus = lin_fit(x_data, Slope-err_Slope[0], Norm+err_Norm[1])
    lcb = np.minimum(cb_minus, cb_plus_minus)
    ucb = np.maximum(cb_plus, cb_minus_plus)
     

    return lcb, ucb


def calculate_BIC(y_data, z, k ):
    n = len(y_data) # number of data points
    k = 3 # model free parameters
    summ = 0 # summatory of (data - model)**2
    for i in range(n):
        diffsqrt = (y_data[i] - z[i])**2
        summ = summ + diffsqrt
    rsos = (1/n)*summ
    
    BIC = n*np.log(rsos/n) + k*np.log(n)

    return BIC
