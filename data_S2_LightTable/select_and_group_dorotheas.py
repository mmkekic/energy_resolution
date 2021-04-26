from invisible_cities.io.dst_io import load_dst, load_dsts, df_writer
from glob import glob
import numpy as np
import pandas as pd
import tables as tb
import re
import matplotlib.pyplot as plt

from krcal.core.selection_functions import selection_in_band
from krcal. core.correction_functions import e0_xy_correction
import os
from invisible_cities.reco.corrections     import read_maps
from invisible_cities.reco.corrections     import norm_strategy
from scipy.optimize import curve_fit




def load_and_sel_data(fname):
    kdst = load_dst (fname, 'DST', 'Events')
    kdst = kdst[(kdst.nS1==1) & (kdst.nS2==1) & (kdst.R<=198)]
    maxZ = np.quantile(kdst.Z, 0.99)
    kdst = kdst[kdst.Z<maxZ]

    file_bootstrap_map = '$ICDIR/database/test_data/kr_emap_xy_100_100_r_6573_time.h5'
    file_bootstrap_map = os.path.expandvars(file_bootstrap_map)
    boot_map      = read_maps(file_bootstrap_map)
    emaps = e0_xy_correction(boot_map, norm_strat  = norm_strategy.max)
    E0    = kdst.S2e.values * emaps(kdst.X.values, kdst.Y.values)

    sel_krband, _, _, _, _ = selection_in_band(kdst.Z,
                                               E0,
                                               range_z = (10, maxZ),
                                               range_e = (10.0e+3,14e+3),
                                               nbins_z = 50,
                                               nbins_e = 50,
                                               nsigma  = 3.5)

    kdst = kdst[sel_krband].reset_index(drop=True)
    return kdst

#find global lt correction inside small radius R<70 random
def e0_lt_fit(z, lt, e0):
    return e0*np.exp(-z/lt)

def get_global_lt (kdst, Rcen=70):
    kdst_cen = kdst[kdst.R<=Rcen]
    popt, pcov = curve_fit(e0_lt_fit, kdst_cen.Z, kdst_cen.S2e, p0=[12000, 10000], bounds=((4000, 0), (20000, 100000)))
    lt, e0cen = popt
    lt_err,e0cen_err = np.sqrt(np.diag(pcov))


    #add lt and lt_err in dst
    kdst = kdst.assign(lt = lt, lt_err = lt_err, e0cen = e0cen, e0cen_err = e0cen_err)
    cols = [f'E_{x}' for x in range(12)]
    kdst = kdst[['event', 'time', 'S2e', 'S2q', 'S2t', 'DT', 'Z', 'X', 'Y', 'R', 'lt', 'lt_err', 'e0cen', 'e0cen_err']+cols]
    return kdst

#finally divide runs in chunks and save each chunk as differen file
def divide_run(kdst, folder_out):
    xbins = np.linspace(-200, 200, 21)
    ybins = np.linspace(-200, 200, 21)
    xcenters = (xbins[1:]+xbins[:-1])/2
    ycenters = (ybins[1:]+ybins[:-1])/2
    binstable = pd.DataFrame({'x':xcenters, 'y':ycenters})
    kdst = kdst.assign(xbin = pd.cut(kdst.X, xbins, labels = xcenters), ybin = pd.cut(kdst.Y, ybins, labels = ycenters))

    fout = fname.split('/')[-1].replace('kdst', 'kdst_chunked')
    fout = folder_out+fout


    groups = kdst.groupby(['xbin', 'ybin'])
    with tb.open_file(fout, 'w') as h5out:
        df_writer (h5out,binstable, 'BINS', 'BinInfo')
        for ind, group in groups:
            x, y = ind
            groupname = 'KDST'
            xindx = np.digitize(x, xbins)-1
            yindx = np.digitize(y, ybins)-1
            tablename = f'b_{xindx}_{yindx}'
            df_writer(h5out, group, groupname, tablename)



#load and select kr data
fnames = glob('/home/mmkekic/energy_resolution_study/kr_data/dorotheas_per_pmt/*.h5')
folder_out = '/home/mmkekic/energy_resolution_study/kr_data/chunked_dorotheas/'

for fname in fnames:
    print(fname)
    kdst = load_and_sel_data(fname)
    kdst = get_global_lt (kdst, Rcen=70)
    divide_run(kdst, folder_out)
