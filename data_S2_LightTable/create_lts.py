from invisible_cities.io.dst_io import load_dst, load_dsts, df_writer
from glob import glob
import numpy as np
import pandas as pd
import tables as tb
import re

import os
from scipy.stats import norm




#load and select kr data
folder_in = '/home/mmkekic/energy_resolution_study/kr_data/data_runs_dorotheas_per_pmt/'
folder_out = '/home/mmkekic/energy_resolution_study/kr_data/moving_runs_average_light_table/'
fnames = glob(folder_in+'*.h5')

#check tablenames in first file
with tb.open_file(fnames[0]) as tab:
    tablenames = tab.root.KDST.__members__


def mean_and_std(dfp):
    ecols = [f'E_{i}' for i in range(12)]
    mus = []
    stds = []
    for col in ecols:
        muf, stdf = norm.fit(dfp[col])
        mus.append(muf)
        stds.append(stdf)
    dct = {f'e_{i}':mus[i] for i in range(12)}
    dct.update({f'stde_{i}':stds[i] for i in range(12)})
    muf, stdf = norm.fit(dfp['S2e'])
    dct.update({'e_tot':muf, 'stde_tot':stdf})
    dct.update({'nevents':len(dfp)})
    return pd.Series(dct)


fnames_cp = [f for f in fnames]
lenfs = len(fnames_cp)
fnames_tmp = fnames_cp[:20]
for i in range(30, lenfs, 5):
    fnames_tp = fnames[i-30:i]
    runs = re.findall('\d+', fnames_tp[0])[0]
    rune = re.findall('\d+', fnames_tp[-1])[0]
    fout = folder_out+'runs_'+runs + '_'+ rune+'.h5'
    mp_chunk = []
    print(runs, rune)
    for tbname in tablenames:
        data = load_dsts(fnames_tp, 'KDST', tbname)
        xbin, ybin = data.xbin.unique()[0], data.ybin.unique()[0]
        xbins = np.linspace(xbin-10, xbin+10, 11)#2mm bins
        ybins = np.linspace(ybin-10, ybin+10, 11)
        xcenters = (xbins[1:]+xbins[:-1])/2
        ycenters = (ybins[1:]+ybins[:-1])/2
        data = data.assign(xbin = pd.cut(data.X, xbins, labels = xcenters), ybin = pd.cut(data.Y, ybins, labels = ycenters))
        #correct columns
        ecols = [f'E_{i}' for i in range(12)] + ['S2e']
        for col in ecols:
            data[col] = data[col]*np.exp(data.Z/data['lt'])/data['e0cen']

        #find mean and std of gaussians
        means = data.groupby(['xbin', 'ybin']).apply(mean_and_std).reset_index()
        mp_chunk.append(means)
    fullmap = pd.concat(mp_chunk, ignore_index=True)
    with tb.open_file(fout, 'w') as tab:
        df_writer(tab, fullmap, 'LT', 'LightTable')




fnames_tp = fnames
runs = re.findall('\d+', fnames_tp[0])[0]
rune = re.findall('\d+', fnames_tp[-1])[0]
fout = folder_out+'runs_'+runs + '_'+ rune+'.h5'
mp_chunk = []
for tbname in tablenames:
    data = load_dsts(fnames_tp, 'KDST', tbname)
    xbin, ybin = data.xbin.unique()[0], data.ybin.unique()[0]
    xbins = np.linspace(xbin-10, xbin+10, 21)#1mm bins
    ybins = np.linspace(ybin-10, ybin+10, 21)
    xcenters = (xbins[1:]+xbins[:-1])/2
    ycenters = (ybins[1:]+ybins[:-1])/2
    data = data.assign(xbin = pd.cut(data.X, xbins, labels = xcenters), ybin = pd.cut(data.Y, ybins, labels = ycenters))
    #correct columns
    ecols = [f'E_{i}' for i in range(12)] + ['S2e']
    for col in ecols:
        data[col] = data[col]*np.exp(data.Z/data['lt'])/data['e0cen']

    #find mean and std of gaussians
    means = data.groupby(['xbin', 'ybin']).apply(mean_and_std).reset_index()
    mp_chunk.append(means)
fullmap = pd.concat(mp_chunk, ignore_index=True)
with tb.open_file(fout, 'w') as tab:
    df_writer(tab, fullmap, 'LT', 'LightTable')
