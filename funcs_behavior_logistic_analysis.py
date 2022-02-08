import numpy as np
import scipy.stats.kde as kde
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from utils_hpd import hpd_grid
from utils_dataset import *
from funcs_model_fit_evaluate import fitModel


def prep_data(dataValid, fitName, N_exclude=0):
    ratList, sessionList, iTrialList, NCorrectForcedBetweenList, accuracyList = [[] for _ in range(5)]
    for rat in dataValid['rat'].unique():
        for session in dataValid.loc[dataValid['rat']==rat,'session'].unique():
            for iBlock in np.arange(NBlocks):
                dataBlock = dataValid[(dataValid['rat']==rat)&(dataValid['session']==session)&(dataValid['block']==iBlock+1)].copy()[N_exclude:].reset_index(drop=True)
                lastWrong = -1
                for iTrial in np.arange(dataBlock.shape[0]):
                    if dataBlock.loc[iTrial,'odor'] == 'free':
                        if lastWrong >= 0:
                            ratList.append(rat)
                            sessionList.append(session)
                            iTrialList.append(iTrial+1)
                            NCorrectForcedBetweenList.append(np.sum((dataBlock.loc[(lastWrong+1):iTrial-1,'correct']==True)&(dataBlock.loc[(lastWrong+1):iTrial-1,'odor']!='free'))) # dataframe indexing includes the end value!
                            accuracyList.append(int(dataBlock.loc[iTrial,'correct']))
                        if dataBlock.loc[iTrial,'correct']:
                            lastWrong = -1
                        else:
                            lastWrong = iTrial

    startSubject = [int(ratList[i+1]!=ratList[i]) for i in range(len(ratList)-1)]
    startSubject = [1] + startSubject
    
    return ratList, startSubject, NCorrectForcedBetweenList, accuracyList


# plot individual 95% HDI and posterior of group mean
def plot_hdi(fitName, stanCodeName, iRegressor, sort=True, title=None, ylim=0.2, reorder=False):  # iRegressor: 0=intercept, 1..n=slopes
    fig = plt.figure(figsize=(12,4))
    gs = fig.add_gridspec(1, 6)
    axi = fig.add_subplot(gs[:-1])
    axg = fig.add_subplot(gs[-1])
    gs.update(wspace=0, hspace=0)

    allSamples = pd.read_csv('model_fits/' + fitName + '_' + stanCodeName + '_allSamples.csv')
    coeff = allSamples.loc[allSamples['warmup']==0, [col for col in allSamples if col.startswith('beta['+str(iRegressor+1))]].values
    if sort:
        coeff = coeff[:, np.argsort(np.mean(coeff, axis=0))] # sort
    if reorder:
        coeff = coeff[:, ratOrder]
    x = np.arange(coeff.shape[1])+1 # index of rat
    y = np.mean(coeff, axis=0)
    err_n = np.quantile(coeff, 0.025, axis=0)
    err_p = np.quantile(coeff, 0.975, axis=0)
    axi.axhline(0, color='gray', linestyle='--', linewidth=1.5)
    axi.errorbar(x=x, y=y, yerr=[y-err_n, err_p-y], ecolor='k', fmt='None', capsize=5, linewidth=2)
    axi.set_xlabel('Rat')
    axi.set_ylabel('Coefficient')
    axi.tick_params(axis='x', length=0)
    axi.tick_params(axis='y', width=1.5)
    axi.set_xticklabels('')
    sns.despine(ax=axi)
    axi.spines['left'].set_linewidth(1.5)
    axi.spines['bottom'].set_linewidth(0)
    axi.set_ylim([-ylim,ylim])
    axi.set_xlim([0,23])
    if title:
        axi.set_title(title)

    coeff_group = allSamples.loc[allSamples['warmup']==0, 'mu_pr['+str(iRegressor+1)+']'].values
    sns.distplot(coeff_group, kde=False, hist_kws={'alpha':0.6}, color='k', vertical=True, ax=axg)
    xmin, xmax = axg.get_xlim()
    axg.set_xticklabels('')
    axg.set_ylabel('')
    axg.set_yticklabels('')
    axg.tick_params(axis='x', length=0)
    axg.tick_params(axis='y', length=0)
    sns.despine(ax=axg)
    axg.spines['bottom'].set_linewidth(0)
    axg.spines['left'].set_linewidth(1.5)
    axg.spines['left'].set_position('zero')
    hdi = hpd_grid(coeff_group, alpha=0.05, roundto=3)[0][0]
    axg.hlines(y=hdi, xmin=xmin, xmax=xmax, color='k', linestyle='--', linewidth=1.5)
    axg.set_ylim([-ylim,ylim])

    print(fitName)
    print(hdi)