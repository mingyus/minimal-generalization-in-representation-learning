import numpy as np
import pandas as pd
import os
import sys
import utils_stan
import matplotlib.pyplot as plt
import seaborn as sns
from utils_models import *
from utils_stan import *


def data2dict(data):
    Ns = data['rat'].unique().size # N subjects
    Nt = data.shape[0] # total number of trials
    NSession = np.array([data.loc[data['rat']==rat,'session'].unique().size for rat in data['rat'].unique()])
    NSessionTotal = np.sum(NSession)
    startSubject = np.concatenate(([1],data['rat'][1:].values!=data['rat'][:-1].values))
    startSession = np.concatenate(([1],data['session'][1:].values!=data['session'][:-1].values))
    block_index = data['block'].fillna(0).values.astype(int)
    odor = np.zeros(data.shape[0])
    odor[data['odor']=='left'] = 1
    odor[data['odor']=='right'] = 2
    odor[data['odor']=='free'] = 3
    odor = odor.astype(int)
    choice = data['choice'].fillna(0).values.astype(int)
    reward = data['rewardAmount'].fillna(0).values.astype(int)
    delay = data['rewardDelay'].fillna(0).values
    trialType = np.zeros(data.shape[0])
    trialType[data['trialType']=='valid'] = 1
    trialType[data['trialType']=='shortStay'] = -1
    trialType = trialType.astype(int)
    sessionType = np.zeros(data.shape[0])
    sessionType[data['sessionType']=='leftBetterFirst'] = 1
    sessionType[data['sessionType']=='rightBetterFirst'] = 2
    sessionType = sessionType.astype(int)
    return dict(Ns=Ns, Nt=Nt, NSession=NSession, NSessionTotal=NSessionTotal, odor=odor, choice=choice, reward=reward, delay=delay, trialType=trialType, sessionType=sessionType, startSubject=startSubject, startSession=startSession, block_index=block_index)


def fitModel(modelName, datasetName, dd=None, samplingInfo=None, moreControl=None):
    # compile the model using stan_utility: it compiles the first time, saves the cached model and reuses it if there's no change
    stanFile = 'model_code_stan/' + modelName + '.stan'
    model = compile_model(stanFile)
    # fit model
    if moreControl is None:
        fit = model.sampling(data=dd, iter=samplingInfo['samples'], warmup=samplingInfo['warmup'], n_jobs=samplingInfo['n_jobs'], chains=samplingInfo['chains'], seed=0, init='random')
    else:
        fit = model.sampling(data=dd, iter=samplingInfo['samples'], warmup=samplingInfo['warmup'], n_jobs=samplingInfo['n_jobs'], chains=samplingInfo['chains'], seed=0, init='random', control=dict(max_treedepth=moreControl['max_treedepth'], adapt_delta=moreControl['adapt_delta']))
    # save the fit to csv
    allSamples = fit.to_dataframe(permuted=False, inc_warmup=True)
    allSamples.to_csv('model_fits/'+datasetName+'_'+modelName+'_allSamples.csv')
    return fit


def checkModelFitDiagnostics(modelInfo, samplingInfo, datasetName, fit):
    # sampler information
    print('date: ', fit.date, '\nsampler_params: ', fit.get_sampler_params(), '\nseed: ', fit.get_seed(), '\ninits: ', fit.get_inits(), '\nsamplingInfo: ', samplingInfo)
    # samples
    print(fit)
    # diagnostics
    check_all_diagnostics(fit)


def allSamples2posterior(allSamples, modelInfo, samplingInfo):
    from collections import OrderedDict
    newposterior = OrderedDict()
    Ns = allSamples.columns.str.startswith(modelInfo['parNames'][0]+'[').sum()
    Nsamples = (samplingInfo['samples']-samplingInfo['warmup'])*samplingInfo['chains']
    for key in ['mu_pr','sigma']:
        newposterior[key] = np.empty((Nsamples, modelInfo['Npars']))
        for iPar in np.arange(modelInfo['Npars']):
            newposterior[key][:,iPar] = allSamples.loc[allSamples['warmup']==0, key+'['+str(iPar+1)+']'].values
    for key in modelInfo['parNames']:
        newposterior[key+'_samp'] = np.empty((Nsamples, Ns))
        for iSub in np.arange(Ns):
            newposterior[key+'_samp'][:,iSub] = allSamples.loc[allSamples['warmup']==0, key+'_samp['+str(iSub+1)+']'].values
    for key in modelInfo['parNames']:
        newposterior[key] = np.empty((Nsamples, Ns))
        for iSub in np.arange(Ns):
            newposterior[key][:,iSub] = allSamples.loc[allSamples['warmup']==0, key+'['+str(iSub+1)+']'].values
    newposterior['lp__'] = allSamples.loc[allSamples['warmup']==0, 'lp__'].values
    return newposterior


def tracePlot(modelInfo, samplingInfo, allSamples, inc_warmup=False):
    Ns = allSamples.columns.str.startswith(modelInfo['parNames'][0]+'[').sum()
    fig, axes = plt.subplots(Ns*2+2, modelInfo['Npars'], figsize=(20,20), sharex=True)
    sampleIdx = np.arange(samplingInfo['samples']) if inc_warmup else np.arange(samplingInfo['warmup'],samplingInfo['samples'])
    for iChain in allSamples['chain'].unique(): #np.arange(samplingInfo['chains'])+1:
        dataThis = allSamples.loc[(allSamples['chain']==iChain) & (True if inc_warmup else allSamples['warmup']==0)].copy()
        for iPar in np.arange(modelInfo['Npars']):
            # group mean
            axes[0,iPar].plot(sampleIdx,dataThis['mu_pr[' + str(iPar+1) +']'])
            axes[0,iPar].set_ylabel('mu_pr[' + str(iPar+1) +']')
            # group std
            axes[1,iPar].plot(sampleIdx,dataThis['sigma[' + str(iPar+1) +']'])
            axes[1,iPar].set_ylabel('sigma[' + str(iPar+1) +']')
            # individual parameters before transformation
            for iSub in np.arange(Ns):
                axes[2+iSub,iPar].plot(sampleIdx,dataThis[modelInfo['parNames'][iPar] + '_samp[' + str(iSub+1) +']'])    
                axes[2+iSub,iPar].set_ylabel(modelInfo['parNames'][iPar] + '_samp[' + str(iSub+1) +']')
                if iSub == Ns-1:
                    axes[iSub+2,iPar].set_xlabel('Sample index')
            # individual parameters after transformation
            for iSub in np.arange(Ns):
                axes[2+Ns+iSub,iPar].plot(sampleIdx,dataThis[modelInfo['parNames'][iPar] + '[' + str(iSub+1) +']'])    
                axes[2+Ns+iSub,iPar].set_ylabel(modelInfo['parNames'][iPar] + '[' + str(iSub+1) +']')
                if iSub == Ns-1:
                    axes[iSub+2,iPar].set_xlabel('Sample index')


def parameterSamplesbySubject(modelInfo, samplingInfo, posterior):
    span = 0.2
    Nsamples = (samplingInfo['samples']-samplingInfo['warmup'])*samplingInfo['chains']
    Ns = posterior[modelInfo['parNames'][0]].shape[1]
    # before transformation
    fig, axes = plt.subplots(modelInfo['Npars'], 1, figsize=(12, modelInfo['Npars']*1.5))
    for iPar in np.arange(modelInfo['Npars']):
        for s in range(Ns):
            axes[iPar].scatter(s+1+np.random.uniform(-span,span,Nsamples),posterior[modelInfo['parNames'][iPar]+'_samp'][:,s])
        axes[iPar].set(ylabel=modelInfo['parNames'][iPar]+'_samp',xticks=[])
    # after transformation
    fig, axes = plt.subplots(modelInfo['Npars'], 1, figsize=(12, modelInfo['Npars']*1.5))
    for iPar in np.arange(modelInfo['Npars']):
        for s in range(Ns):
            axes[iPar].scatter(s+1+np.random.uniform(-span,span,Nsamples),posterior[modelInfo['parNames'][iPar]][:,s])
        axes[iPar].set(ylabel=modelInfo['parNames'][iPar],xticks=[])
    sns.despine()

    
def parameterSamplesHistogram(modelInfo, samplingInfo, posterior):
    Ns = posterior[modelInfo['parNames'][0]].shape[1]
    # check the group mean and std
    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    fig.suptitle('Group-level parameters', fontsize=18)
    for iPar in np.arange(modelInfo['Npars']):
        sns.distplot(posterior['mu_pr'][:,iPar],label=modelInfo['parNames'][iPar], ax=axes[0]);
        sns.distplot(posterior['sigma'][:,iPar],label=modelInfo['parNames'][iPar], ax=axes[1]);
        axes[0].set_ylabel('mean (mu_pr)')
        axes[1].set_ylabel('std (sigma)')
        plt.legend()
    # check the histogram of individual parameters before transformation
    fig, axes = plt.subplots(modelInfo['Npars'], 1, figsize=(12, modelInfo['Npars']*4))
    axes[0].set_title('Individual parameters (before transformation)')
    for iPar in np.arange(modelInfo['Npars']):
        axes[iPar].set_ylabel(modelInfo['parNames'][iPar]+'_samp')
        for iSub in np.arange(Ns):
            sns.distplot(posterior[modelInfo['parNames'][iPar]+'_samp'][:,iSub], ax=axes[iPar])
    # check the histogram of individual parameters after transformation
    fig, axes = plt.subplots(modelInfo['Npars'], 1, figsize=(12, modelInfo['Npars']*4))
    axes[0].set_title('Individual parameters (after transformation)')
    for iPar in np.arange(modelInfo['Npars']):
        axes[iPar].set_ylabel(modelInfo['parNames'][iPar])
        for iSub in np.arange(Ns):
            sns.distplot(posterior[modelInfo['parNames'][iPar]][:,iSub], ax=axes[iPar])
    
    
def scatterPlotParameters(modelInfo, posterior):
    Ns = posterior[modelInfo['parNames'][0]].shape[1]
    # before transformation
    fig, axes = plt.subplots(modelInfo['Npars']-1, modelInfo['Npars']-1, figsize=(modelInfo['Npars']*4,modelInfo['Npars']*4))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for iPar1 in np.arange(modelInfo['Npars']):
        for iPar2 in np.arange(iPar1+1, modelInfo['Npars']):
            if modelInfo['Npars'] > 2:
                ax = axes[iPar1,iPar2-1]
            else:
                ax = axes
            for iSub in np.arange(Ns):
                ax.scatter(posterior[modelInfo['parNames'][iPar1]+'_samp'][:,iSub],posterior[modelInfo['parNames'][iPar2]+'_samp'][:,iSub], c=colors[iSub%len(colors)], alpha=0.5)
            ax.set(xlabel=modelInfo['parNames'][iPar1]+'_samp', ylabel=modelInfo['parNames'][iPar2]+'_samp')
    # after transformation
    fig, axes = plt.subplots(modelInfo['Npars']-1, modelInfo['Npars']-1, figsize=(modelInfo['Npars']*4,modelInfo['Npars']*4))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for iPar1 in np.arange(modelInfo['Npars']):
        for iPar2 in np.arange(iPar1+1, modelInfo['Npars']):
            if modelInfo['Npars'] > 2:
                ax = axes[iPar1,iPar2-1]
            else:
                ax = axes
            for iSub in np.arange(Ns):
                ax.scatter(posterior[modelInfo['parNames'][iPar1]][:,iSub],posterior[modelInfo['parNames'][iPar2]][:,iSub], c=colors[iSub%len(colors)], alpha=0.5)
            ax.set(xlabel=modelInfo['parNames'][iPar1], ylabel=modelInfo['parNames'][iPar2])
    sns.despine()
    
            
def transformationPlots(modelInfo, samplingInfo, posterior):
    import scipy as sp
    
    def test_phi_approx(x): # Phi_approx(x) = logit^{-1}(0.07056 x^3 + 1.5976 x)
        return sp.special.expit(0.07056*(x**3) + 1.5976*x)
    
    def test_transformation(x, bounds):
        lb = bounds[0]
        ub = bounds[1]
        if (np.isinf(lb) & np.isinf(ub)):
            return x
        else:
            return test_phi_approx(x)*(ub-lb) + lb
        
    Ns = posterior[modelInfo['parNames'][0]].shape[1]

    ## plot the average parameter (for each subject) before and after transformation: i.e. post vs mu_pr + sigma * pre_samp
    # (mu_pr, sigma and pre_samp are averaged over samples)
    # equation: post = Phi_approx( mu_pr + sigma * pre_samp );
    fig, ax = plt.subplots()
    for iPar in np.arange(modelInfo['Npars']):
        x = np.zeros(Ns)
        for iSub in np.arange(Ns):
            x[iSub] = np.mean(posterior['mu_pr'][:,iPar]) + np.mean(posterior['sigma'][:,iPar]) * np.mean(posterior[modelInfo['parNames'][iPar]+'_samp'][:,iSub])
        if not (np.sum(np.isinf(modelInfo['parBounds'][iPar])) > 0): # parameters that are bounded on both sides that go through transformation
            ax.scatter(x,test_phi_approx(x), label=modelInfo['parNames'][iPar])
        else: # parameters that don't have both bounds (or unbounded at all)
            ax.scatter(x, [0]*Ns, label=modelInfo['parNames'][iPar]) # plot on the axis
    x_test = np.arange(ax.get_xlim()[0],ax.get_xlim()[1],0.01)
    ax.plot(x_test,test_phi_approx(x_test),'k');
    ax.legend()
    ax.set_title('Phi_approx transformation')

    ## plot group-level distribution (curve): N(mu_pr, sigma^2) put through Phi_approx function, where mu_pr and sigma are averages over samples
    # with individual samples (tiny points) and individual means (points) overlaid
    fig, axes = plt.subplots(1, modelInfo['Npars'], figsize=(18,4))
    fig.suptitle('Group-level distribution, with individual samples', fontsize=18)
    for iPar in np.arange(modelInfo['Npars']):
        axes[iPar].set_xlabel(modelInfo['parNames'][iPar])
        # parameters for group-level gaussian (before transformation)
        groupMean = np.mean(posterior['mu_pr'][:,iPar])
        groupStd = np.mean(posterior['sigma'][:,iPar])
        # get a reasonable range of x
        xmin = np.min((np.tile(posterior['mu_pr'][:,iPar],(Ns,1)).T+posterior[modelInfo['parNames'][iPar]+'_samp']*np.tile(posterior['sigma'][:,iPar],(Ns,1)).T).flatten())
        xmax = np.max((np.tile(posterior['mu_pr'][:,iPar],(Ns,1)).T+posterior[modelInfo['parNames'][iPar]+'_samp']*np.tile(posterior['sigma'][:,iPar],(Ns,1)).T).flatten())
        xvalues = np.arange(xmin,xmax,(xmax-xmin)/100)
        # plot the group-level distribution after transformation
        axes[iPar].plot(test_transformation(xvalues, modelInfo['parBounds'][iPar]), sp.stats.norm.pdf(xvalues,groupMean,groupStd))
        # plot individual parameters
        span = 0.3*np.max(sp.stats.norm.pdf(xvalues,groupMean,groupStd))
        Nsamples = (samplingInfo['samples']-samplingInfo['warmup'])*samplingInfo['chains']
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for iSub in np.arange(Ns):
            x_samp = posterior[modelInfo['parNames'][iPar]+'_samp'][:,iSub]
            axes[iPar].scatter(test_transformation(posterior['mu_pr'][:,iPar]+x_samp*posterior['sigma'][:,iPar], modelInfo['parBounds'][iPar]), np.random.uniform(-span*(iSub+2)/Ns,-span*(iSub+1)/Ns,Nsamples), s=0.1, c=colors[iSub%len(colors)])
            axes[iPar].scatter(test_transformation(np.mean(posterior['mu_pr'][:,iPar]+x_samp*posterior['sigma'][:,iPar]), modelInfo['parBounds'][iPar]), 0, c=colors[iSub%len(colors)])


def func_WAIC(likelihood, typeCorrection=2):
    # likelihood: #samples * #trials
    lppd = np.sum(np.log(np.mean(likelihood, axis=0)))
    if typeCorrection == 1:
        pWAIC = 2 * np.sum( np.log(np.mean(likelihood, axis=0)) - np.mean(np.log(likelihood), axis=0) )
    else:
        pWAIC = np.sum(np.var(np.log(likelihood), axis=0, ddof=1))
    return - 2 * lppd + 2 * pWAIC


def func_WAIC_comparison(likelihood1, likelihood2, typeCorrection=2):
    # likelihood1/2: #samples * #trials
    # take model 1 as baseline
    Ntrials = likelihood1.shape[1]
    
    lppd1_trial = np.log(np.mean(likelihood1, axis=0))
    lppd2_trial = np.log(np.mean(likelihood2, axis=0))
    if typeCorrection == 1:
        pWAIC1_trial = 2 * ( np.log(np.mean(likelihood1, axis=0)) - np.mean(np.log(likelihood1), axis=0) )
        pWAIC2_trial = 2 * ( np.log(np.mean(likelihood2, axis=0)) - np.mean(np.log(likelihood2), axis=0) )
    else:
        pWAIC1_trial = np.var(np.log(likelihood1), axis=0, ddof=1)
        pWAIC2_trial = np.var(np.log(likelihood2), axis=0, ddof=1)
    
    waic1_trial = - 2 * lppd1_trial + 2 * pWAIC1_trial
    waic2_trial = - 2 * lppd2_trial + 2 * pWAIC2_trial
    
    waic_diff = np.sum(waic2_trial - waic1_trial)
    waic_diff_se = np.sqrt(Ntrials * np.var(waic2_trial - waic1_trial))
    
    return waic_diff, waic_diff_se, waic_diff/Ntrials, waic_diff_se/Ntrials, 


def calculate_likelihood(datasetName, data, baselineModel, modelList):
    # import data
    dd = data2dict(data)

    # get the likelihood for the baseline model
    allSamples_baseline = pd.read_csv('model_fits/'+datasetName+'_'+baselineModel+'_allSamples.csv')
    parSamples_baseline = allSamples_baseline.loc[allSamples_baseline['warmup']==0, [col for col in allSamples_baseline if np.sum([col.startswith(parName+'[') for parName in modelInfo[baselineModel]['parNames']])>0]]
    likelihood_baseline = modelPredictions[baselineModel](parSamples_baseline, dd)

    # calculate waic difference
    waicDiff, waicDiff_se = [dict.fromkeys(modelList) for i in range(2)]
    waicDiff_rat_perTrial, waicDiff_rat_perTrial_se = [pd.DataFrame() for i in range(2)]

    for model in modelList:
        allSamples = pd.read_csv('model_fits/'+datasetName+'_'+model+'_allSamples.csv')
        parSamples = allSamples.loc[allSamples['warmup']==0, [col for col in allSamples if np.sum([col.startswith(parName+'[') for parName in modelInfo[model]['parNames']])>0]]
        likelihood = modelPredictions[model](parSamples, dd)

        # for the entire group
        waicDiff[model], waicDiff_se[model], _, _ = func_WAIC_comparison(likelihood_baseline, likelihood, typeCorrection=2)

        # for each individual animal
        for rat in data['rat'].unique():
            _, _, diff_perTrial, se_perTrial = func_WAIC_comparison(likelihood_baseline[:, data['rat'] == rat], likelihood[:, data['rat'] == rat])
            waicDiff_rat_perTrial.loc[rat, model] = diff_perTrial
            waicDiff_rat_perTrial_se.loc[rat, model] = se_perTrial

    metrics_group = pd.DataFrame(data=[waicDiff, waicDiff_se], index=['waicDiff','waicDiff_se'])
    waicDiff_rat_perTrial = waicDiff_rat_perTrial.reset_index().rename(columns={'index':'rat'})
    waicDiff_rat_perTrial_se = waicDiff_rat_perTrial_se.reset_index().rename(columns={'index':'rat'})

    return metrics_group, waicDiff_rat_perTrial, waicDiff_rat_perTrial_se