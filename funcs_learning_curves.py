import numpy as np
import matplotlib.pyplot as plt
from utils_dataset import n_blocks


def plotLearningCurve(data, N=10, ifReturnCurveData=False, ifLegend=False):
    
    if ifReturnCurveData:
        curveData = dict.fromkeys((trialType, dataName) for trialType in ['forced','free'] for dataName in ['x','y','err'])
    else:
        # plot setting
        from IPython.display import set_matplotlib_formats
        set_matplotlib_formats('png', 'pdf')
        plt.rcParams.update({'font.family': 'arial'})
        lineWidth = 2
        fig, ax = plt.subplots(figsize=(10,3.5))
    
    # create x variable (trial index)
    trialIndices = []
    for i_block in range(n_blocks*2):
        if i_block == 0:
            start = 1
        else:
            start = trialIndices[-2] + (2 if i_block%2 else 1)
        trialIndices = np.concatenate((trialIndices, np.arange(start, start+N), [np.nan]))   
    
    # note that the correct choice for free trials is defined based on block type (which is defined based on detected block change point, can be later than real change point)
    data['correctChoice'] = 1*(data['odor']=='left') + 2*(data['odor']=='right') + (data['odor']=='free')*(
            1*((data['blockType']=='short_long')|(data['blockType']=='big_small')) +
            2*((data['blockType']=='long_short')|(data['blockType']=='small_big')) )
    data['correct'] = (data['correctChoice'] == data['choice'])
    
    for trialType in ['forced','free']:
        learningCurves = []
        for iRat, rat in enumerate(data['rat'].unique()):
            learningCurve = []
            for i_block in np.arange(n_blocks)+1:
                sessions = data.loc[data['rat']==rat,'session'].unique()
                NSessions = sessions.shape[0]
                firstN = np.empty((NSessions, N))
                lastN = np.empty((NSessions, N))
                for i_session in range(NSessions):
                    sessionData = data[(data['rat']==rat) & (data['block']==i_block) & (data['session']==sessions[i_session]) & (data['trialType']=='valid')]
                    thisFirst, thisLast = getFirstLastNTrials(N, sessionData, trialType)
                    firstN[i_session, :] = thisFirst
                    lastN[i_session, :] = thisLast
                if NSessions > 1:
                    learningCurve.append(np.nanmean(firstN, axis=0))
                    learningCurve.append(np.nanmean(lastN, axis=0))
                else:
                    learningCurve.append(firstN[0, :])
                    learningCurve.append(lastN[0, :])
            learningCurves.append(np.concatenate([np.concatenate((curve,[np.nan])) for curve in learningCurve]))
            
        NValidRat = np.sum(~np.isnan(np.stack(learningCurves)), axis=0)
        y = np.nanmean(learningCurves, axis=0)
        err = np.nanstd(learningCurves, axis=0)/np.sqrt(NValidRat)
        if ifReturnCurveData:
            curveData[trialType, 'x'] = trialIndices
            curveData[trialType, 'y'] = y
            curveData[trialType, 'err'] = err
        else:
            ax.plot(trialIndices, y, 'r' if trialType=='forced' else 'b', label='Forced' if trialType=='forced' else 'Free', linewidth=lineWidth)
            ax.fill_between(trialIndices, y-err, y+err, color='r' if trialType=='forced' else 'b', alpha=0.3)

    if ifReturnCurveData:
        return curveData
    else:
        # plot the block switch points and general figure settings
        for blockChange in np.array([N*2+1, N*4+2, N*6+3])+0.5:
            ax.axvline(x=blockChange, linestyle='--', color='gray', linewidth=lineWidth)
        ax.set_ylim([0, 1])
        ax.set_xlim([0, N*8+4])
        ax.set_xlabel('Trial')
        ax.set_ylabel('Accuracy')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticklabels('')
        ax.tick_params(axis='x', length=0)
        ax.set_xlabel('')
        ax.tick_params(axis='y', width=1.5, pad=5, direction='out')
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=20)
        ax.set_ylabel('Accuracy', fontsize=25)
        ax.yaxis.labelpad = 10
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        if ifLegend:
            ax.legend(loc='lower right', frameon=False)


def getFirstLastNTrials(N, sessionData, trialType, rewardType=None):    
    first = sessionData.head(N).reset_index(drop=True).copy()
    last = sessionData.tail(N).reset_index(drop=True).copy()
    if rewardType is None:
        first.loc[(((first['odor']=='left')|(first['odor']=='right')) if trialType=='free' else first['odor']=='free'), 'correct'] = np.nan
        last.loc[((last['odor']=='left')|(last['odor']=='right') if trialType=='free' else last['odor']=='free'), 'correct'] = np.nan
    else:
        first.loc[((first['odor']=='left')|(first['odor']=='right') if trialType=='free' else first['odor']=='free') & (first['rewardType']==1 if rewardType == 'highR' else first['rewardType']==2), 'correct'] = np.nan
        last.loc[((last['odor']=='left')|(last['odor']=='right') if trialType=='free' else last['odor']=='free') & (last['rewardType']==1 if rewardType == 'highR' else last['rewardType']==2), 'correct'] = np.nan
    return (filltoNtrials(N, first['correct'].values, -1), filltoNtrials(N, last['correct'].values, 0))


def filltoNtrials(N, tmp, loc):
    if loc == -1:
        if tmp.size<N:
            for i in range(N-tmp.size):
                tmp = np.append(tmp, np.nan)
    elif loc == 0:
        if tmp.size<N:
            for i in range(N-tmp.size):
                tmp = np.insert(tmp, 0, np.nan)
                np.append(tmp,[np.nan])
    return tmp