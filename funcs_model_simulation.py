import numpy as np
import pandas as pd
from utils_sampling import phi_approx
from utils_models import *


def get_params(datasetName, modelName):
    allSamples = pd.read_csv('model_fits/' + datasetName + '_' + modelName + '_allSamples.csv')
    mu_pr = np.empty((modelInfo[modelName]['Npars']))
    for iPar, parName in enumerate(modelInfo[modelName]['parNames']):
        mu_pr[iPar] = allSamples.loc[allSamples['warmup']==0, 'mu_pr[' + str(iPar+1) + ']'].values.mean()
    pars = transform_params(modelName, mu_pr)
    return pars


def transform_params(modelName, mu_pr):
    pars = phi_approx(mu_pr);
    pars[0] = pars[0] * 10;  # beta
    if modelName in ['sixState_full', 'fourState_full']:
        pars[3] = pars[3] * 4 - 2;
        pars[4] = pars[4] * 4 - 2;        
    elif modelName in ['hybridValue_full', 'hybridLearning_full']:
        pars[4] = pars[4] * 4 - 2;
        pars[5] = pars[5] * 4 - 2;
    return pars


def model_simulate(model, pars, NSessions, NTrials, iRat=-1):
    # simulation info
    NBlocks = 4
    NOdors = NTrials
    
    # get parameter values
    if model in ['sixState_full', 'fourState_full']:
        beta, eta, gamma, sb, pers, lapse = pars[0], pars[1], pars[2], pars[3], pars[4], pars[5]
    elif model == 'hybridValue_full':
        beta, w4, eta, gamma, sb, pers, lapse = pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6]
    elif model == 'hybridLearning_full':
        beta, eta, eta4state, gamma, sb, pers, lapse = pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6]
    
    # initialization
    odorchoice2state = np.array([[1,4],[3,2],[1,2]])
    sessionList, sessionTypeList, trialList, blockList, blockTypeList, odorList, choiceList, rewardAmountList, rewardDelayList, trialTypeList, trialCondCodeList, trialCondList = [[] for _ in range(12)]
    
    for iSession in range(NSessions):
        # session info
        sessionType = np.random.choice(['leftBetterFirst', 'rightBetterFirst'])
        if sessionType == 'leftBetterFirst':
            blockTypeSeq = ['short_long', 'long_short', 'big_small', 'small_big']
        else:
            blockTypeSeq = ['long_short', 'short_long', 'small_big', 'big_small']
        
        # at the start of a new session, reset Q values and the perseveration term
        if model == 'sixState_full':
            Q = np.zeros((3, 2));
        elif model == 'fourState_full':
            Q = np.zeros(4);
        elif model == 'hybridValue_full':
            Q4 = np.zeros(4);
            Q6 = np.zeros((3, 2));
        elif model == 'hybridLearning_full':
            Q = np.zeros((3, 2));
        preChoice = 0;
        perseveration = np.zeros(2);
        
        for iBlock in range(NBlocks):
            blockType = blockTypeSeq[iBlock]
            longDelay = 0
            freeChoices = []
            
            # generate odor sequence
            odorSeq = []
            while len(odorSeq) < NOdors:
                newSeq = [1]*8 + [2]*8 + [3]*7
                np.random.shuffle(newSeq)
                newSeq = list(newSeq)
                if not (([1, 1, 1] in newSeq) or ([2, 2, 2] in newSeq) or ([3, 3, 3] in newSeq)):
                    odorSeq = odorSeq + newSeq
            
            iOdor = 0
            for iTrial in range(NTrials):
            
                # trial info
                odor = odorSeq[iOdor]

                # decision variable of left and right choices
                if preChoice > 0:  # not the first choice of a session
                    perseveration[preChoice-1] = pers;
                    perseveration[2-preChoice] = 0;
                if model == 'sixState_full':
                    DVLeft = Q[odor-1, 0] + perseveration[0]
                    DVRight = Q[odor-1, 1] + sb + perseveration[1]
                elif model == 'fourState_full':
                    DVLeft = Q[odorchoice2state[odor-1,0]-1] + perseveration[0]
                    DVRight = Q[odorchoice2state[odor-1,1]-1] + sb + perseveration[1]
                elif model == 'hybridValue_full':
                    DVLeft = w4 * Q4[odorchoice2state[odor-1,0]-1] + (1 - w4) * Q6[odor-1, 0] + perseveration[0]
                    DVRight = w4 * Q4[odorchoice2state[odor-1,1]-1] + (1 - w4) * Q6[odor-1, 1] + sb + perseveration[1]
                elif model == 'hybridLearning_full':
                    DVLeft = Q[odor-1, 0] + perseveration[0]
                    DVRight = Q[odor-1, 1] + sb + perseveration[1]

                # simulate choice
                pLeft = (1 - lapse) * 1 / (1 + np.exp(- beta * (DVLeft - DVRight))) + lapse/2
                choice = 2 - (np.random.random() < pLeft)
                if odor == 3:
                    freeChoices.append(choice)

                # simulate reward
                ifReward = (odor == choice) + (odor == 3)
                if ifReward:
                    # reward amount
                    if (blockType in ['short_long', 'long_short']) or (blockType == 'small_big' and choice == 1) or (blockType == 'big_small' and choice == 2):
                        reward = 1
                    else:
                        reward = 2
                    # reward delay
                    if (blockType == 'long_short' and choice == 1) or (blockType == 'short_long' and choice == 2): # if choosing the long reward side
                        longDelay = longDelay + 1
                        if longDelay > 7:
                            longDelay = 7
                        delay = longDelay
                    else:
                        delay = 0.5
                    # trial condition code
                    if choice == 1:
                        if blockType == 'big_small':
                            trialCondCode = 1
                        elif blockType == 'small_big':
                            trialCondCode = 3
                        elif blockType == 'short_long':
                            trialCondCode = 5
                        elif blockType == 'long_short':
                            trialCondCode = 7
                    else:
                        if blockType == 'big_small':
                            trialCondCode = 4
                        elif blockType == 'small_big':
                            trialCondCode = 2
                        elif blockType == 'short_long':
                            trialCondCode = 8
                        elif blockType == 'long_short':
                            trialCondCode = 6
                    # discounted reward
                    rewardPerceived = reward * np.power(gamma, delay)
                else:
                    reward = 0
                    delay = np.nan
                    trialCondCode = np.nan
                    rewardPerceived = 0
                
                # in timing blocks, reduce longDelay by 1s if chosen less than 8 times in the last 10 free choice trials, to a minimum of 3s
                if (blockType in ['short_long', 'long_short']) and (longDelay > 3):
                    choiceLong = 1 if blockType == 'long_short' else 2
                    if (len(freeChoices) >= 10) and (len(np.where(np.array(freeChoices[-10:]) == choiceLong)[0]) < 8):
                        longDelay = longDelay - 1

                # update Q values
                if model == 'sixState_full':
                    delta = rewardPerceived - Q[odor-1, choice-1]
                    Q[odor-1, choice-1] += eta * delta
                elif model == 'fourState_full':
                    delta = rewardPerceived - Q[odorchoice2state[odor-1, choice-1] - 1]
                    Q[odorchoice2state[odor-1, choice-1] - 1] += eta * delta
                elif model == 'hybridValue_full':
                    delta4 = rewardPerceived - Q4[odorchoice2state[odor-1, choice-1] - 1]
                    delta6 = rewardPerceived - Q6[odor-1, choice-1]
                    Q4[odorchoice2state[odor-1, choice-1] - 1] += eta * delta4
                    Q6[odor-1, choice-1] += eta * delta6;
                elif model == 'hybridLearning_full':
                    Q[odor-1, choice-1] += eta * (rewardPerceived - Q[odor-1, choice-1])
                    if odor < 3:
                        if choice == odor: # correct forced choices (rewarded)
                            Q[2, choice-1] += eta4state * (rewardPerceived - Q[2, choice-1])
                    else: # free choices (always rewarded)
                        Q[choice-1, choice-1] += eta4state * (rewardPerceived - Q[choice-1, choice-1])
                
                # update the previous choice
                preChoice = choice;
                
                # record data
                sessionList.append(iSession+1)
                sessionTypeList.append(sessionType)
                trialList.append(iTrial+1)
                blockList.append(iBlock+1)
                blockTypeList.append(blockType)
                odorList.append('left' if odor==1 else ('right' if odor==2 else 'free'))
                choiceList.append(choice)
                rewardAmountList.append(reward)
                rewardDelayList.append(delay)
                trialTypeList.append('valid')
                trialCondCodeList.append(trialCondCode)
                
                # proceed to the next trial
                if not ((odor < 3) and (choice != odor)): # if correct forced choice or any free choice
                    iOdor += 1
    
    trialCondMapping = ['big_left','big_right','small_left','small_right','short_left','short_right','long_left','long_right']
    trialCondList = [np.nan if np.isnan(trialCondCode) else trialCondMapping[trialCondCode - 1] for trialCondCode in trialCondCodeList]
    dataset = ['simulation'] * len(sessionList)
    rat = [iRat] * len(sessionList)
    data = pd.DataFrame(list(zip(dataset, rat, sessionList, sessionTypeList, trialList, blockList, blockTypeList, odorList, choiceList, rewardAmountList, rewardDelayList, trialTypeList, trialCondCodeList, trialCondList)), columns=['dataset', 'rat', 'session', 'sessionType', 'trial', 'block', 'blockType', 'odor', 'choice', 'rewardAmount', 'rewardDelay', 'trialType', 'trialCondCode', 'trialCond'])
    
    return data
