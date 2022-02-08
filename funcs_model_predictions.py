import numpy as np


# For all models: samples is a dataframe with samples of parameters; dd is a data dictionary
def sixState_full_predict(samples, dd):
    # extract each parameter (beta, eta, gamma: #samples * #subjects)
    beta = np.array(samples.loc[:, ['beta['+str(iSub+1)+']' for iSub in np.arange(dd['Ns'])]])
    eta = np.array(samples.loc[:, ['eta['+str(iSub+1)+']' for iSub in np.arange(dd['Ns'])]])
    gamma =  np.array(samples.loc[:, ['gamma['+str(iSub+1)+']' for iSub in np.arange(dd['Ns'])]])
    sb =  np.array(samples.loc[:, ['sb['+str(iSub+1)+']' for iSub in np.arange(dd['Ns'])]])
    pers =  np.array(samples.loc[:, ['pers['+str(iSub+1)+']' for iSub in np.arange(dd['Ns'])]])
    lapse =  np.array(samples.loc[:, ['lapse['+str(iSub+1)+']' for iSub in np.arange(dd['Ns'])]])
    
    NSamples = samples.shape[0]
    
    likelihood = np.empty((NSamples, dd['Nt'])) * np.nan
    currentSubject = -1;
    currentSession = -1;
    # Likelihood of all the data
    for tr in np.arange(dd['Nt']):
        if dd['startSubject'][tr]>0: # if this is the start of a new subject
            currentSubject += 1;
        if dd['startSession'][tr]>0: # if this is the start of a new session
            currentSession += 1;
            # reset Q values and the perseveration term
            Q = np.zeros((NSamples, 3, 2));
            preChoice = 0;
            perseveration = np.zeros((NSamples, 2));
        if dd['trialType'][tr] != 0: # valid trials or early exit trials (trials with choices)
            if preChoice > 0:  # not the first choice of a session
                perseveration[:, preChoice-1] = pers[:, currentSubject];
                perseveration[:, 2-preChoice] = 0;
            # likelihood of observed choice
            DVLeft = Q[:, dd['odor'][tr]-1, 0] + perseveration[:, 0]
            DVRight = Q[:, dd['odor'][tr]-1, 1] + sb[:, currentSubject] + perseveration[:, 1]
            pLeft = (1 - lapse[:, currentSubject]) * 1 / (1 + np.exp(- beta[:, currentSubject] * (DVLeft - DVRight))) + lapse[:, currentSubject]/2
            likelihood[:, tr] = pLeft if dd['choice'][tr] == 1 else (1-pLeft)
            # calculate reward prediction error
            delta = dd['reward'][tr] * np.power(gamma[:, currentSubject], dd['delay'][tr]) - Q[:, dd['odor'][tr]-1, dd['choice'][tr]-1];
            # update Q values
            Q[:, dd['odor'][tr]-1, dd['choice'][tr]-1] += eta[:, currentSubject] * delta;
            # update the previous choice
            preChoice = dd['choice'][tr];
    
    return likelihood


def fourState_full_predict(samples, dd):
    # extract each parameter (beta, eta, gamma: #samples * #subjects)
    beta = np.array(samples.loc[:, ['beta['+str(iSub+1)+']' for iSub in np.arange(dd['Ns'])]])
    eta = np.array(samples.loc[:, ['eta['+str(iSub+1)+']' for iSub in np.arange(dd['Ns'])]])
    gamma =  np.array(samples.loc[:, ['gamma['+str(iSub+1)+']' for iSub in np.arange(dd['Ns'])]])
    sb =  np.array(samples.loc[:, ['sb['+str(iSub+1)+']' for iSub in np.arange(dd['Ns'])]])
    pers =  np.array(samples.loc[:, ['pers['+str(iSub+1)+']' for iSub in np.arange(dd['Ns'])]])
    lapse =  np.array(samples.loc[:, ['lapse['+str(iSub+1)+']' for iSub in np.arange(dd['Ns'])]])
    
    NSamples = samples.shape[0]
    
    odorchoice2state = np.array([[1,4],[3,2],[1,2]])
    
    likelihood = np.empty((NSamples, dd['Nt'])) * np.nan
    currentSubject = -1;
    currentSession = -1;
    # Likelihood of all the data
    for tr in np.arange(dd['Nt']):
        if dd['startSubject'][tr]>0: # if this is the start of a new subject
            currentSubject += 1;
        if dd['startSession'][tr]>0: # if this is the start of a new session
            currentSession += 1;
            # reset Q values and the perseveration term
            Q = np.zeros((NSamples, 4));
            preChoice = 0;
            perseveration = np.zeros((NSamples, 2));
        if dd['trialType'][tr] != 0: # valid trials or early exit trials (trials with choices)
            if preChoice > 0:  # not the first choice of a session
                perseveration[:, preChoice-1] = pers[:, currentSubject];
                perseveration[:, 2-preChoice] = 0;
            # likelihood of observed choice
            DVLeft = Q[:, odorchoice2state[dd['odor'][tr]-1,0] -1] + perseveration[:, 0]
            DVRight = Q[:, odorchoice2state[dd['odor'][tr]-1,1] -1] + sb[:, currentSubject] + perseveration[:, 1]
            pLeft = (1 - lapse[:, currentSubject]) * 1 / (1 + np.exp(- beta[:, currentSubject] * (DVLeft - DVRight))) + lapse[:, currentSubject]/2
            likelihood[:, tr] = pLeft if dd['choice'][tr] == 1 else (1-pLeft)
            # calculate reward prediction error
            delta = dd['reward'][tr] * np.power(gamma[:, currentSubject], dd['delay'][tr]) - Q[:, odorchoice2state[dd['odor'][tr]-1, dd['choice'][tr]-1] - 1 ];
            # update Q values
            Q[:, odorchoice2state[dd['odor'][tr]-1, dd['choice'][tr]-1] - 1 ] += eta[:, currentSubject] * delta;
            # update the previous choice
            preChoice = dd['choice'][tr];
    
    return likelihood


def hybridValue_full_predict(samples, dd):
    # extract each parameter (beta, eta, gamma: #samples * #subjects)
    beta = np.array(samples.loc[:, ['beta['+str(iSub+1)+']' for iSub in np.arange(dd['Ns'])]])
    w4 = np.array(samples.loc[:, ['w4['+str(iSub+1)+']' for iSub in np.arange(dd['Ns'])]])
    eta = np.array(samples.loc[:, ['eta['+str(iSub+1)+']' for iSub in np.arange(dd['Ns'])]])
    gamma =  np.array(samples.loc[:, ['gamma['+str(iSub+1)+']' for iSub in np.arange(dd['Ns'])]])
    sb =  np.array(samples.loc[:, ['sb['+str(iSub+1)+']' for iSub in np.arange(dd['Ns'])]])
    pers =  np.array(samples.loc[:, ['pers['+str(iSub+1)+']' for iSub in np.arange(dd['Ns'])]])
    lapse =  np.array(samples.loc[:, ['lapse['+str(iSub+1)+']' for iSub in np.arange(dd['Ns'])]])
    
    NSamples = samples.shape[0]
    
    odorchoice2state = np.array([[1,4],[3,2],[1,2]])
    
    likelihood = np.empty((NSamples, dd['Nt'])) * np.nan
    currentSubject = -1;
    currentSession = -1;
    # Likelihood of all the data
    for tr in np.arange(dd['Nt']):
        if dd['startSubject'][tr]>0: # if this is the start of a new subject
            currentSubject += 1;
        if dd['startSession'][tr]>0: # if this is the start of a new session
            currentSession += 1;
            # reset Q values and the perseveration term
            Q4 = np.zeros((NSamples, 4));
            Q6 = np.zeros((NSamples, 3, 2));
            preChoice = 0;
            perseveration = np.zeros((NSamples, 2));
        if dd['trialType'][tr] != 0: # valid trials or early exit trials (trials with choices)
            if preChoice > 0:  # not the first choice of a session
                perseveration[:, preChoice-1] = pers[:, currentSubject];
                perseveration[:, 2-preChoice] = 0;
            # likelihood of observed choice
            DVLeft = w4[:, currentSubject] * Q4[:, odorchoice2state[dd['odor'][tr]-1,0] -1] + (1 - w4[:, currentSubject]) * Q6[:, dd['odor'][tr]-1, 0] + perseveration[:, 0]
            DVRight = w4[:, currentSubject] * Q4[:, odorchoice2state[dd['odor'][tr]-1,1] -1] + (1 - w4[:, currentSubject]) * Q6[:, dd['odor'][tr]-1, 1] + sb[:, currentSubject] + perseveration[:, 1]
            pLeft = (1 - lapse[:, currentSubject]) * 1 / (1 + np.exp(- beta[:, currentSubject] * (DVLeft - DVRight))) + lapse[:, currentSubject]/2
            likelihood[:, tr] = pLeft if dd['choice'][tr] == 1 else (1-pLeft)
            # calculate reward prediction error
            delta4 = dd['reward'][tr] * np.power(gamma[:, currentSubject], dd['delay'][tr]) - Q4[:, odorchoice2state[dd['odor'][tr]-1, dd['choice'][tr]-1] - 1 ];
            delta6 = dd['reward'][tr] * np.power(gamma[:, currentSubject], dd['delay'][tr]) - Q6[:, dd['odor'][tr]-1, dd['choice'][tr]-1];
            # update Q values
            Q4[:, odorchoice2state[dd['odor'][tr]-1, dd['choice'][tr]-1] - 1 ] += eta[:, currentSubject] * delta4;
            Q6[:, dd['odor'][tr]-1, dd['choice'][tr]-1] += eta[:, currentSubject] * delta6;
            # update the previous choice
            preChoice = dd['choice'][tr];

    return likelihood


def hybridLearning_full_predict(samples, dd):
    # extract each parameter (beta, eta, gamma: #samples * #subjects)
    beta = np.array(samples.loc[:, ['beta['+str(iSub+1)+']' for iSub in np.arange(dd['Ns'])]])
    eta = np.array(samples.loc[:, ['eta['+str(iSub+1)+']' for iSub in np.arange(dd['Ns'])]])
    eta4state = np.array(samples.loc[:, ['eta4state['+str(iSub+1)+']' for iSub in np.arange(dd['Ns'])]])
    gamma =  np.array(samples.loc[:, ['gamma['+str(iSub+1)+']' for iSub in np.arange(dd['Ns'])]])
    sb =  np.array(samples.loc[:, ['sb['+str(iSub+1)+']' for iSub in np.arange(dd['Ns'])]])
    pers =  np.array(samples.loc[:, ['pers['+str(iSub+1)+']' for iSub in np.arange(dd['Ns'])]])
    lapse =  np.array(samples.loc[:, ['lapse['+str(iSub+1)+']' for iSub in np.arange(dd['Ns'])]])
    
    NSamples = samples.shape[0]
    
    likelihood = np.empty((NSamples, dd['Nt'])) * np.nan
    currentSubject = -1;
    currentSession = -1;
    # Likelihood of all the data
    for tr in np.arange(dd['Nt']):
        if dd['startSubject'][tr]>0: # if this is the start of a new subject
            currentSubject += 1;
        if dd['startSession'][tr]>0: # if this is the start of a new session
            currentSession += 1;
            # reset Q values and the perseveration term
            Q = np.zeros((NSamples, 3, 2));
            preChoice = 0;
            perseveration = np.zeros((NSamples, 2));
        if dd['trialType'][tr] != 0: # valid trials or early exit trials (trials with choices)
            if preChoice > 0:  # not the first choice of a session
                perseveration[:, preChoice-1] = pers[:, currentSubject];
                perseveration[:, 2-preChoice] = 0;
            # likelihood of observed choice
            DVLeft = Q[:, dd['odor'][tr]-1, 0] + perseveration[:, 0]
            DVRight = Q[:, dd['odor'][tr]-1, 1] + sb[:, currentSubject] + perseveration[:, 1]
            pLeft = (1 - lapse[:, currentSubject]) * 1 / (1 + np.exp(- beta[:, currentSubject] * (DVLeft - DVRight))) + lapse[:, currentSubject]/2
            likelihood[:, tr] = pLeft if dd['choice'][tr] == 1 else (1-pLeft)
            # calculate perceived reward
            rewardPerceived = dd['reward'][tr] * np.power(gamma[:, currentSubject], dd['delay'][tr]);
            # update Q values
            Q[:, dd['odor'][tr]-1, dd['choice'][tr]-1] += eta[:, currentSubject] * (rewardPerceived - Q[:, dd['odor'][tr]-1, dd['choice'][tr]-1]);
            if (dd['odor'][tr]<3): # correct forced choices (rewarded)
                if (dd['choice'][tr] == dd['odor'][tr]):
                    # update Q value for free-choice 
                    Q[:, 2, dd['choice'][tr]-1] += eta4state[:, currentSubject] * (rewardPerceived - Q[:, 2, dd['choice'][tr]-1]);
            else: # free choices (always rewarded)
                # update Q value for forced-choice with this choice as the correct response
                Q[:, dd['choice'][tr]-1, dd['choice'][tr]-1] += eta4state[:, currentSubject] * (rewardPerceived - Q[:, dd['choice'][tr]-1, dd['choice'][tr]-1]);
            # update the previous choice
            preChoice = dd['choice'][tr];
    return likelihood

