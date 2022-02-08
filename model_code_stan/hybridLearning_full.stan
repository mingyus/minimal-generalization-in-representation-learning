data {

        // Metadata
        int Ns;               // number of subjects
        int Nt;               // total number of trials

        // Data
        int startSubject[Nt]; // subject number at start of each subject sequence
        int startSession[Nt]; // session number at start of each session

        int odor[Nt];         // 1-3: left, right, free
        int choice[Nt];       // 1: left, 2: right
        int reward[Nt];       // reward in number of pellets
        real delay[Nt];       // reward delay in units of seconds
        int trialType[Nt];    // 1: valid, -1: early exit, 0: all other invalid trials (with no choice)

}   // data

parameters {

        // Group level parameters - eta, beta, and gamma are drawn from this...
        vector[7] mu_pr;
        vector<lower=0>[7] sigma;

        // Subject-level parameters (raw; these are unit normal for equal density sampling across each par)
        vector[Ns] beta_samp;
        vector[Ns] eta_samp;
        vector[Ns] eta4state_samp;
        vector[Ns] gamma_samp;
        vector[Ns] sb_samp;
        vector[Ns] pers_samp;
        vector[Ns] lapse_samp;

}   // parameters

transformed parameters {

        // Subject-level parameters (transformed)
        vector<lower=0,upper=10>[Ns] beta;
        vector<lower=0,upper=1>[Ns]  eta;
        vector<lower=0,upper=1>[Ns]  eta4state;
        vector<lower=0,upper=1>[Ns]  gamma;
        vector<lower=-2,upper=2>[Ns] sb;
        vector<lower=-2,upper=2>[Ns] pers;
        vector<lower=0,upper=1>[Ns]  lapse;

        for (i in 1:Ns) {
                beta[i]  = Phi_approx( mu_pr[1] + sigma[1] * beta_samp[i] ) * 10;
                eta[i] = Phi_approx( mu_pr[2] + sigma[2] * eta_samp[i] );
                eta4state[i] = Phi_approx( mu_pr[3] + sigma[3] * eta4state_samp[i] );
                gamma[i] = Phi_approx( mu_pr[4] + sigma[4] * gamma_samp[i] );
                sb[i] = Phi_approx( mu_pr[5] + sigma[5] * sb_samp[i] ) * 4 - 2;
                pers[i] = Phi_approx( mu_pr[6] + sigma[6] * pers_samp[i] ) * 4 - 2;
                lapse[i] = Phi_approx( mu_pr[7] + sigma[7] * lapse_samp[i] );
        }

}   // transformed parameters

model {

        // Declarations of variables
        matrix[3,2] Q;
        real rewardPerceived;
        int currentSubject;
        int currentSession;
        int preChoice;
        vector[2] perseveration;

        // Group level priors
        mu_pr ~ normal(0, 1);
        sigma ~ cauchy(0, 5);      // with the lower bound of 0, this is a half-cauchy(0,5)

        // Subject-level priors - these are for the sampled parameters
        beta_samp ~ normal(0, 1);
        eta_samp ~ normal(0, 1);
        eta4state_samp ~ normal(0, 1);
        gamma_samp ~ normal(0, 1);
        sb_samp ~ normal(0, 1);
        pers_samp ~ normal(0, 1);
        lapse_samp ~ normal(0, 1);

        // Likelihood of all the data
        rewardPerceived = 0;
        currentSubject = 0;
        currentSession = 0;
        
        for (tr in 1:Nt) {

            if (startSubject[tr]>0) { // if this is the start of a new subject
                currentSubject += 1;
            }

            if (startSession[tr]>0) { // if this is the start of a new session
                currentSession += 1;
                // reset variables
                Q = [[0, 0], [0, 0], [0, 0]];
                preChoice = 0;
                perseveration = rep_vector(0, 2);
            }
            
            if (trialType[tr]!=0) {   // valid trials or early exit trials (trials with choices)
            
                if (preChoice > 0){  // not the first choice of a session
                    perseveration[preChoice] = pers[currentSubject];
                    perseveration[3-preChoice] = 0;
                }

                // likelihood of observed choice
                choice[tr] ~ categorical( (1-lapse[currentSubject]) * softmax( beta[currentSubject] * (Q[odor[tr]]' + sb[currentSubject]*[0,1]' + perseveration) ) + lapse[currentSubject]/2 );

                // calculate perceived reward
                rewardPerceived = reward[tr] * pow(gamma[currentSubject], delay[tr]);
                
                // update Q values
                Q[odor[tr], choice[tr]] += eta[currentSubject] * (rewardPerceived - Q[odor[tr], choice[tr]]);
                
                if (odor[tr]<3) {
                    // correct forced choices (rewarded)
                    if (choice[tr] == odor[tr]){   
                        Q[3, choice[tr]] += eta4state[currentSubject] * (rewardPerceived - Q[3, choice[tr]]);             // update Q value for free-choice 
                    }
                }
                else { // free choices (always rewarded)
                    // update Q value for forced-choice with this choice as the correct response
                    Q[choice[tr], choice[tr]] += eta4state[currentSubject] * (rewardPerceived - Q[choice[tr], choice[tr]]);
                }
                
                // update the previous choice
                preChoice = choice[tr];
                
            }
                
        } // trials

}     // model
