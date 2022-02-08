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
        vector[6] mu_pr;
        vector<lower=0>[6] sigma;

        // Subject-level parameters (raw; these are unit normal for equal density sampling across each par)
        vector[Ns] beta_samp;
        vector[Ns] eta_samp;
        vector[Ns] gamma_samp;
        vector[Ns] sb_samp;
        vector[Ns] pers_samp;
        vector[Ns] lapse_samp;

}   // parameters

transformed parameters {

        // Subject-level parameters (transformed)
        vector<lower=0,upper=10>[Ns] beta;
        vector<lower=0,upper=1>[Ns]  eta;
        vector<lower=0,upper=1>[Ns]  gamma;
        vector<lower=-2,upper=2>[Ns] sb;
        vector<lower=-2,upper=2>[Ns] pers;
        vector<lower=0,upper=1>[Ns]  lapse;

        for (i in 1:Ns) {
                beta[i]  = Phi_approx( mu_pr[1] + sigma[1] * beta_samp[i] ) * 10;
                eta[i] = Phi_approx( mu_pr[2] + sigma[2] * eta_samp[i] );
                gamma[i] = Phi_approx( mu_pr[3] + sigma[3] * gamma_samp[i] );
                sb[i] = Phi_approx( mu_pr[4] + sigma[4] * sb_samp[i] ) * 4 - 2;
                pers[i] = Phi_approx( mu_pr[5] + sigma[5] * pers_samp[i] ) * 4 - 2;
                lapse[i] = Phi_approx( mu_pr[6] + sigma[6] * lapse_samp[i] );
        }

}   // transformed parameters

model {

        // Declarations of variables
        vector[4] Q; // the four states are: [left,right,leftWrong,rightWrong]
        real delta;
        int currentSubject;
        int currentSession;
        int odorchoice2state[3, 2];
        int preChoice;
        vector[2] perseveration;

        // Group level priors
        mu_pr ~ normal(0, 1);
        sigma ~ cauchy(0, 5);      // with the lower bound of 0, this is a half-cauchy(0,5)

        // Subject-level priors - these are for the sampled parameters
        beta_samp ~ normal(0, 1);
        eta_samp ~ normal(0, 1);
        gamma_samp ~ normal(0, 1);
        sb_samp ~ normal(0, 1);
        pers_samp ~ normal(0, 1);
        lapse_samp ~ normal(0, 1);

        // the mapping from (odor,choice) to state index
        odorchoice2state[1,1] = 1;  // left-forced odor, choosing left
        odorchoice2state[1,2] = 4;  // left-forced odor, choosing right
        odorchoice2state[2,1] = 3;  // right-forced odor, choosing left
        odorchoice2state[2,2] = 2;  // right-forced odor, choosing right
        odorchoice2state[3,1] = 1;  // free odor, choosing left
        odorchoice2state[3,2] = 2;  // free odor, choosing right
        
        // Likelihood of all the data
        
        delta = 0;
        currentSubject = 0;
        currentSession = 0;
        
        for (tr in 1:Nt) {

            if (startSubject[tr]>0) { // if this is the start of a new subject
                currentSubject += 1;
            }

            if (startSession[tr]>0) { // if this is the start of a new session
                currentSession += 1;
                // reset variables
                Q = rep_vector(0, 4);
                preChoice = 0;
                perseveration = rep_vector(0, 2);
            }
            
            if (trialType[tr]!=0) {   // valid trials or early exit trials (trials with choices)
            
                if (preChoice > 0){  // not the first choice of a session
                    perseveration[preChoice] = pers[currentSubject];
                    perseveration[3-preChoice] = 0;
                }

                // likelihood of observed choice
                
                choice[tr] ~ categorical( (1-lapse[currentSubject]) * softmax( beta[currentSubject] * (Q[odorchoice2state[odor[tr]]] + sb[currentSubject]*[0,1]' + perseveration) ) + lapse[currentSubject]/2 );
                
                // calculate reward prediction error
                delta = reward[tr] * pow(gamma[currentSubject], delay[tr]) - Q[odorchoice2state[odor[tr], choice[tr]]];
                
                // update Q values
                Q[odorchoice2state[odor[tr], choice[tr]]] += eta[currentSubject] * delta;
                
                // update the previous choice
                preChoice = choice[tr];
                
            }
                
        } // trials

}     // model
