data {
    
    int<lower=1> P;                          // number of predictors
    int<lower=1> Ns;                         // number of subjects
    int<lower=0> Nt;                         // total number of trials/data points
    int<lower=0,upper=1> startSubject[Nt];   // start of a new subject
    int<lower=0,upper=1> y[Nt];
    matrix[Nt,P] X;

}   // data

parameters {

    // Group level parameters
    vector[P+1] mu_pr;
    vector<lower=0>[P+1] sigma;

    // Subject-level parameters
    matrix[P+1, Ns] beta_raw;                      // the first element is the intercept

}   // parameters

transformed parameters {
    matrix[P+1,Ns] beta;
    for (s in 1:Ns) {
            for (p in 1:P+1) {
                beta[p,s] = mu_pr[p] + sigma[p] * beta_raw[p,s];
            }
    }
}

model {

        // Declarations of variables
        int subj;

        // Group level priors
        mu_pr ~ normal(0, 5);
        sigma ~ normal(0, 5);  // Angela: gamma(1, 0.5)
            
        // Subject-level priors
        for (s in 1:Ns) {
            for (p in 1:P+1) {
                beta_raw[p,s] ~ std_normal();
            }
        }
        
        // Likelihood of all the data
        subj = 0;
        for (tr in 1:Nt) {
            if (startSubject[tr] == 1) {
                subj += 1;
            }
            y[tr] ~ bernoulli_logit(X[tr,:] * beta[2:P+1,subj] + beta[1,subj]);
        }

}     // model
