from scipy.special import expit

samples = 2000 # how many samples in total per chain?
warmup = 1500 # how many warmup samples? these do not contribute to posterior
chains = 4 # how many chains?
thin = 1 # period for saving samples
n_jobs = 4 # how many cores?

samplingInfo = dict(samples=samples, warmup=warmup, chains=chains, thin=thin, n_jobs=n_jobs)

def phi_approx(x): # Phi_approx(x) = logit^{-1}(0.07056 x^3 + 1.5976 x)
    return expit(0.07056*(x**3) + 1.5976*x)