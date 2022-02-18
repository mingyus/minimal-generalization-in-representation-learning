This repository contains data and code that accompany the paper titled "Minimal cross-trial generalization in learning the representation of an odor-guided choice task".

## Table of Contents

* `data/takahashi2016roesch2009burton2018Valid.csv`: behavioral data aggregated from three studies: Roesch et al. (2009), Takahashi et al. (2016), and Burton et al. (2018). Only valid trials (i.e., trials in which animals made a choice response, and received the reward delivery if reward was available) are included. See [Data](#data) for details.
* `model_code_stan/`: model codes in stan
* `model_fits/`: model fits (posterior samples of model parameters)
* `model_simulation/`: model simulation results (due to space limit, we only include the aggregated results used for plotting average reward and learning curves)
* `*.ipynb`: analysis notebooks
* `*.py`: helper functions

## Data

Each row in the data file corresponds to a trial. The columns correspond to (some contain redundent information):

* `dataset`: "roesch2009", "takahashi2016", or "burton2018"
* `rat`: rat index (1-22)
* `session`: session index for current animal
* `sessionType`: "leftBetterFirst" or "rightBetterFirst" (denotes which side has the better reward in the first block)
* `trial`: trial index in a session (note that animals did not make valid responses in some trials; those invalid trials are excluded from analyses in this paper)
* `block`: block index (1-4)
* `blockType`: "short\_long", "long\_short", "big\_small", or "small\_big" (denotes the reward type in left and right wells respectively)
* `odor`: "left", "right" or "free"
* `choice`: 1 = left, 2 = right
* `rewardAmount`: 0, 1 or 2
* `rewardDelay`: in seconds
* `trialType`: all are "valid"
* `trialCond`: reward type (short, long, big, small) + reward side (left, right)
* `trialCondCode`: code for trialCond (1 = big\_left, 2 = big\_right, 3 = small\_left, 4 = small\_right, 5 = short\_left, 6 = short\_right, 7 = long\_left, 8 = long\_right)
* Reaction times (RTs): all in seconds
	* `odorEntryRT`: time between light on and odor port entry
	* `odorExitRT`: time between odor delivery and odor port exit
	* `wellEntryRT`: time between odor port exit and reward well entry
	* `wellExitRT`: time between the last reward delivery and reward well exit

## Models and model fitting

Models (including the hierarchical logistic regression model and reinforcement learning models) are implemented in [PyStan](https://pystan.readthedocs.io/)[1], available in `model_code_stan/`.

Model fitting and evaluation functions are implemented in `funcs_model_fit_evaluate.py`. To fit the hierarchical logistic regression model, use `behavior_logistic_analysis.ipynb`. To fit the reinforcement learning models, use `model_fitting.ipynb`.

Model fitting results (posterior samples of model parameters) are saved in `model_fits/`.

## Analyses and figures

The following notebooks reproduce the analyses and figures in the paper:

* Fig 1C: `behavior_learning_curves.ipynb`
* Fig 1D: `behavior_logistic_analysis.ipynb`
* Fig 3A, Fig S1: `model_comparison.ipynb`
* Fig 3B-E, Fig4C: `model_parameter_analyses.ipynb`
* Fig 4A,B: `model_simulation.ipynb`
* Fig S2: `split_half_analysis.ipynb`
* Fig S3: `model_parameter_posterior.ipynb`

[1] To execute notebooks in this repository, a working Python3 environment with PyStan installed is required.

**Reference**

Roesch, M. R., Singh, T., Brown, P. L., Mullins, S. E., & Schoenbaum, G. (2009). Ventral striatal neurons encode the value of the chosen action in rats deciding between differently delayed or sized rewards. *Journal of Neuroscience*, 29(42), 13365-13376.

Takahashi, Y. K., Langdon, A. J., Niv, Y., & Schoenbaum, G. (2016). Temporal specificity of reward prediction errors signaled by putative dopamine neurons in rat VTA depends on ventral striatum. *Neuron*, 91(1), 182-193.

Burton, A. C., Bissonette, G. B., Vazquez, D., Blume, E. M., Donnelly, M., Heatley, K. C., ... & Roesch, M. R. (2018). Previous cocaine self-administration disrupts reward expectancy encoding in ventral striatum. *Neuropsychopharmacology*, 43(12), 2350-2360.