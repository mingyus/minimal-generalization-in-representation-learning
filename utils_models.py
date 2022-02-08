import numpy as np
from funcs_model_predictions import *

modelInfo = {
    'sixState_full': {
        'modelName': 'sixState_full',
        'parNames': ['beta','eta','gamma','sb','pers','lapse'],
        'Npars': 6,
        'parBounds': [[0,10], [0,1], [0,1], [-2,2], [-2,2], [0,1]]
    },
    'fourState_full': {
        'modelName': 'fourState_full',
        'parNames': ['beta','eta','gamma','sb','pers','lapse'],
        'Npars': 6,
        'parBounds': [[0,10], [0,1], [0,1], [-2,2], [-2,2], [0,1]]
    },
    'hybridValue_full': {
        'modelName': 'hybridValue_full',
        'parNames': ['beta','w4','eta','gamma','sb','pers','lapse'],
        'Npars': 7,
        'parBounds': [[0,10], [0,1], [0,1], [0,1], [-2,2], [-2,2], [0,1]]
    },
    'hybridLearning_full': {
        'modelName': 'hybridLearning_full',
        'parNames': ['beta','eta','eta4state','gamma','sb','pers','lapse'],
        'Npars': 7,
        'parBounds': [[0,10], [0,1], [0,1], [0,1], [-2,2], [-2,2], [0,1]]
    },
}

modelPredictions = {
    'sixState_full': sixState_full_predict,
    'fourState_full': fourState_full_predict,
    'hybridValue_full': hybridValue_full_predict,
    'hybridLearning_full': hybridLearning_full_predict,
}