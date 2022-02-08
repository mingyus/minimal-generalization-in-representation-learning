import numpy as np

datasets = ['roesch2009', 'takahashi2016', 'burton2018']

dataset2marker = {
    'roesch2009': '+',
    'takahashi2016': '.',
    'burton2018': 'x'
}
ratio_s = {
    'roesch2009': 4/3,
    'takahashi2016': 5/3,
    'burton2018': 1
}
dataset_labels = {
    'roesch2009': 'Roesch et al. (2009)',
    'takahashi2016': 'Takahashi et al. (2016)',
    'burton2018': 'Burton et al. (2018)'
}

rat2dataset = dict()
for i in range(1,23):
    if 1 <= i <= 9:
        rat2dataset[i] = 'takahashi2016'
    elif 10 <= i <= 16:
        rat2dataset[i] = 'roesch2009'
    else:
        rat2dataset[i] = 'burton2018'

dataset2rat = {
    'roesch2009': [1,2,3,4,5,6,7,8,9],
    'takahashi2016': [10,11,12,13,14,15,16],
    'burton2018': [17,18,19,20,21,22]
}

ratOrder = np.array([10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9,17,18,19,20,21,22]) - 1

n_blocks = 4