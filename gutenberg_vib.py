
import math

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from load_gutenberg import Gutenberg
from vib import VIB


OUT_DIM = 2

# load data
gutenberg = Gutenberg()
train_msg_pad, test_msg_pad, train_labels, test_labels = gutenberg.get_data()
data = {'train_data': train_msg_pad, 'test_data': test_msg_pad, 
        'train_labels': train_labels, 'test_labels': test_labels}

vib = VIB(encoder_args={'num_layers':3, 'num_units':[128,64,32]})

# train the model
num_samples = 50 # number of samples to average over
betas = [10**x for x in [-6,-5.5,-5,-4.5,-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2]]

beta_train_accuracy = []
beta_test_accuracy = []
try:
    for beta in betas:
        tmp_train_acc = []
        tmp_test_acc = []

        for i in range(num_samples):
            res = vib.train(data, epochs=10, batch_size=50, beta=beta, alpha=1.0)

            tmp_train_acc.append(res['Train acc'].iloc[-1])
            tmp_test_acc.append(['Test acc'].iloc[-1])

        beta_train_accuracy.append(np.mean(tmp_train_acc))
        beta_test_accuracy.append(np.mean(tmp_test_acc))

    pd.DataFrame(
        {'beta':betas,'train accuracy':beta_train_accuracy,
         'test accuracy':beta_test_accuracy}
    ).to_csv('beta_accuracy.csv', index=False)
    
except Exception as e:
    print(e)
    print(beta)
    print(betas)
    print(beta_train_accuracy)
    print(beta_test_accuracy)
