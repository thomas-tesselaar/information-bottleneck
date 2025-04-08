import numpy as np
import pandas as pd
from vib import VIB
from load_gutenberg import Gutenberg

gutenberg = Gutenberg()

# betas = [10**x for x in [-6,-5.5,-5,-4.5,-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2]]
betas = [10**x for x in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]]
num_trials = 25
epochs = 5
test_acc = {x:[] for x in betas}
train_acc = {x:[] for x in betas}
# print('HERE')
for trial in range(num_trials):
    train_msg_pad, test_msg_pad, train_labels, test_labels = gutenberg.get_data(normalize=False)
    data = {'train_data': train_msg_pad, 'test_data': test_msg_pad, 
            'train_labels': train_labels, 'test_labels': test_labels}
    
    for beta in betas:
        print(f"Training with beta = {beta}, trial = {trial + 1}")
        vib = VIB(encoder_args={'num_layers':2, 'num_units':[128,64]}, decoder_args={'out_dim':2})
        res = vib.train(data, epochs=epochs, beta=beta, alpha=1.0)
        test_acc[beta].append(res['Test avg_acc'].iloc[-1])
        train_acc[beta].append(res['Train avg_acc'].iloc[-1])

res = pd.DataFrame({'beta':betas, 
                    'train accuracy': [np.mean(train_acc[x]) for x in betas], 
                    'train std': [np.std(train_acc[x]) for x in betas], 
                    'test accuracy': [np.mean(test_acc[x]) for x in betas], 
                    'test std': [np.std(test_acc[x]) for x in betas]})
res.to_csv('beta_tuning_results2.csv', index=False)
print("Beta tuning completed. Results saved to 'beta_tuning_results.csv'.")