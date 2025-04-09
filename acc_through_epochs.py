import numpy as np
import pandas as pd
from vib import VIB
from load_gutenberg import Gutenberg

gutenberg = Gutenberg()

num_trials = 50
epochs = 25
test_acc = {x:[] for x in range(epochs)}
train_acc = {x:[] for x in range(epochs)}
# print('HERE')
for trial in range(num_trials):
    train_msg_pad, test_msg_pad, train_labels, test_labels = gutenberg.get_data(normalize=False)
    data = {'train_data': train_msg_pad, 'test_data': test_msg_pad, 
            'train_labels': train_labels, 'test_labels': test_labels}
    
    print(f"Trial {trial + 1} of {num_trials}")
    vib = VIB(encoder_args={'num_layers':2, 'num_units':[128,64]}, decoder_args={'out_dim':2})
    res = vib.train(data, epochs=epochs, beta=10**-3, alpha=1.0)
    for i in range(epochs):
        test_acc[i].append(res['Test avg_acc'].iloc[i])
        train_acc[i].append(res['Train avg_acc'].iloc[i])

res = pd.DataFrame({'epoch':i, 
                    'train accuracy': [np.mean(train_acc[x]) for x in range(epochs)], 
                    'train std': [np.std(train_acc[x]) for x in range(epochs)], 
                    'test accuracy': [np.mean(test_acc[x]) for x in range(epochs)], 
                    'test std': [np.std(test_acc[x]) for x in range(epochs)]})
res.to_csv('results/epoch_tuning_results.csv', index=False)
print("Epoch tuning completed. Results saved to 'epoch_tuning_results.csv'.")