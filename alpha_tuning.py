import numpy as np
import pandas as pd
from vib import VIB
from load_gutenberg import Gutenberg

gutenberg = Gutenberg()

alphas = [10**-2, 10**-1.5, 0.1, 10**-0.5, 1.0, 2.0, 5.0, 8.0, 10.0]
num_trials = 50
epochs = 3
alpha_test_acc = {x:[] for x in alphas}
alpha_train_acc = {x:[] for x in alphas}
# print('HERE')
for trial in range(num_trials):
    train_msg_pad, test_msg_pad, train_labels, test_labels = gutenberg.get_data(normalize=False)
    data = {'train_data': train_msg_pad, 'test_data': test_msg_pad, 
            'train_labels': train_labels, 'test_labels': test_labels}
    
    for alpha in alphas:
        print(f"Training with alpha = {alpha}, trial = {trial + 1}")
        vib = VIB(encoder_args={'num_layers':2, 'num_units':[128,64]}, decoder_args={'out_dim':2})
        res = vib.train(data, epochs=epochs, beta=10**-4, alpha=alpha)
        alpha_test_acc[alpha].append(res['Test avg_acc'].iloc[-1])
        alpha_train_acc[alpha].append(res['Train avg_acc'].iloc[-1])

res = pd.DataFrame({'alpha':alphas, 
                    'train accuracy': [np.mean(alpha_train_acc[x]) for x in alphas], 
                    'train std': [np.std(alpha_train_acc[x]) for x in alphas], 
                    'test accuracy': [np.mean(alpha_test_acc[x]) for x in alphas], 
                    'test std': [np.std(alpha_test_acc[x]) for x in alphas]})
res.to_csv('results/alpha_tuning_results.csv', index=False)
print("Alpha tuning completed. Results saved to 'alpha_tuning_results.csv'.")