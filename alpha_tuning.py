import numpy as np
import pandas as pd
from vib import VIB
from load_gutenberg import Gutenberg

gutenberg = Gutenberg()

alphas = [0.1, 0.5, 1.0, 2.0, 5.0]
num_trials = 5
epochs = 5
alpha_acc = {x:[] for x in alphas}

for trial in range(num_trials):
    train_msg_pad, test_msg_pad, train_labels, test_labels = gutenberg.get_data(normalize=False)
    data = {'train_data': train_msg_pad, 'test_data': test_msg_pad, 
            'train_labels': train_labels, 'test_labels': test_labels}
    
    for alpha in alphas:
        print(f"Training with alpha = {alpha}, trial = {trial + 1}")
        vib = VIB(encoder_args={'num_layers':2, 'num_units':[128,64]})
        res = vib.train(data, epochs=epochs, beta=10**-4, alpha=alpha)
        alpha_acc[alpha].append(res['Test avg_acc'].iloc[-1])

res = pd.DataFrame({'alpha':alphas, 'avg_acc': [np.mean(alpha_acc[x]) for x in alphas], 
                    'std_acc': [np.std(alpha_acc[x]) for x in alphas]})
res.to_csv('alpha_tuning_results.csv', index=False)
print("Alpha tuning completed. Results saved to 'alpha_tuning_results.csv'.")