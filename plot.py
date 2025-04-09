import matplotlib.pyplot as plt
import pandas as pd 

var = 'epoch'
var_true = 'epoch' # 'α' # 'β' # 

results = pd.read_csv(f'results/{var}_tuning_results.csv')
results.set_index(var)[['train accuracy','test accuracy']].mul(100).plot(
    linestyle='--', marker='o')
# plt.xscale('log')
plt.title(f'Accuracy vs. {var_true} plot')
plt.xlabel(var_true)
plt.ylabel('Accuracy (%)')
plt.savefig(f'/Users/thomastesselaar/Downloads/{var}.png', dpi=1200)
fig = plt.gcf()
fig.set_size_inches(6.4, 3.2)
plt.tight_layout()
plt.savefig(f'/Users/thomastesselaar/Downloads/{var}_wide.png', dpi=1200)