import pickle
import plotting

# open results
with open('results.pkl','rb') as f:
    results=pickle.load(f)

# drawing plots
plotting.compare_regrets(results)
print("Plot 1 -----  Finished!")
plotting.compare_ranks(results)
print("plot2 ------- Finished!")