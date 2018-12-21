# pmf-automl
Data and Python code related to the paper "Probabilistic matrix factorization for automated machine learning", NeurIPS, 2018.

Rishit Sheth, 12/21/2018

## Data

**all_normalized_accuracy_with_pipelineID.zip** contains the performance observations from running 42K pipelines on 553 OpenML datasets. The task was classification and the performance metric was balanced accuracy. Unzip prior to running code.

**pipelines.json** contains the 42K pipeline instantiations.

**ids_train.csv** and **ids_test.csv** contain the OpenML dataset IDs for the training and test sets.

**data_feats_featurized.csv** contains dataset features.

## Code

Tested with Python 3.6.5 on Ubuntu 16.04.5. Requires numpy, pandas, sklearn, torch, and scipy.

**run.py** will load the data, run training, and then run evaluation (compared against random baselines).

**gplvm.py** contains the PyTorch model definition.

**kernels.py** contains a few GP kernel definitions.

**bo.py** contains the Bayesian optimization class definition, Expected Improvement acquisition function, and L1 warm-start initialization.

**utils.py** contains some useful auxiliary classes and functions.

**plotting.py** contains functions to visualize the results of the evaluation. Additionally requires matplotlib. To plot regret/rank curves after evaluation finishes:

    import plotting

    plotting.compare_regrets(results)
    plotting.compare_ranks(results)

## Notes

1. **Typo** in the paper and supplementary material: **553 total datasets were used in the experiments**, not 564.

	Our train/test split procedure described in Supplementary Section 1 allocated 453 datasets to train and 100 datasets to test. 11 of the datasets allocated to test were used to train auto-sklearn. Thus, for a fair comparison, these 11 datasets were placed into train resulting in 464 datasets to train and 89 datasets to test (and, as described in the supplementary material, 5 of these test datasets were not used in the evaluation due to failures by auto-sklearn and the FMLP implementation).

2. **Typo** in the supplementary material: During evaluation, auto-sklearn failed to complete 200 iterations on the datasets with **IDs 8, 197, 279, and 1472**, not 16, 20, 225, and 334. The FMLP implementation failed on the dataset with **ID 887**, not 40.

3. The training code was originally written in TensorFlow. We decided to port to PyTorch for the release since PyTorch is somewhat easier to work with. However, the PyTorch implementation is 2-3x slower.
