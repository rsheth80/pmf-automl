import numpy as np
import pandas as pd
import sklearn.decomposition
import sklearn.impute
import time
import torch
import kernels
import gplvm
from utils import transform_forward, transform_backward
import bo
import pickle

torch.set_default_tensor_type(torch.FloatTensor)

fn_data = 'all_normalized_accuracy_with_pipelineID.csv'
fn_train_ix = 'ids_train.csv'
fn_test_ix = 'ids_test.csv'
fn_data_feats = 'data_feats_featurized.csv'

def get_data():
    """
    returns the train/test splits of the dataset as N x D matrices and the
    train/test dataset features used for warm-starting bo as D x F matrices.
    N is the number of pipelines, D is the number of datasets (in train/test),
    and F is the number of dataset features.
    """

    df = pd.read_csv(fn_data)
    pipeline_ids = df['Unnamed: 0'].tolist()
    dataset_ids = df.columns.tolist()[1:]
    dataset_ids = [int(dataset_ids[i]) for i in range(len(dataset_ids))]
    Y = df.values[:,1:].astype(np.float64)

    ids_train = np.loadtxt(fn_train_ix).astype(int).tolist()
    ids_test = np.loadtxt(fn_test_ix).astype(int).tolist()

    ix_train = [dataset_ids.index(i) for i in ids_train]
    ix_test = [dataset_ids.index(i) for i in ids_test]

    Ytrain = Y[:, ix_train]
    Ytest = Y[:, ix_test]

    df = pd.read_csv(fn_data_feats)
    dataset_ids = df[df.columns[0]].tolist()

    ix_train = [dataset_ids.index(i) for i in ids_train]
    ix_test = [dataset_ids.index(i) for i in ids_test]

    Ftrain = df.values[ix_train, 1:]
    Ftest = df.values[ix_test, 1:]

    return Ytrain, Ytest, Ftrain, Ftest

def train(m, optimizer, f_callback=None, f_stop=None):

    it = 0
    while True:

        try:
            t = time.time()

            optimizer.zero_grad()
            nll = m()
            nll.backward()
            optimizer.step()

            it += 1
            t = time.time() - t

            if f_callback is not None:
                f_callback(m, nll, it, t)

            # f_stop should not be a substantial portion of total iteration time
            if f_stop is not None and f_stop(m, nll, it, t):
                break

        except KeyboardInterrupt:
            break

    return m

def bo_search(m, bo_n_init, bo_n_iters, Ytrain, Ftrain, ftest, ytest,
              do_print=False):
    """
    initializes BO with L1 warm-start (using dataset features). returns a
    numpy array of length bo_n_iters holding the best performance attained
    so far per iteration (including initialization).

    bo_n_iters includes initialization iterations, i.e., after warm-start, BO
    will run for bo_n_iters - bo_n_init iterations.
    """

    preds = bo.BO(m.dim, m.kernel, bo.ei,
                  variance=transform_forward(m.variance))
    ix_evaled = []
    ix_candidates = np.where(np.invert(np.isnan(ytest)))[0].tolist()
    ybest_list = []

    ix_init = bo.init_l1(Ytrain, Ftrain, ftest).tolist()
    for l in range(bo_n_init):
        ix = ix_init[l]
        if not np.isnan(ytest[ix]):
            preds.add(m.X[ix], ytest[ix])
            ix_evaled.append(ix)
            ix_candidates.remove(ix)
        yb = preds.ybest
        if yb is None:
            yb = np.nan
        ybest_list.append(yb)

        if do_print:
            print('Iter: %d, %g [%d], Best: %g' % (l, ytest[ix], ix, yb))

    for l in range(bo_n_init, bo_n_iters):
        ix = ix_candidates[preds.next(m.X[ix_candidates])]
        preds.add(m.X[ix], ytest[ix])
        ix_evaled.append(ix)
        ix_candidates.remove(ix)
        ybest_list.append(preds.ybest)

        if do_print:
            print('Iter: %d, %g [%d], Best: %g' \
                                    % (l, ytest[ix], ix, preds.ybest))

    return np.asarray(ybest_list)

def random_search(bo_n_iters, ytest, speed=1, do_print=False):
    """
    speed denotes how many random queries are performed per iteration.
    """

    ix_evaled = []
    ix_candidates = np.where(np.invert(np.isnan(ytest)))[0].tolist()
    ybest_list = []
    ybest = np.nan

    for l in range(bo_n_iters):
        for ll in range(speed):
            ix = ix_candidates[np.random.permutation(len(ix_candidates))[0]]
            if np.isnan(ybest):
                ybest = ytest[ix]
            else:
                if ytest[ix] > ybest:
                    ybest = ytest[ix]
            ix_evaled.append(ix)
            ix_candidates.remove(ix)
        ybest_list.append(ybest)

        if do_print:
            print('Iter: %d, %g [%d], Best: %g' % (l, ytest[ix], ix, ybest))

    return np.asarray(ybest_list)


if __name__=='__main__':

    # train and evaluation settings
    Q = 20
    batch_size = 50
    n_epochs = 300
    lr = 1e-7
    N_max = 1000
    bo_n_init = 5
    bo_n_iters = 200
    save_checkpoint = False
    fn_checkpoint = None
    checkpoint_period = 50

    # train
    Ytrain, Ytest, Ftrain, Ftest = get_data()
    maxiter = int(Ytrain.shape[1]/batch_size*n_epochs)

    
    def f_stop(m, v, it, t):

        if it >= maxiter-1:
            print('maxiter (%d) reached' % maxiter)
            return True

        return False

    varn_list = []
    logpr_list = []
    t_list = []
    

    def f_callback(m, v, it, t):
        varn_list.append(transform_forward(m.variance).item())
        logpr_list.append(m().item()/m.D)
        if it == 1:
            t_list.append(t)
        else:
            t_list.append(t_list[-1] + t)

        if save_checkpoint and not (it % checkpoint_period):
            torch.save(m.state_dict(), fn_checkpoint + '_it%d.pt' % it)

        print('it=%d, f=%g, varn=%g, t: %g'
              % (it, logpr_list[-1], transform_forward(m.variance), t_list[-1]))

    # create initial latent space with PCA, first imputing missing observations
    imp = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean')
    X = sklearn.decomposition.PCA(Q).fit_transform(
                                            imp.fit(Ytrain).transform(Ytrain))

    # define model
    kernel = kernels.Add(kernels.RBF(Q, lengthscale=None), kernels.White(Q))
    m = gplvm.GPLVM(Q, X, Ytrain, kernel, N_max=N_max, D_max=batch_size)
    if save_checkpoint:
        torch.save(m.state_dict(), fn_checkpoint + '_it%d.pt' % 0)

    # optimize
    print('training...')
    optimizer = torch.optim.SGD(m.parameters(), lr=lr)
    m = train(m, optimizer, f_callback=f_callback, f_stop=f_stop)
    
    if save_checkpoint:
        torch.save(m.state_dict(), fn_checkpoint + '_itFinal.pt')


    # evaluate model and random baselines
    print('evaluating...') 
    with torch.no_grad():
        Ytest = Ytest.astype(np.float32)

        regrets_automl = np.zeros((bo_n_iters, Ytest.shape[1]))
        regrets_random1x = np.zeros((bo_n_iters, Ytest.shape[1]))
        regrets_random2x = np.zeros((bo_n_iters, Ytest.shape[1]))
        regrets_random4x = np.zeros((bo_n_iters, Ytest.shape[1]))

        for d in np.arange(Ytest.shape[1]):
            print(d)
            ybest = np.nanmax(Ytest[:,d])
            regrets_random1x[:,d] = ybest - random_search(bo_n_iters,
                                                          Ytest[:,d], speed=1)
            regrets_random2x[:,d] = ybest - random_search(bo_n_iters,
                                                          Ytest[:,d], speed=2)
            regrets_random4x[:,d] = ybest - random_search(bo_n_iters,
                                                          Ytest[:,d], speed=4)
            regrets_automl[:,d] = ybest - bo_search(m, bo_n_init, bo_n_iters,
                                                    Ytrain, Ftrain, Ftest[d,:],
                                                    Ytest[:,d])

        results = {'pmf': regrets_automl,
                   'random1x': regrets_random1x,
                   'random2x': regrets_random2x,
                   'random4x': regrets_random4x,
                  }
        
        with open('results.pkl','wb') as f:
            pickle.dump(results,f)
