import numpy as np
import time
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from config import config
from hyperparameters import all_models
import os


# return boolean arrays with length corresponding to n_samples
# the split is done based on the number of IDs
def split(ids, train, val, test):
    assert (train + val + test == 1)
    IDs = np.unique(ids)
    num_ids = len(IDs)

    # priority given to the test/val sets
    test_split = math.ceil(test * num_ids)
    val_split = math.ceil(val * num_ids)
    train_split = num_ids - val_split - test_split

    train = np.isin(ids, IDs[:train_split])
    val = np.isin(ids, IDs[train_split:train_split+val_split])
    test = np.isin(ids, IDs[train_split+val_split:])

    return train, val, test


def try_models(trainX_PA, trainY_PA, trainX_LG, trainY_LG, ids_PA, ids_LG, 
               models, N = 4, scoring = None, scale = False, save_trail = '', save = False):

    train_PA, val_PA, test_PA = split(ids_PA, 0.75, 0.15, 0.10)
    train_LG, val_LG, test_LG = split(ids_LG, 0.75, 0.15, 0.10)
    
    X_train, y_train = trainX_LG[train_LG], trainY_LG[train_LG]
    X_val, y_val = trainX_LG[val_LG], trainY_LG[val_LG]


    X_test_LG, y_test_LG = trainX_LG[test_LG], trainY_LG[test_LG]
    X_test_PA, y_test_PA = trainX_PA[test_PA], trainY_PA[test_PA]


    all_runs = []
    statistics = []

    for name, model in models.items():

        model_runs = []
        PA_ratio = 0.2

        for i in range(N):
            # create the model with the corresponding parameters
            trainer = model[0](**model[1])
            start_time = time.time()

            # Taking care of saving and loading
            path = config['checkpoint_dir'] + 'run' + str(i + 1) + '/'
            if not os.path.exists(path):
                os.makedirs(path)

            if config['retrain']:
                trainer.fit(X_train, y_train, X_val, y_val)
            else:
                trainer.load(path)

            if config['save_models']:
                trainer.save(path)
                print(path)


            #Creating the new testing dataset
            cntPA = math.floor(PA_ratio * len(y_test_PA))
            X_test = np.concatenate((X_test_PA[:cntPA], X_test_LG[cntPA:]), axis = 0)
            y_test = np.concatenate((y_test_PA[:cntPA], y_test_LG[cntPA:]), axis = 0)
            PA_ratio += 0.2
            
            #Testing the trained models on this dataset
            score = scoring(y_test, trainer.predict(X_test))
            

            #Storing the results
            runtime = (time.time() - start_time)
            all_runs.append([name, score, runtime])
            model_runs.append([score, runtime])
            
        
        model_runs = np.array(model_runs)
        model_scores, model_runtimes = model_runs[:,0], model_runs[:,1]
        statistics.append([name, model_scores.mean(), model_scores.std(), model_runtimes.mean(), model_runtimes.std()])

    np.savetxt(config['model_dir'] + '/runs' + save_trail+'.csv', all_runs, fmt='%s', delimiter=',', header='Model,Score,Runtime', comments='')
    np.savetxt(config['model_dir'] + '/statistics' + save_trail+'.csv', statistics, fmt='%s', delimiter=',', header='Model,Mean_score,Std_score,Mean_runtime,Std_runtime', comments='')

def benchmark(trainX_PA, trainY_PA, trainX_LG, trainY_LG):
    np.savetxt(config['model_dir']+'/config.csv', [config['task'], config['dataset'], config['preprocessing']], fmt='%s')
    models = all_models[config['task']][config['dataset']][config['preprocessing']]

    # The first column are the Id-s, we take the second which are labels
    ids_PA = trainY_PA[:, 0]
    ids_LG = trainY_LG[:, 0]
    y_PA = trainY_PA[:, 1]
    y_LG = trainY_LG[:, 1]

    scoring = (lambda y, y_pred: accuracy_score(y, y_pred.ravel()))  # Subject to change to mean euclidean distance.
    try_models(trainX_PA = trainX_PA, trainY_PA = y_PA, trainX_LG = trainX_LG, trainY_LG = y_LG, 
               ids_PA = ids_PA, ids_LG = ids_LG, models=models, scoring = scoring)
        