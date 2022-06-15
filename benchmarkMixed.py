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

def mix(ratio, X_PA, X_LG, y_PA, y_LG):
    cntPA = math.floor(ratio * len(y_PA))
    X_test = np.concatenate((X_PA[:cntPA], X_LG[cntPA:]), axis = 0)
    y_test = np.concatenate((y_PA[:cntPA], y_LG[cntPA:]), axis = 0)
    return X_test, y_test


def try_models(trainX_PA, trainY_PA, trainX_LG, trainY_LG, ids_PA, ids_LG, 
               models, N = 4, scoring = None, scale = False, save_trail = ''):

    train_PA, val_PA, test_PA = split(ids_PA, 0.75, 0.15, 0.10)
    train_LG, val_LG, test_LG = split(ids_LG, 0.75, 0.15, 0.10)
    
    #Full Training, testing, and cross-validation sets for LG and PA
    X_train_LG, y_train_LG = trainX_LG[train_LG], trainY_LG[train_LG]
    X_val_LG, y_val_LG = trainX_LG[val_LG], trainY_LG[val_LG]
    X_test_LG, y_test_LG = trainX_LG[test_LG], trainY_LG[test_LG]

    X_train_PA, y_train_PA = trainX_PA[train_PA], trainY_PA[train_PA]
    X_val_PA, y_val_PA = trainX_PA[val_PA], trainY_PA[val_PA]
    X_test_PA, y_test_PA = trainX_PA[test_PA], trainY_PA[test_PA]

    
    PA_train_ratio = config['PA_train_ratio']
    PA_test_ratio = config['PA_test_ratio']


    all_runs = []
    statistics = []

    for name, model in models.items():

        model_runs = []
        for i in range(N):
            # create the model with the corresponding parameters
            trainer = model[0](**model[1])
            start_time = time.time()

            # Taking care of saving and loading
            path = config['checkpoint_dir'] + 'run' + str(i + 1) + '/'
            if not os.path.exists(path):
                os.makedirs(path)

            #Defining the training set
            X_train, y_train = mix(PA_train_ratio, X_train_PA, X_train_LG,  y_train_PA, y_train_LG)
            X_val, y_val = mix(PA_train_ratio, X_val_PA, X_val_LG,  y_val_PA, y_val_LG)

            #Defining the testing dataset
            X_test, y_test = mix(PA_test_ratio, X_test_PA, X_test_LG, y_test_PA, y_test_LG)

            
            if config['retrain']:
                trainer.fit(X_train, y_train, X_val, y_val)
            else:
                trainer.load(path)

            if config['save_models']:
                trainer.save(path)
                print(path)
            
            #Testing the trained models on this dataset
            score = scoring(y_test, trainer.predict(X_test))
            

            #Storing the results
            runtime = (time.time() - start_time)
            all_runs.append([name, score, runtime])
            model_runs.append([score, runtime])
            
        
        model_runs = np.array(model_runs)
        model_scores, model_runtimes = model_runs[:,0], model_runs[:,1]
        statistics.append([name, PA_train_ratio, PA_test_ratio, model_scores.mean(), model_scores.std(), model_runtimes.mean(), model_runtimes.std()])

    np.savetxt(config['model_dir'] + '/runs-' + str(PA_train_ratio) + '-' + str(PA_test_ratio) + save_trail +'.csv', all_runs, fmt='%s', delimiter=',', header='Model,Score,Runtime', comments='')
    np.savetxt(config['model_dir'] + '/statistics-' + str(PA_train_ratio) + '-' + str(PA_test_ratio) + save_trail+'.csv', statistics, fmt='%s', delimiter=',', header='Model, PA_train_ratio, PA_test_ratio, Mean_score,Std_score,Mean_runtime,Std_runtime', comments='')

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
        
