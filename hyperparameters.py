from StandardClassifier_1D import StandardClassifier_1D
from config import config

your_models = {
    'LR_task': {
        'antisaccade' : {
            'min' : {
                    'sLDA' : [StandardClassifier_1D, {'model_name':'sLDA', 'solver': 'lsqr', 'shrinkage': 'auto'}],
                    'LDA' : [StandardClassifier_1D, {'model_name':'LDA'}],
                    'MLP' : [StandardClassifier_1D, {'model_name':'MLP', 'random_state': 1, 'max_iter': 300}],
                    'RUSBoost' : [StandardClassifier_1D, {'model_name':'RUSBoost', 'n_estimators': 200, 'random_state': 1}],
                }
        }
    }
}


our_ML_models = {
    'LR_task' : {
        'antisaccade' : {
            'min' : {
                'KNN' : [StandardClassifier_1D, {'model_name':'KNN', 'leaf_size': 10, 'n_neighbors': 10, 'n_jobs' : -1}],
                'GaussianNB' : [StandardClassifier_1D, {'model_name':'GaussianNB', 'var_smoothing': 0.0004941713361323833}],
                'LinearSVC' : [StandardClassifier_1D, {'model_name':'LinearSVC', 'C': 0.01, 'tol' : 1e-5, 'max_iter' : 1200}],
                'RBF SVC' : [StandardClassifier_1D, {'model_name':'RBF SVC', 'C': 1, 'gamma': 0.01, 'tol' : 1e-5, 'max_iter' : 1200}],
                'DecisionTree' : [StandardClassifier_1D, {'model_name':'DecisionTree', 'max_depth': 5}],
                'RandomForest' : [StandardClassifier_1D, {'model_name':'RandomForest', 'max_depth': 10, 'n_estimators': 250, 'n_jobs' : -1}],
                'GradientBoost' : [StandardClassifier_1D, {'model_name':'GradientBoost', 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 50}],
                'AdaBoost' : [StandardClassifier_1D, {'model_name':'AdaBoost', 'learning_rate': 0.5, 'n_estimators': 100}],
                'XGBoost' : [StandardClassifier_1D, {'model_name':'XGBoost', 'objective' : 'binary:logistic', 'eval_metric' : 'logloss', 'eta': 0.1, 'max_depth': 5, 'n_estimators': 250, 'use_label_encoder' : False}]
            }
        }
    }
}

our_ML_dummy_models = {
    'LR_task' : {
        'antisaccade' : {
            'min' : {
                "Stratified" : [StandardClassifier_1D, {'model_name':'Stratified', 'strategy' : 'stratified'}],
                "MostFrequent" : [StandardClassifier_1D, {'model_name':'MostFrequent', 'strategy' : 'most_frequent'}],
                "Prior" : [StandardClassifier_1D, {'model_name':'Prior', 'strategy' : 'prior'}],
                "Uniform" : [StandardClassifier_1D, {'model_name': 'Uniform', 'strategy' : 'uniform'}]
            }
        }
    }
}


# merge two dict, new_dict overrides base_dict in case of incompatibility
def merge_models(base_dict, new_dict):
    result = dict()
    keys = base_dict.keys() | new_dict.keys()
    for k in keys:
        if k in base_dict and k in new_dict:
            if type(base_dict[k]) == dict and type(new_dict[k]) == dict:
                result[k] = merge_models(base_dict[k], new_dict[k])
            else:
                # overriding
                result[k] = new_dict[k]
        elif k in base_dict:
            result[k] = base_dict[k]
        else:
            result[k] = new_dict[k]
    return result

all_models = dict()

if config['include_ML_models']:
    all_models = merge_models(all_models, our_ML_models)
if config['include_dummy_models']:
    all_models = merge_models(all_models, our_ML_dummy_models)
if config['include_your_models']:
    all_models = merge_models(all_models, your_models)