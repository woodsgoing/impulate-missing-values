# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 09:12:12 2018

@author: june
"""

import pandas as pd
import numpy as np
import math
import os
import time
import dask.dataframe as dd
import fast_impute
import csv
from sklearn.model_selection import KFold
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

_DEBUG = True
global outputfilename
outputfilename=''
global outputfilepath
outputfilepath=''
global out_trial_file_name
out_trial_file_name=''
global time_line
time_line='.'

CLASS_NUM_BIN = 2

def setEnvInfo(filepath, filename):
    """
    Configure data file path and file name. 
    This must be used before other method, as it generate log info storage path.
    Parameters
    ----------
    filepath : string
        log file path
    filename : string
        log file name
    Output
    -------
    Generate path filepath/filename/ to storage result.    
    """
    global outputfilename
    global outputfilepath
    global out_trial_file_name
    outputfilename = filename
    outputfilepath = filepath
    out_trial_file_name = outputfilepath+'xxx_trials.csv'
    if not os.path.exists(outputfilepath):
        os.mkdir(outputfilepath) 
    global time_line
    time_line = time.strftime("%Y_%m_%d", time.localtime()) 

def _log(*arg, mode):
    global outputfilename
    global outputfilepath
    if outputfilename == '' or outputfilepath == '':
        return  
    with open(outputfilepath+outputfilename+mode+time_line+'.bayes.opt', "a+") as text_file:
        print(*arg, file=text_file)

def trace(*arg):
    _log(*arg, mode='trace')

def debug(*arg):
    if _DEBUG == True:
        _log(*arg, mode = 'debug')

############################ LGBM domain ###################################################
def int_module_lgbm_space(is_classifier=True):
    space = {
        'class_weight': hp.choice('class_weight', [None, 'balanced']),
        'boosting_type': hp.choice('boosting_type', 
                                   [{'boosting_type': 'gbdt', 
                                        'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, 
                                     {'boosting_type': 'dart', 
                                         'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                     {'boosting_type': 'goss'}]),
        'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.5)),
        'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
        'min_child_samples': hp.quniform('min_child_samples', 20, 1000, 5),
        'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
        'reg_lambda': hp.uniform('reg_lambda', 0.0, 10.0),
        'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
    }   
    return space

def int_module_lgbm_param(is_classifier=True):
    param = {
        'class_weight': 'balanced',
        'boosting_type': {'boosting_type': 'gbdt', 
                          'subsample': 0.6},
        'num_leaves': 60,
        'learning_rate': 0.05,
        'subsample_for_bin': 300000,
        'min_child_samples': 20,
        'reg_alpha': 0.2,
        'reg_lambda': 8,
        'colsample_bytree': 0.9
    }   
    return param


def int_module_lgbm_cv(params,dataframe, target, test_dataframe=None, n_folds = 5):
    """
    Internal lgbm model with cross validation, supporting both classification 
    prediction and regression prediction. It's aimed for easy usage and reuse 
    for all kinds of situation.
    
    Parameters
    ----------
    params : dictionary
        Parameter set with dictionary format for lgbm model.
    dataframe : pandas.Dataframe
        Dataframe to process.
    target : string
        Feature name, target identifies some column which is used for 
        prediction analyze.
    test_dataframe : pandas.Dataframe, optional
        Dataframe to predict.
    n_folds : integer, optional
        Cross validation times when run lgbm model with given dataframe. It's 
        often 5 or 10.
    Output
    -------
    Score when run lgbm model with given param and dataframe. For regression 
    prediction, score as R2; for binary classification, score as ROC; for 
    multi-classification, score as accuracy.
    Test result from prediction if specify test_dataframe.  
    """

    from lightgbm import LGBMClassifier
    from lightgbm import LGBMRegressor
    from sklearn.metrics import roc_auc_score
    PREDICT_NAME = 'predict'
    
    df = dataframe

    subsample = params['boosting_type'].get('subsample', 1.0)    
    # Extract the boosting type
    params['boosting_type'] = params['boosting_type']['boosting_type']
    params['subsample'] = subsample    
    # Make sure parameters that need to be integers are integers
    for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
        params[parameter_name] = int(params[parameter_name])     

    train_df = df.drop([target],axis=1)
    train_target = df[target]
    train_df, _ = one_hot_encoder(train_df, True)
    valid = df[[target]]
    valid[PREDICT_NAME] = float(0.0)

    predict_classifier_bin, predict_classifier_nominal = _check_classifier(df, target)
    predict_df = pd.DataFrame({'result':[]})
    folds = KFold(n_splits= n_folds, shuffle=True, random_state=1001)

    lgbm = None
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_target)):
        train_x, train_y = train_df.iloc[train_idx], train_target.iloc[train_idx]
        valid_x, valid_y = train_df.iloc[valid_idx], train_target.iloc[valid_idx]
       
        if predict_classifier_bin == True or predict_classifier_nominal == True:
            lgbm = LGBMClassifier(class_weight=params['class_weight'],\
                                  boosting_type=params['boosting_type'],\
                                  subsample=params['subsample'],\
                                  num_leaves=params['num_leaves'],\
                                  learning_rate=params['learning_rate'],\
                                  subsample_for_bin=params['subsample_for_bin'],\
                                  min_child_samples=params['min_child_samples'],\
                                  reg_alpha=params['reg_alpha'],\
                                  reg_lambda=params['reg_lambda'],\
                                  colsample_bytree=params['colsample_bytree'])

            lgbm.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], \
                                                 verbose= 1000, early_stopping_rounds= 200)
            if predict_classifier_bin:
                predict_result = lgbm.predict_proba(valid_x)[:, 1]
            else:
                predict_result = lgbm.predict(valid_x)
            predict_temp_df = pd.DataFrame({'result':predict_result},index=valid_x.index)
            predict_df = pd.concat([predict_df,predict_temp_df])
        else:
            lgbm = LGBMRegressor(class_weight=params['class_weight'],\
                                  boosting_type=params['boosting_type'],\
                                  subsample=params['subsample'],\
                                  num_leaves=params['num_leaves'],\
                                  learning_rate=params['learning_rate'],\
                                  subsample_for_bin=params['subsample_for_bin'],\
                                  min_child_samples=params['min_child_samples'],\
                                  reg_alpha=params['reg_alpha'],\
                                  reg_lambda=params['reg_lambda'],\
                                  colsample_bytree=params['colsample_bytree'])
            lgbm.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], \
                                                 verbose= 1000, early_stopping_rounds= 200)
            predict_result = lgbm.predict(valid_x)
            predict_temp_df = pd.DataFrame({'result':predict_result},index=valid_x.index)
            predict_df = pd.concat([predict_df,predict_temp_df])
        debug('++++++++++++++++++++LGBM+++++++++++++++++++++++++++++++++++++++++++')

    predict_df.sort_index(axis=0, inplace=True)
    valid[PREDICT_NAME] = predict_df['result']
    score = 0
    if predict_classifier_bin == True:
        score = roc_auc_score(valid[target],valid[PREDICT_NAME])
    elif predict_classifier_nominal:
        valid['compare'] = valid.apply(lambda x: x[target]==x[PREDICT_NAME], axis=1)
        score = np.sum(valid['compare'])/len(valid[target])
    else:
        score = np.square(np.corrcoef(valid[target],valid[PREDICT_NAME])[0,1])   


    trace('lgbm: '+target+', score: '+str(score)) 
    if test_dataframe is None:
        return score
    else:
        test_prediction = lgbm.predict(test_dataframe)
        return test_prediction, score


############################ LGBM domain end ###################################################
    
############################ Random Forest domain ###################################################

def int_module_random_forest_space(is_classifier=True):   
    space = {
        'n_estimators': hp.quniform('n_estimators', 60, 500, 20), 
        'criterion': hp.choice('criterion', ['gini', 'entropy']),
        'max_depth': hp.quniform('max_depth', 6, 15, 1),
#        'min_samples_split': hp.quniform('min_samples_split', 10, 100, 10),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 5, 100, 5),
#        'min_weight_fraction_leaf': hp.quniform('min_weight_fraction_leaf', 30, 150, 1),
        'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
#        'max_leaf_nodes': hp.quniform('max_leaf_nodes', 30, 150, 10),
#        'min_impurity_decrease': hp.quniform('min_impurity_decrease', 0.00001, 0.0001, 0.00001),
#        'min_impurity_split': hp.uniform('min_impurity_split', 0.0, 1.0), # deprecated by min_impurity_decrease
        'bootstrap': hp.choice('bootstrap', [True]),
        'oob_score': hp.choice('oob_score', [True]),
        'n_jobs': hp.choice('n_jobs', [-1]),
        'random_state': hp.choice('random_state', [2019]),
#        'verbose': hp.choice('verbose', [1]),
#        'warm_start': hp.choice('warm_start', [False]),
        'class_weight': hp.choice('class_weight', ['balanced', 'balanced_subsample', None])
        }
    if is_classifier== False:
        space['criterion'] = hp.choice('criterion', ['mse'])
    return space

def int_module_random_forest_param(is_classifier=True):   
    param = {
        'n_estimators': 200, 
        'criterion': 'gini',
        'max_depth': 12,
#        'min_samples_split': hp.quniform('min_samples_split', 10, 100, 10),
        'min_samples_leaf': 5,
#        'min_weight_fraction_leaf': hp.quniform('min_weight_fraction_leaf', 30, 150, 1),
        'max_features': 'sqrt',
#        'max_leaf_nodes': hp.quniform('max_leaf_nodes', 30, 150, 10),
#        'min_impurity_decrease': hp.quniform('min_impurity_decrease', 0.00001, 0.0001, 0.00001),
#        'min_impurity_split': hp.uniform('min_impurity_split', 0.0, 1.0), # deprecated by min_impurity_decrease
        'bootstrap': True,
        'oob_score': True,
        'n_jobs': -1,
        'random_state': 2019,
#        'verbose': hp.choice('verbose', [1]),
#        'warm_start': hp.choice('warm_start', [False]),
        'class_weight': 'balanced'
        }
    if is_classifier== False:
        param['criterion'] = 'mse'
    return param

def int_module_random_forest(params, dataframe, target, test_dataframe=None, \
                             n_folds=5):
    """
    Internal random forest model with cross validation, supporting both classification 
    prediction and regression prediction. It's aimed for easy usage and reuse 
    for all kinds of situation.
    
    Parameters
    ----------
    params : dictionary
        Parameter set with dictionary format for random forest model.
    dataframe : pandas.Dataframe
        Dataframe to process.
    target : string
        Feature name, target identifies some column which is used for 
        prediction analyze.
    test_dataframe : pandas.Dataframe, optional
        Dataframe to predict.
    n_folds : integer, optional
        Cross validation times when run random forest model with given dataframe. It's 
        often 5 or 10.
    Output
    -------
    Score when run random forest model with given param and dataframe. For regression 
    prediction, score as R2; for binary classification, score as ROC; for 
    multi-classification, score as accuracy.
    Test result from prediction if specify test_dataframe.  
    """
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import roc_auc_score
    PREDICT_NAME = 'predict'
    
    df = dataframe

    for parameter_name in ['n_estimators', 'max_depth', 'min_samples_leaf']:
        params[parameter_name] = int(params[parameter_name])
    for f_ in df.columns:
        df,_ = fast_impute.impute_mean(df, f_, target=target, intern=True)

    train_df = df.drop([target],axis=1)
    train_df, _ = one_hot_encoder(train_df, True)
    train_target = df[target]
    valid = df[[target]]
    valid[PREDICT_NAME] = 0
    

    predict_classifier_bin, predict_classifier_nominal = _check_classifier(df, target)
    predict_df = pd.DataFrame({'result':[]})
    folds = KFold(n_splits= n_folds, shuffle=True, random_state=1001)

    rf = None
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_target)):
        train_x, train_y = train_df.iloc[train_idx], train_target.iloc[train_idx]
        valid_x, valid_y = train_df.iloc[valid_idx], train_target.iloc[valid_idx]
       
        if predict_classifier_bin == True or predict_classifier_nominal == True:
            rf = RandomForestClassifier(class_weight=params['class_weight'],\
                                        n_estimators=params['n_estimators'],\
                                        criterion=params['criterion'],\
                                        max_depth=params['max_depth'],\
                                        min_samples_leaf=params['min_samples_leaf'],\
                                        max_features=params['max_features'],\
                                        bootstrap=params['bootstrap'],\
                                        oob_score=params['oob_score'],\
                                        n_jobs=params['n_jobs'],\
                                        random_state=params['random_state'])

            rf.fit(train_x, train_y)         
            if predict_classifier_bin:
                predict_result = rf.predict_proba(valid_x)[:, 1]
            else:
                predict_result = rf.predict(valid_x)
            predict_temp_df = pd.DataFrame({'result':predict_result},index=valid_x.index)
            predict_df = pd.concat([predict_df,predict_temp_df])

        else:
            rf = RandomForestRegressor(
                    
                                  n_estimators=params['n_estimators'],\
                                  criterion=params['criterion'],\
                                  max_depth=params['max_depth'],\
                                  min_samples_leaf=params['min_samples_leaf'],\
                                  max_features=params['max_features'],\
                                  bootstrap=params['bootstrap'],\
                                  oob_score=params['oob_score'],\
                                  n_jobs=params['n_jobs'],\
                                  random_state=params['random_state'])
            rf.fit(train_x, train_y)
            predict_result = rf.predict(valid_x)
            predict_temp_df = pd.DataFrame({'result':predict_result},index=valid_x.index)
            predict_df = pd.concat([predict_df,predict_temp_df])

        debug('++++++++++++++++++++random forest+++++++++++++++++++++++++++++++++++++++++++')


    predict_df.sort_index(axis=0, inplace=True)
    valid[PREDICT_NAME] = predict_df['result']
    score = 0
    if predict_classifier_bin == True:
        score = roc_auc_score(valid[target],valid[PREDICT_NAME])
    elif predict_classifier_nominal:
        valid['compare'] = valid.apply(lambda x: x[target]==x[PREDICT_NAME], axis=1)
        score = np.sum(valid['compare'])/len(valid[target])
    else:
        score = np.square(np.corrcoef(valid[target],valid[PREDICT_NAME])[0,1])
    
    trace('random forest: '+target+', score: '+str(score))
    if test_dataframe is None:
        return score
    else:
        test_prediction = rf.predict(test_dataframe)
        return test_prediction, score

############################ Random Forest domain end ###################################################
############################ LR domain ###################################################

def int_module_linear_regression_space(is_classifier=True):
    space = {
        'penalty': hp.choice('penalty',
                             [{'penalty': 'l1',
#                                   'solver':hp.choice('l1_solver',['saga'])},
                                   'solver':hp.choice('l1_solver',['liblinear','saga'])},
                              {'penalty': 'l2',
                               'solver':hp.choice('l2_solver',['liblinear', 'sag'])}
#                               'solver':hp.choice('l2_solver',['newton-cg','lbfgs', 'liblinear', 'sag'])}
                              ]),
#        'dual': hp.choise('dual',[False]),
        'tol': hp.loguniform('tol', np.log(0.00001), np.log(10)),
        'C': hp.loguniform('C', np.log(0.0001), np.log(10)),
#        'fit_intercept': hp.choise('fit_intercept',[True]),
        'class_weight': hp.choice('class_weight', [None, 'balanced']),
        'random_state': hp.choice('random_state', [2019]),
        'max_iter': hp.quniform('max_iter', 100, 1000,50),
#        'multi_class': hp.choice('multi_class',['auto']),
        'n_jobs': hp.choice('n_jobs',[-1]),
        }
    if is_classifier == False:
        space = {  
                'fit_intercept': hp.choice('fit_intercept',[True]),
                'normalize': hp.choice('normalize',[True, False]),
                'n_jobs': hp.choice('n_jobs',[-1]),
                }  

    return space

def int_module_linear_regression_param(is_classifier=True):
    param = {
        'penalty': {'penalty': 'l2', 'solver': 'sag'} 
#                               'solver':hp.choice('l2_solver',['newton-cg','lbfgs', 'liblinear', 'sag'])}
                              ,
#        'dual': hp.choise('dual',[False]),
        'tol': 0.001,
        'C': 0.1,
#        'fit_intercept': hp.choise('fit_intercept',[True]),
        'class_weight': None,
        'random_state': 2019,
        'max_iter': 1000,
#        'multi_class': hp.choice('multi_class',['auto']),
        'n_jobs': -1,
        }
    if is_classifier == False:
        param = {  
                'fit_intercept': True,
                'normalize': True,
                'n_jobs': -1,
                }  

    return param

def int_module_linear_regression(params, dataframe, target, \
                                 test_dataframe=None, n_folds=5):
    """
    Internal linear regression model with cross validation, supporting both classification 
    prediction and regression prediction. It's aimed for easy usage and reuse 
    for all kinds of situation.
    
    Parameters
    ----------
    params : dictionary
        Parameter set with dictionary format for linear regression model.
    dataframe : pandas.Dataframe
        Dataframe to process.
    target : string
        Feature name, target identifies some column which is used for 
        prediction analyze.
    test_dataframe : pandas.Dataframe, optional
        Dataframe to predict.
    n_folds : integer, optional
        Cross validation times when run linear regression model with given dataframe. It's 
        often 5 or 10.
    Output
    -------
    Score when run linear regression model with given param and dataframe. For regression 
    prediction, score as R2; for binary classification, score as ROC; for 
    multi-classification, score as accuracy.
    Test result from prediction if specify test_dataframe.  
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import MinMaxScaler
    PREDICT_NAME = 'predict'
    
    df = dataframe
#    df = df.reset_index(drop=True)
    for f_ in df.columns:
        df,_ = fast_impute.impute_mean(df, f_, target=target, intern=True)

    train_df = df.drop([target],axis=1)
    train_df, _ = one_hot_encoder(train_df, True)
    train_target = df[target]
    valid = df[[target]]
    valid[PREDICT_NAME] = 0

    min_max_scaler = MinMaxScaler()
    train_df=pd.DataFrame(min_max_scaler.fit_transform(train_df),index=train_df.index)

    predict_classifier_bin, predict_classifier_nominal = _check_classifier(df, target)
    predict_df = pd.DataFrame({'result':[]})
    folds = KFold(n_splits= n_folds, shuffle=True, random_state=1001)

    if predict_classifier_bin == True or predict_classifier_nominal == True:
        params['solver']  = params['penalty'].get('solver')
        params['penalty'] = params['penalty']['penalty']
        for parameter_name in ['max_iter']:
            params[parameter_name] = int(params[parameter_name]) 

    lr = None            
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_target)):
        train_x, train_y = train_df.iloc[train_idx], train_target.iloc[train_idx]
        valid_x, valid_y = train_df.iloc[valid_idx], train_target.iloc[valid_idx]
        
        if predict_classifier_bin == True or predict_classifier_nominal == True:            
            lr = LogisticRegression(penalty=params['penalty'],\
                                  solver=params['solver'],\
                                  tol=params['tol'],\
                                  C=params['C'],\
                                  class_weight=params['class_weight'],\
                                  random_state=params['random_state'],\
                                  max_iter=params['max_iter'],\
                                  n_jobs=params['n_jobs'])

            lr.fit(train_x, train_y)
            if predict_classifier_bin:
                predict_result = lr.predict_proba(valid_x)[:, 1]
            else:
                predict_result = lr.predict(valid_x)
            predict_temp_df = pd.DataFrame({'result':predict_result},index=valid_x.index)
            predict_df = pd.concat([predict_df,predict_temp_df])

        else:
            lr = LinearRegression(fit_intercept=params['fit_intercept'],\
                                  normalize=params['normalize'],\
                                  n_jobs=params['n_jobs'])
            lr.fit(train_x, train_y)
            predict_result = lr.predict(valid_x)
            predict_temp_df = pd.DataFrame({'result':predict_result},index=valid_x.index)
            predict_df = pd.concat([predict_df,predict_temp_df])

        debug('++++++++++++++++++++Linear+++++++++++++++++++++++++++++++++++++++++++')

    predict_df.sort_index(axis=0, inplace=True)
    valid[PREDICT_NAME] = predict_df['result']
    score = 0

    if predict_classifier_bin == True:
        score = roc_auc_score(valid[target],valid[PREDICT_NAME])
    elif predict_classifier_nominal:
        valid['compare'] = valid.apply(lambda x: x[target]==x[PREDICT_NAME], axis=1)
        score = np.sum(valid['compare'])/len(valid[target])
    else:
        score = np.square(np.corrcoef(valid[target],valid[PREDICT_NAME])[0,1])    
    
    trace('linear regression: '+target+', score: '+str(score))      
    if test_dataframe is None:
        return score
    else:
        test_prediction = lr.predict(test_dataframe)
        return test_prediction, score
############################ LR domain end ###################################################

############################ KNN domain ###################################################

def int_module_knn_space(is_classifier=True):
    space = {  
            'n_neighbors': hp.quniform('n_neighbors', 5, 100, 5),
            'weights': hp.choice('weights', ['uniform','distance']),
            'algorithm': hp.choice('algorithm', ['auto']),
            'leaf_size': hp.quniform('leaf_size', 20, 100, 5),
            'p': hp.choice('p', [1, 2]),
    #        'metric': hp.choice('metric', ['minkowski']),        
            'n_jobs': hp.choice('n_jobs', [-1])
            }    
    return space

def int_module_knn_param(is_classifier=True):
    param = {  
            'n_neighbors': 20,
            'weights': 'distance',
            'algorithm': 'auto',
            'leaf_size': 30,
            'p': 2,
    #        'metric': hp.choice('metric', ['minkowski']),        
            'n_jobs': -1
            }     
    return param

def int_module_knn(params, dataframe, target, test_dataframe=None, n_folds=5):
    """
    Internal KNN model with cross validation, supporting both classification 
    prediction and regression prediction. It's aimed for easy usage and reuse 
    for all kinds of situation.
    
    Parameters
    ----------
    params : dictionary
        Parameter set with dictionary format for KNN model.
    dataframe : pandas.Dataframe
        Dataframe to process.
    target : string
        Feature name, target identifies some column which is used for 
        prediction analyze.
    test_dataframe : pandas.Dataframe, optional
        Dataframe to predict.
    n_folds : integer, optional
        Cross validation times when run KNN model with given dataframe. It's 
        often 5 or 10.
    Output
    -------
    Score when run KNN model with given param and dataframe. For regression 
    prediction, score as R2; for binary classification, score as ROC; for 
    multi-classification, score as accuracy.
    Test result from prediction if specify test_dataframe.  
    """
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import MinMaxScaler
    PREDICT_NAME = 'predict'
    
    df = dataframe
    for f_ in df.columns:
        df,_ = fast_impute.impute_mean(df, f_, target=target, intern=True)
    # Make sure parameters that need to be integers are integers
    for parameter_name in ['n_neighbors', 'leaf_size', 'p']:
        params[parameter_name] = int(params[parameter_name])     

    train_df = df.drop([target],axis=1)
    train_df, _ = one_hot_encoder(train_df, True)

    train2 = train_df.dropna(axis=0)  
    train2 = pd.concat([train2,df[target]],axis=1)

    df_importance = explore_importance_features(train2, target)
    feature_importance = df_importance.loc[df_importance['importance']>0.001, 'feature']
    feature_list = feature_importance.values
    debug(feature_list)


    train_df = train_df[feature_list]  
    min_max_scaler = MinMaxScaler()
    train_df=pd.DataFrame(min_max_scaler.fit_transform(train_df), index=train_df.index)

    train_target = df[target]
    valid = df[[target]]
    valid[PREDICT_NAME] = 0
    
    predict_classifier_bin, predict_classifier_nominal = _check_classifier(df, target)
    predict_df = pd.DataFrame({'result':[]})
    folds = KFold(n_splits= n_folds, shuffle=True, random_state=1001)

    knn = None
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_target)):
        train_x, train_y = train_df.iloc[train_idx], train_target.iloc[train_idx]
        valid_x, valid_y = train_df.iloc[valid_idx], train_target.iloc[valid_idx]

        if predict_classifier_bin == True or predict_classifier_nominal == True:
            knn = KNeighborsClassifier(n_neighbors=params['n_neighbors'],\
                                  weights=params['weights'],\
                                  algorithm=params['algorithm'],\
                                  leaf_size=params['leaf_size'],\
                                  p=params['p'],\
                                  n_jobs=params['n_jobs'])
    
            knn.fit(train_x, train_y)
#            valid.ix[valid_idx,[PREDICT_NAME]] = knn.predict_proba(valid_x)[:, 1]
            if predict_classifier_bin:
                predict_result = knn.predict_proba(valid_x)[:, 1]
            else:
                predict_result = knn.predict(valid_x)
            predict_temp_df = pd.DataFrame({'result':predict_result},index=valid_x.index)
            predict_df = pd.concat([predict_df,predict_temp_df])
    
        else:
            knn = KNeighborsRegressor(n_neighbors=params['n_neighbors'],\
                                  weights=params['weights'],\
                                  algorithm=params['algorithm'],\
                                  leaf_size=params['leaf_size'],\
                                  p=params['p'],\
                                  n_jobs=params['n_jobs'])
            
            knn.fit(train_x, train_y)
            predict_result = knn.predict(valid_x)
            predict_temp_df = pd.DataFrame({'result':predict_result},index=valid_x.index)
            predict_df = pd.concat([predict_df,predict_temp_df])


        debug('++++++++++++++++++++ KNN +++++++++++++++++++++++++++++++++++++++++')

    predict_df.sort_index(axis=0, inplace=True)
    valid[PREDICT_NAME] = predict_df['result']
    score = 0

    if predict_classifier_bin == True:
        score = roc_auc_score(valid[target],valid[PREDICT_NAME])
    elif predict_classifier_nominal:
        valid['compare'] = valid.apply(lambda x: x[target]==x[PREDICT_NAME], axis=1)
        score = np.sum(valid['compare'])/len(valid[target])
    else:
        score = np.square(np.corrcoef(valid[target],valid[PREDICT_NAME])[0,1])
    
    trace('knn: '+target+', score: '+str(score))   
    if test_dataframe is None:
        return score
    else:
        test_prediction = knn.predict(test_dataframe[feature_list])
        return test_prediction, score

############################ KNN domain end ###################################################

def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    df = pd.get_dummies(df, dummy_na= True,drop_first=True)
    new_columns = [c for c in df.columns if c not in original_columns]
    const_columns = [c for c in new_columns if df[c].dtype != 'object' and sum(df[c]) == 0 and np.std(df[c]) == 0]
    df.drop(const_columns, axis = 1, inplace = True)
    new_columns = [c for c in new_columns if c not in const_columns]
    return df, new_columns


def prediction_score(df, feature, filled):
    from sklearn.metrics import roc_auc_score
    classifier_bin, classifier_nominal= _check_classifier(df, feature)
    if classifier_bin == True:
        score = roc_auc_score(df[feature],df[filled])
    elif classifier_nominal:
        df['compare'] = df.apply(lambda x: x[feature]==x[filled], axis=1)
        score = np.sum(df['compare'])/len(df[feature])
    else:
        score = np.square(np.corrcoef(df[feature],df[filled])[0,1])
    return score


def explore_importance_features(dataframe, target, method='random forest'):
    from sklearn import preprocessing
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import RandomForestRegressor

    categorical_feats = [
        f for f in dataframe.columns if dataframe[f].dtype == 'object' and f != target
    ]

    for col in categorical_feats:
        lb = preprocessing.LabelEncoder()
        lb.fit(list(dataframe[col].values.astype('str')))
        dataframe[col] = lb.transform(list(dataframe[col].values.astype('str')))

    dataframe.fillna(-999, inplace = True)
    rf = 0
    if _is_classifier(dataframe, target):
        rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=4, max_features=0.75, random_state=2018)
    else:
        rf = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=4, max_features=0.75, random_state=2018)
    rf.fit(dataframe.drop([target],axis=1), dataframe[target])
    features = dataframe.drop([target],axis=1).columns.values

    importance_df = pd.DataFrame()
    importance_df['feature'] = features
    importance_df['importance'] = rf.feature_importances_
    importance_df.sort_values('importance',inplace=True,ascending=False)
    debug('The importance related with target',importance_df)
    return importance_df

def _check_classifier(dataframe, feature):    
    sample_num = min(10000, dataframe.shape[0])
    classifier_nominal = dataframe[feature].dtype=='object'
    classifier_bin = dataframe[feature].sample(sample_num).nunique()<=CLASS_NUM_BIN \
        and not classifier_nominal
    return classifier_bin, classifier_nominal

def _is_classifier(dataframe, target):
    sample_num = min(dataframe.shape[0], 10000)
    return dataframe[target].sample(sample_num).nunique() <= 2 or dataframe[target].dtype=='object'