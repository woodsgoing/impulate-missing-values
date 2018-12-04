# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 13:48:02 2018

@author: june
"""

import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import time
import os

_DEBUG = True

FILL_NA = 'nan_fill'

def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    df = pd.get_dummies(df, dummy_na= True,drop_first=True)
    new_columns = [c for c in df.columns if c not in original_columns]
    const_columns = [c for c in new_columns if df[c].dtype != 'object' and sum(df[c]) == 0 and np.std(df[c]) == 0]
    df.drop(const_columns, axis = 1, inplace = True)
    new_columns = [c for c in new_columns if c not in const_columns]
    return df, new_columns

def one_hot_encoder_plus(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df2 = df[categorical_columns]
    df = pd.get_dummies(df, dummy_na= True, drop_first=True)
    new_columns = [c for c in df.columns if c not in original_columns]
    const_columns = [c for c in new_columns if df[c].dtype != 'object' and sum(df[c]) == 0]
    df.drop(const_columns, axis = 1, inplace = True)

    df = pd.concat([df2,df],axis=1)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


#global G_VALUE_RATIO
#def nominal_probality(df):
#    if df.dtype != 'object':
#        return df.mode().values[0]
#    else:
#        global G_VALUE_RATIO
#        x = df.value_counts().sort_values(ascending = False)/len(df)*100
#        for i in x.index.values:
#            if i in G_VALUE_RATIO.index.values:
#                x[i] = (x[i]-G_VALUE_RATIO[i])/G_VALUE_RATIO[i]*100
#            else:
#                x[i] = -100
#        x.sort_values(ascending = False,inplace=True)
#        return x.index.values[0]

global sourcefilename
sourcefilename=''
global sourcefilepath
sourcefilepath=''

def setEnvInfo(filepath, filename):
    """
    Configure data file path and file name. 
    This must be used before other method, as it generate log info 
    storage.
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
    global sourcefilename
    global sourcefilepath
    sourcefilename = filename
    sourcefilepath = filepath
    if not os.path.exists(sourcefilepath):
        os.mkdir(sourcefilepath) 
#    sourcefilepath = sourcefilepath + sourcefilename + '/'
#    if not os.path.exists(sourcefilepath):
#        os.mkdir(sourcefilepath) 

def _log(*arg, mode):
    global sourcefilename
    global sourcefilepath
    if sourcefilename == '' or sourcefilepath == '':
        return  
    timeline = time.strftime("%Y_%m_%d", time.localtime()) 
    with open(sourcefilepath+sourcefilename+mode+timeline+'.fillna', "a+") as text_file:
        print(*arg, file=text_file)

def trace(*arg):
    _log(*arg, mode='trace')

def debug(*arg):
    if _DEBUG == True:
        _log(*arg, mode = 'debug')


    
def process_missing(dataframe, target='', model = 'tree', method='auto', \
                    binning_missing_ratio=0.75, neighbour_num=-1):
    """
    General API to processing missing info, including add nan status and impulate nan.
    Main entrance of all internal methods for general purpose processing.
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
                dataframe to trackle missing value
    target :    string
                name of target feature, it's excluded when processing missing
    model :     options as 'tree', 'regression', 'xgboost'
                model type with which to predict target
                'tree' represent decision tree, request result contain no
                missing data
                'regression' represent all regression, like logistic regression
                or linear regression,
                request result contain no missing, no collinear and all 
                features are of number type
                'xgboost' represent xgboost, lgbm. It has little constraint and 
                can handle missing data
    Method :    options as 'auto', 'mean', 'knn', 'decision tree', 'bin',
                'bayes','lgbm'
                missing value impulate method.
    binning_missing_ratio ：     float, (0, 1]
                specify the missing ratio threshold above which binning will be
                adopted to treat missing as a category
    neighbour_num : int, large than 0 and less than dataframe size.
                -1 means auto generated value. 
                specify KNN neighbour number and only useful when method is 
                'auto' or 'knn'
        
    Return
    -------
    dataframe after process missing values from all features expect target
    """
    if sourcefilename != '' or sourcefilepath != '':
        dataframe.to_csv(sourcefilepath+sourcefilename+'.in.csv', index= False)
    dataframe = add_nan_ratio(dataframe)
    dataframe = fill_nan(dataframe, target=target, model=model, method=method, \
                         binning_missing_ratio=binning_missing_ratio, \
                         neighbour_num=neighbour_num)
#    dataframe = fill_nan_status(dataframe)
    if sourcefilename != '' or sourcefilepath != '':
        dataframe.to_csv(sourcefilepath+sourcefilename+'.out.csv', index= False)
NAN_STATUS = '_is_nan'
def fill_nan_status(dataframe, feature=''):
    """
    Add missing status for features from dataframe
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
                dataframe to process
    
    feature :   string, option
                specify feature name whose missing status is added as new 
                column. if not specified, missing status of all features are 
                added
        
    Return
    -------
    dataframe with new missing status feature(s)
    """
    df = dataframe.copy(deep=True)
    nan_num = df.isnull().sum()
#    if feature != '' and nan_num[feature] > 0:
    if feature != '':
        feature_nan = df[feature].isnull()
        return feature_nan
    else :
        for f_ in df.columns:
            if nan_num[f_] > 0:
                df[f_+NAN_STATUS] = df[f_].isnull()
        return df


def fill_nan(dataframe, target='', model = 'tree', method='auto', binning_missing_ratio=0.7, \
                 neighbour_num=-1):
    """
    Fill missing values inside dataframe with some impute algorithm
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
                dataframe to impute missing value
    
    target :    string
                name of target feature, it's excluded when processing missing
    model :     options as 'tree', 'regression', 'xgboost'
                model type with which to predict target
                'tree' represent decision tree, request result contain no
                missing data
                'regression' represent all regression, like logistic regression
                or linear regression,
                request result contain no missing, no collinear and all 
                features are of number type
                'xgboost' represent xgboost, lgbm. It has little constraint and 
                can handle missing data
                
    Method :    'auto', 'mean', 'knn', 'decision tree', 'bin','bayes','lgbm',
                'hardcode'
                'auto', adopt a relatively complex method to impute missing 
                value. It consider missing s
                
                'mean' fill mean value for number feature, normal value for 
                category value.
                'knn' fill missing with KNN method. neighbour_num specify the 
                nearest neighbour number. 
                'decision tree' fill missig with decision tree algorithm. 
                'bayes' use bayes method to fill.
                'lgbm' adopt lgbm method to predict missing
                'hardcode' fills nominal feature with missing category and fill
                number feature with some outlier value
    
    binning_missing_ratio :
                float, between (0,1) 
                Specify the threshold above which binning this feature.
    
    neighbour_num : 
                int, above 1 and less than dataframe size
                Special neighbout number of KNN. -1 means auto.
        
    Return
    -------
    dataframe with filled feature after process
    """
    dataframe_int = dataframe
    nan_num = dataframe_int.isnull().sum().sort_values(ascending=True)
    percent = dataframe_int.isnull().sum()/dataframe_int.shape[0]

    # TODO Improve if impulate in loops for better imputed other features
    
    for f_ in nan_num.index:
        if nan_num[f_] == 0:
            continue;

        # binning if binning_missing_ratio is setup
        trace(f_+' missing ratio: '+str(percent[f_]))
        if percent[f_] > binning_missing_ratio:
            dataframe_int = binning_feature(dataframe_int, f_)
            continue
        # fill nan with hardcode if it's inexistence
        if check_inexistence_nan(dataframe_int, f_, target):
            dataframe_int = fill_nan_hardcode(dataframe_int, f_)
            continue
        # filt features if nan status is highly correlated, which would mislead
        # impute
        df_train = filt_nan_hi_corr_feature(dataframe_int, f_, target)        

        if method == 'auto':
            df, acc = fill_nan_auto(df_train, f_,target)
        elif method == 'mean':
            df,acc = fill_nan_mean(df_train, f_)
        elif method == 'knn':
            df,acc = fill_nan_knn(df_train, f_, target)
        elif method == 'lgbm':
            df,acc = fill_nan_lgbm(df_train, f_, target)
        elif method == 'decision tree':
            df,acc = fill_nan_decisiontree(df_train, f_, target)
        elif method == 'bayes':
            df,acc = fill_nan_bayes(df_train, f_, target)
        elif method == 'hardcode':
            df = fill_nan_hardcode(df_train, f_)
        elif method == 'bin':
            df = binning_feature(df_train, f_)
        
        dataframe_int[f_] = df[f_]

    return dataframe_int

NAN_RATIO_FEATURE = 'na_ratio'
def add_nan_ratio(dataframe):
    """
    Add one more feature, present missing ratio. 
    If no missing at all, nothing change.
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
        
    Return
    -------
    dataframe with missing_ratio feature after process
    """
    df = dataframe
    df[NAN_RATIO_FEATURE] = df.isnull().sum(axis=1)/(df.shape[0])*100
    return df
 
BINNING_NUM = 100
def binning_feature(dataframe, feature):
    df = fill_nan_hardcode(dataframe, feature)
    if df[feature].dtype != 'object':
        df[feature] = pd.qcut(df[feature],BINNING_NUM,duplicates='drop')        
    return df

def fill_na_performance(df, feature, filled):
        if _DEBUG == False:
            return
        feature_diff = pd.DataFrame()
        feature_diff['original'] = df[feature]
        feature_diff['filled'] = df[filled]
        feature_diff.dropna(axis=0, inplace=True)
        if df[feature].dtype != 'object':
            feature_diff['diff'] = feature_diff['filled']-feature_diff['original']
            std_diff = np.std(feature_diff['diff'])
            std_orig = np.std(feature_diff['original'])+0.0001
            R2 = 1 - np.square(std_diff/std_orig)

            debug('numeric filling_na')
            debug('std_diff', std_diff)
            debug('std_org', std_orig)
            return R2
        else:
            feature_diff['diff'] = feature_diff['filled']==feature_diff['original']
            hit_ratio = feature_diff['diff'].sum()/feature_diff.shape[0]
            debug('nominal filling_na', hit_ratio)
            return hit_ratio

# check if inexistence corr, or inexist & value corr
# if fillna, similar with inexistence corr.
INEXISTENCE_NAN_SYNC_RATIO = 0.5
INEXISTENCE_NAN_CORR_RATIO = 0.98
INEXISTENCE_NAN_NUMBER2NOMINAL_NUM = 30
INEXISTENCE_NAN_NUMBER2NOMINAL_RATIO = 1/10
def check_inexistence_nan(dataframe, feature, target):
    feature_nan = dataframe[feature].isnull()
#    total_len = dataframe.shape[0]
    featue_nan_len = sum(feature_nan)
    if featue_nan_len == 0:
        return False
    
    for feature_test in dataframe.columns:
        if feature_test == feature or feature_test == target:
            continue
        
#        if feature_test == 'ELEVATORS_MODE':
#            feature_test = feature_test
            
        f_temp = dataframe[feature_test]
        df = pd.concat([feature_nan,f_temp],axis=1)
        df.columns = [feature,feature_test]
#        f_nan = dataframe[feature_test].isnull()
#        df = pd.concat([df,f_nan],axis=1)
#        df.columns = [feature,feature_test,'feature_test_nan']
        df.dropna(subset=[feature_test],inplace=True)
        featue_nan_len2 = sum(df[feature])   
#        if featue_nan_len2/featue_nan_len < INEXISTENCE_NAN_SYNC_RATIO
        if featue_nan_len2 == 0:
            continue
        
        # transform number2Str which behaviors as category
        if df[feature_test].dtype != 'object':
            value_count = df[feature_test].value_counts()
            if len(value_count) < INEXISTENCE_NAN_NUMBER2NOMINAL_NUM \
            and len(value_count)/df.shape[0] < INEXISTENCE_NAN_NUMBER2NOMINAL_RATIO:
                df[feature_test] = df[feature_test].apply(lambda x:str(x))
                debug(feature+' inexistant with' + feature_test + ' value count: '+str(value_count)+' Ratio: '+str(len(value_count)/df.shape[0]))
        
        df_feature_nan = df[df[feature]==True]
        # binning number which range contain feature_nan (95%)
        '''
        #TODO consider more
        if df[feature_test].dtype != 'object':
            range_max = max(df_feature_nan[feature_test])
            range_min = min(df_feature_nan[feature_test])
            #TODO update with confidence interval
            df[feature_test] = df[feature_test].apply(lambda x:\
              'nan_stuff' if x >= range_min and x <= range_max else 'not_nan')
            df_feature_nan = df[df[feature]==True]
        '''
        # Here all feature_test are of object type
        # binning cat which contain most(95%) feature_na
#        if df[feature_test].dtype == 'object':
        if df_feature_nan[feature_test].dtype == 'object':
            df_feature_nan_ratio = df_feature_nan[feature_test].value_counts().sort_values(ascending = False)\
                /df_feature_nan.shape[0]
            count = 0
            value_list = []
            for value_index in df_feature_nan_ratio.index:
                if df_feature_nan_ratio.index.dtype != 'object':
                    value_index = str(value_index) 
                count = count + df_feature_nan_ratio.loc[value_index]
                value_list.append(value_index)
                if count >= INEXISTENCE_NAN_CORR_RATIO:
                    break
            debug(feature+' inexistant with '+feature_test+' ratio count: '+str(count))
            df_nan_len = df_feature_nan.shape[0]


            index_list = [index for index in df.index if df[feature_test][index] in value_list]
            df_in_value = df.loc[index_list]
            df_in_value_len = df_in_value.shape[0]
            test_values_ratio = df_nan_len / df_in_value_len

            debug(feature+' inexistant with '+feature_test+' test ratio: ---------- '+str(test_values_ratio))
            if test_values_ratio > INEXISTENCE_NAN_CORR_RATIO:
                trace('inexistant '+feature+' -> '+feature_test+' test ratio: '+str(test_values_ratio))
                return True
    return False

NAN_INT_RATIO = 0.50
NAN_INT_EXT_RATIO = 1.5
def filt_nan_hi_corr_feature(dataframe, feature, target):
    df_nan = fill_nan_status(dataframe)
    nan_status_list = [f_ for f_ in df_nan.columns if NAN_STATUS in f_ \
                       and target not in f_ and feature not in f_]
    feature_nan = df_nan[feature+NAN_STATUS]
    feature_data = 1 - feature_nan
    nan_hi_corr = []
    for f_ in nan_status_list:
        internal_sub_nan = df_nan[f_] & feature_nan
        internal_nan_ratio = sum(internal_sub_nan)/sum(feature_nan)
        external_sub_nan = df_nan[f_] & feature_data
        external_nan_ratio = sum(external_sub_nan)/sum(feature_data)
        if internal_nan_ratio > NAN_INT_RATIO\
        or internal_nan_ratio/(external_nan_ratio+0.001)>NAN_INT_EXT_RATIO:
            nan_hi_corr.append(f_)
            debug('nan_hi_corr_feature '+feature+' & '+f_[:-6]+', nan_ratio: '+str(internal_nan_ratio))
            debug('nan_hi_corr_feature: int/ext nan_ratio'+str(internal_nan_ratio/(external_nan_ratio+0.001)))
    feature_drop = [f_.replace(NAN_STATUS,'') for f_ in nan_hi_corr]
    df_train = dataframe.drop(feature_drop,axis=1)
    trace('nan_hi_corr_feature '+feature+' ', feature_drop)
    return df_train

AUTO_NAN_RATIO_THRESHOLD_BINNING = 0.50
AUTO_TARGET_OBJECT_ACC = 0.65
AUTO_TARGET_NUMBER_R2 = 0.5
def fill_nan_auto(dataframe, feature,target, model='tree'):
    '''
    model tree, regression, xgboost(lgbm)
    '''

    # champion prediction
    predict = [\
               fill_nan_knn, \
               fill_nan_lgbm, \
               fill_nan_decisiontree,\
               fill_nan_bayes,\
               fill_nan_mean]
    df = pd.DataFrame()
    acc = 0
    method_index = -1
    for index in range(len(predict)):
        test_df = dataframe[dataframe[feature].isnull()].drop(feature,axis=1)
        df_temp, acc_temp = predict[index](dataframe.copy(deep=True), feature,target)
        test_df2 = df_temp[df_temp[feature].isnull()].drop(feature,axis=1)
        if test_df2.shape[0] != 0:
            trace('fill_nan_auto fail to impute with ' + predict[method_index].__name__)
            trace('initial len: ' + str(test_df.shape[0])+' After len: '+str(test_df2.shape[0]))
        if acc_temp > acc:
            df = df_temp
            acc = acc_temp
            method_index = index

    trace('fill_nan_auto feature: '+feature+predict[method_index].__name__+' Accuracy:'+str(acc))    
    temp = dataframe.copy(deep=True)
    temp[feature] = df[feature]
    df = temp       
            
    if df[feature].dtype == 'object' and acc > AUTO_TARGET_OBJECT_ACC:
        acc = acc
    elif acc > AUTO_TARGET_NUMBER_R2:
        acc = acc
    else:
        percent = dataframe[[feature]].isnull().sum()/dataframe[[feature]].shape[0]
        if model == 'regression' and percent[feature] > AUTO_NAN_RATIO_THRESHOLD_BINNING:
            df = binning_feature(dataframe,feature)
            trace('fill_nan_auto binning feature:'+feature)    
        elif model == 'tree':
            trace('fill_nan_auto hardcode feature:'+feature)  
            df = fill_nan_hardcode(dataframe,feature)
    return df,acc


def fill_nan_mean(dataframe, feature, target='', intern=False):
    df = dataframe
    mean = 0
    if df[feature].dtype == 'object':
        mean = df[feature].fillna(FILL_NA).mode().values[0]
    else:
        mean = df[feature].mean()
        if df[feature].dtype == 'int':
            mean = round(mean)
      
    df_pfm = pd.DataFrame()
    df_pfm['original'] = dataframe[feature]
    df_pfm['filled'] = mean
    df_pfm = df_pfm.loc[df_pfm['original'].notnull()]
    acc = 0
    if intern == False:
        acc = fill_na_performance(df_pfm, 'original', 'filled')
        
    debug('fill_nan_mean intern:' + str(intern) + feature+' acc: '+ str(acc))
    df[feature].fillna(mean, inplace=True)
    return df,acc

NEIGHBOR_SIZE = 6
def fill_nan_knn(dataframe, feature, target,neighbour_num=-1):
    KNN_NAME=feature+'_knn'
    feature_list = [f_ for f_ in dataframe.columns if f_ != feature and f_!=target]
#    size = len(dataframe[feature])
    size = dataframe.shape[0]
    df = dataframe
    df = df.reset_index(drop=True)

    train = df[feature_list]
    train, _ = one_hot_encoder(train, True)
    for f_ in train.columns:
        train,_=fill_nan_mean(train, f_, intern=True)
    from sklearn.preprocessing import MinMaxScaler
    min_max_scaler = MinMaxScaler()
    train=pd.DataFrame(min_max_scaler.fit_transform(train))

    train2 = pd.concat([train, df[feature]],axis=1)
    train2.dropna(axis=0, inplace=True)
#    # check missing state
#    total = train2.isnull().sum().sort_values(ascending = False)
#    percent = (train2.isnull().sum()/train2.isnull().count()*100).sort_values(ascending = False)
#    missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#    debug('missing rank',missing_data.head(40 if len(total)>40 else len(total)))
    
    df_importance = explore_importance_features(train2, feature)
    feature_importance = df_importance.loc[df_importance['importance']>0.001, 'feature']
    feature_list = feature_importance.values
    debug(feature_list)

    train = train[feature_list]
    if neighbour_num == -1:
        neighbour_num = int(size/100)
        if size/100 > NEIGHBOR_SIZE:
            neighbour_num = NEIGHBOR_SIZE

    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=neighbour_num, algorithm='auto').fit(train)
    indices = nbrs.kneighbors(train, return_distance=False)
            
    # fill nan value with KNN value
    df[KNN_NAME] = None
    if df[feature].dtype == 'object':
        for i in df.index.values:
            if len(df[feature][indices[i]].mode()) > 0:
                df[KNN_NAME][i] = df[feature][indices[i]].mode().values[0]
    else:
        for i in df.index.values:
            df[KNN_NAME][i] = df[feature][indices[i]].mean()
            
    # eliminate nan value with 2nd order KNN value
    if df[feature].dtype == 'object':
        for i in df.index.values:
            if df[KNN_NAME][i] == np.nan:
                if len(df[KNN_NAME][indices[i]].mode()) > 0:
                    df[KNN_NAME][i] = df[KNN_NAME][indices[i]].mode().values[0]
    else:
        for i in df.index.values:
            if df[KNN_NAME][i] == np.nan:
                df[KNN_NAME][i] = df[KNN_NAME][indices[i]].mean()

    # handle some special case when nan are clustered
    df, _=fill_nan_mean(df, KNN_NAME, intern=True)

    df_pf = df.loc[df[feature].notnull()]
    acc = fill_na_performance(df_pf, feature, KNN_NAME)    
    df[feature].fillna(df[KNN_NAME],inplace=True)
    df.drop(KNN_NAME, axis=1, inplace=True)
    trace('fill_nan_knn' + feature + ' acc: '+str(acc))
    return df, acc

def fill_nan_bayes(dataframe, feature, target):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import BayesianRidge
    from sklearn.model_selection import KFold
    PREDICT_NAME = 'predict'
 
    df = dataframe

    feature_list = [f_ for f_ in dataframe.columns if f_ != feature and f_ != target]
    train = df[feature_list]
    train, _ = one_hot_encoder(train, True)
    for f_ in train.columns:
        train, _=fill_nan_mean(train, f_, intern=True)

    train = pd.concat([train, df[feature]],axis=1)
    train_df = train[train[feature].notnull()].drop(feature,axis=1)
    train_target = train.loc[train[feature].notnull(),feature]
    test_df = train[train[feature].isnull()].drop(feature,axis=1)
    test_target = train[train[feature].isnull()][[feature]]
    valid = train[train[feature].notnull()][[feature]]
    valid[PREDICT_NAME] = 0
    valid.reset_index(inplace=True)
    folds = KFold(n_splits= 5, shuffle=True, random_state=1001)

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_target)):
        train_x, train_y = train_df.iloc[train_idx], train_target.iloc[train_idx]
        valid_x, valid_y = train_df.iloc[valid_idx], train_target.iloc[valid_idx]
    
        rf = 0
        if train_target.dtype == 'object':
            rf = GaussianNB()
        else:
            rf = BayesianRidge(compute_score=True)
#            BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False,\
#            copy_X=True, fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06,\
#            n_iter=300, normalize=False, tol=0.001, verbose=False)
        rf.fit(train_x, train_y)
        valid.ix[valid_idx,[PREDICT_NAME]] = rf.predict(valid_x)
#        d= rf.predict(test_df)
        debug('+++++++++++++++++++++++++++BAYES +++'+feature+'+++++++++++++++++++++++++++++++++')
        if test_target.shape[0]>0:
            test_target.loc[:,PREDICT_NAME] = rf.predict(test_df) 
    
    acc = fill_na_performance(valid, feature, PREDICT_NAME)
    for i in test_target.index.values:
        df.loc[i,feature] = test_target.loc[i,PREDICT_NAME]
    
    trace('fill_nan_bayes ' + feature + ' acc: '+str(acc))    
    return df, acc
   

def fill_nan_lgbm(dataframe, feature, target):

    from sklearn import preprocessing
    from lightgbm import LGBMClassifier
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import KFold
    PREDICT_NAME = 'predict'
    
    df = dataframe
    df = df.reset_index(drop=True)

    feature_list = [f_ for f_ in dataframe.columns if f_ != feature and f_ != target]
    train = df[feature_list]
    train, _ = one_hot_encoder(train, True)
    for f_ in train.columns:
        train, _=fill_nan_mean(train, f_,intern = True)

    train = pd.concat([train, df[feature]],axis=1)
    train_df = train[train[feature].notnull()].drop(feature,axis=1)
    train_target = train.loc[train[feature].notnull(),feature]
    test_df = train[train[feature].isnull()].drop(feature,axis=1)
    test_target = train[train[feature].isnull()][[feature]]
    valid = train[train[feature].notnull()][[feature]]
    valid[PREDICT_NAME] = 0
    valid.reset_index(inplace=True)
    folds = KFold(n_splits= 5, shuffle=True, random_state=1001)

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_target)):
        train_x, train_y = train_df.iloc[train_idx], train_target.iloc[train_idx]
        valid_x, valid_y = train_df.iloc[valid_idx], train_target.iloc[valid_idx]
    
        # Fix new label in valid_y
        if train_y.dtype == 'object':
            train_y_value_list = train_y.unique()
            train_y_value_mode = train_y.mode().values[0]
            valid_y = valid_y.apply(lambda x: x if x in train_y_value_list\
                                    else train_y_value_mode)
        
        
        lgbm = 0
        if train_target.dtype == 'object':
            lgbm = LGBMClassifier(
            nthread=4,
            n_estimators=1000,
            learning_rate=0.02,
#            num_leaves=34,
            num_leaves=6,
            colsample_bytree=0.95,
            subsample=0.87,
#            max_depth=8,
            reg_alpha=0.04,
            reg_lambda=0.07,
            min_split_gain=0.02,
            min_child_weight=40,
            silent=-1,
            verbose=-1, )

        else:
            lgbm = LGBMRegressor(
            nthread=4,
            n_estimators=1000,
            learning_rate=0.02,
#            num_leaves=34,
            num_leaves=6,
            colsample_bytree=0.95,
            subsample=0.87,
#            max_depth=8,
            reg_alpha=0.04,
            reg_lambda=0.07,
            min_split_gain=0.02,
            min_child_weight=40,
            silent=-1,
            verbose=-1, )

        debug('++++++++++++++++++++LGBM++++++++++'+feature+'+++++++++++++++++++++++++++++++++')
        lgbm.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], \
                                             verbose= 1000, early_stopping_rounds= 200)
        valid.ix[valid_idx,[PREDICT_NAME]] = lgbm.predict(valid_x)
        if test_target.shape[0]>0:
            test_target.loc[:,PREDICT_NAME] = lgbm.predict(test_df) 
    
    acc = fill_na_performance(valid, feature, PREDICT_NAME)
    if test_target.shape[0]>0:
        for i in test_target.index.values:
            df.loc[i,feature] = test_target.loc[i,PREDICT_NAME]
    
    trace('fill_nan_lgbm ' + feature + ' acc: '+str(acc))       
    return df, acc

NAN_DISTANCE = -2
def fill_nan_hardcode(dataframe, feature):
    df = dataframe
    if df[feature].dtype == 'object':
        df[feature].fillna(FILL_NA, inplace=True)
    else:
        mean = dataframe[feature].dropna().mean()
        factor = NAN_DISTANCE if mean>=0 else NAN_DISTANCE*-1
        fill_na = factor*max(abs(dataframe[feature]))
        df[feature].fillna(fill_na,inplace=True)    
    return df


     
    
DECISION_TREE_MIN_SAMPLES_LEAF_MAX = 5
def fill_nan_decisiontree(dataframe, feature,target):
        
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import KFold
    PREDICT_NAME = 'predict'
 
    df = dataframe

    feature_list = [f_ for f_ in dataframe.columns if f_ != feature and f_ != target]
    train = df[feature_list]
    train, _ = one_hot_encoder(train, True)
    for f_ in train.columns:
        train, _=fill_nan_mean(train, f_,intern=True)
  
    train = pd.concat([train, df[feature]],axis=1)
    train_df = train[train[feature].notnull()].drop(feature,axis=1)
    train_target = train.loc[train[feature].notnull(),feature]
    test_df = train[train[feature].isnull()].drop(feature,axis=1)
    test_target = train[train[feature].isnull()][[feature]]
    valid = train[train[feature].notnull()][[feature]]
    valid[PREDICT_NAME] = 0
    valid.reset_index(inplace=True)
    folds = KFold(n_splits= 5, shuffle=True, random_state=1001)

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_target)):
        train_x, train_y = train_df.iloc[train_idx], train_target.iloc[train_idx]
        valid_x, valid_y = train_df.iloc[valid_idx], train_target.iloc[valid_idx]
    
        rf = 0
        if train_target.dtype == 'object':
            rf = DecisionTreeClassifier(random_state=2018)
        else:
            rf = DecisionTreeRegressor(random_state=2018)
            
        rf.fit(train_x, train_y)
        valid.ix[valid_idx,[PREDICT_NAME]] = rf.predict(valid_x)        
        if test_target.shape[0]>0:
            test_target.loc[:,PREDICT_NAME] = rf.predict(test_df) 
    
    acc = fill_na_performance(valid, feature, PREDICT_NAME)
    for i in test_target.index.values:
        df.loc[i,feature] = test_target.loc[i,PREDICT_NAME]

    trace('fill_nan_decisiontree ' + feature + ' acc: '+str(acc))           
    return df, acc

def nominal_mode(df):
    return df.mode().values[0]
DECISION_TREE_CLUSTER_MIN_SAMPLES_LEAF_MAX = 10
def fill_nan_decisiontree_cluster(dataframe, feature,target):
    from sklearn import tree    
    df = dataframe
    x = df.drop([target,feature],axis=1)
    y = df[target]
    x, _ = one_hot_encoder(x, True)
    for f_ in x.columns:
        x, _=fill_nan_mean(x,f_,intern=True)

    min_samples = DECISION_TREE_CLUSTER_MIN_SAMPLES_LEAF_MAX \
    if int(len(df[feature])/100)>DECISION_TREE_CLUSTER_MIN_SAMPLES_LEAF_MAX \
    else int(len(df[feature])/100)
    clf = tree.DecisionTreeClassifier(min_samples_leaf= min_samples)
    clf = clf.fit(x, y)
    df['leaf'] = clf.apply(x)
    aggr = {}
    
    if df[feature].dtype == 'object':
#        global G_VALUE_RATIO
#        G_VALUE_RATIO = df[feature].value_counts().sort_values(ascending = False)/ \
#        len(dataframe[feature])*100
#        aggr[feature] = [nominal_probality]
        aggr[feature] = [nominal_mode]
    else:
        aggr[feature] = ['mean']
        
    x_agg = df.groupby('leaf').agg(aggr)
    x_agg.reset_index(inplace=True)
    x_agg.columns = ['leaf','value']
    df['leaf_value'] = df['leaf'].apply(lambda a: x_agg.loc[x_agg['leaf']==a, 'value'].values[0])
    df[feature].fillna(df['leaf_value'], inplace=True)
    
    acc = fill_na_performance(df, feature, 'leaf_value')

    df.drop(['leaf','leaf_value'], axis=1, inplace=True)
    
    trace('fill_nan_decisiontree_cluster ' + feature + ' acc: '+str(acc))   
    return df, acc

def fill_nan_ensemble_tree(dataframe, feature, target):
# performance horribly slow
    from sklearn import preprocessing
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold
    PREDICT_NAME = 'predict'
#    
#    df = dataframe
#    categorical_feats = [
#        f for f in df.columns if df[f].dtype == 'object' \
#        and f != target and f != feature
#    ]
#
#    for col in categorical_feats:
#        lb = preprocessing.LabelEncoder()
#        lb.fit(list(df[col].values.astype('str')))
#        df[col] = lb.transform(list(df[col].values.astype('str')))
#
#
#    df.fillna(-999, inplace = True)
#    
    df = dataframe
#    df = df.reset_index(drop=True)

    feature_list = [f_ for f_ in dataframe.columns if f_ != feature and f_ != target]
    train = df[feature_list]
    train, _ = one_hot_encoder(train, True)
    for f_ in train.columns:
        train, _=fill_nan_mean(train, f_ ,intern=True)
  
    train = pd.concat([train, df[feature]],axis=1)
    train_df = train[train[feature].notnull()].drop(feature,axis=1)
    train_target = train.loc[train[feature].notnull(),feature]
    test_df = train[train[feature].isnull()].drop(feature,axis=1)
    test_target = train[train[feature].isnull()][[feature]]
    valid = train[train[feature].notnull()][[feature]]
    valid[PREDICT_NAME] = 0
    valid.reset_index(inplace=True)
    folds = KFold(n_splits= 5, shuffle=True, random_state=1001)

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_target)):
        train_x, train_y = train_df.iloc[train_idx], train_target.iloc[train_idx]
        valid_x, valid_y = train_df.iloc[valid_idx], train_target.iloc[valid_idx]
    
        rf = 0
        if train_target.dtype == 'object':
            rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=10, max_features=0.8, random_state=2018)
        else:
            rf = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=10, max_features=0.8, random_state=2018)
        rf.fit(train_x, train_y)

        valid.ix[valid_idx,[PREDICT_NAME]] = rf.predict(valid_x)

        if test_target.shape[0]>0:
            test_target.loc[:,PREDICT_NAME] = rf.predict(test_df) 
    
    acc = fill_na_performance(valid, feature, PREDICT_NAME)
    for i in test_target.index.values:
        df.loc[i,feature] = test_target.loc[i,PREDICT_NAME]

    trace('fill_nan_ensembletree ' + feature + ' acc: '+str(acc))           
    return df, acc
    
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
    if dataframe[target].dtype == 'object':
        rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=4, max_features=0.5, random_state=2018)
    else:
        rf = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=4, max_features=0.5, random_state=2018)
    rf.fit(dataframe.drop([target],axis=1), dataframe[target])
    features = dataframe.drop([target],axis=1).columns.values

    importance_df = pd.DataFrame()
    importance_df['feature'] = features
    importance_df['importance'] = rf.feature_importances_
    importance_df.sort_values('importance',inplace=True,ascending=False)
    debug('The importance related with target',importance_df)
    return importance_df