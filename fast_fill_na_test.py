import pandas as pd
import fast_feature_transform as transform
import fast_fill_na as fill_na
import numpy as np
import matplotlib.pyplot as plt # for plotting
import seaborn as sns 

 
def test_fill_na_general():
    file_path = 'E:/python/credit/input/'
    file_name = 'application_train.csv'

    table = pd.read_csv(file_path+file_name)

    table = table.sample(5000)
    table.reset_index(drop=True,inplace=True)
    fill_na.setEnvInfo('E:/python/credit/input/','application_train.log')
    
    test_process_missing(table)
    test_fill_na_auto(table)
    test_fill_na_mean(table)
    test_fill_na_knn(table)
    test_fill_na_decisiontree(table)
    test_fill_na_decisiontree_cluster(table)
    test_fill_nan_lgbm(table)
    test_fill_nan_bayes(table)
    test_fill_nan_ensemble_tree(table)
#    fill_na.check_inexistence_nan(table, 'LIVINGAREA_MEDI', 'TARGET')

def test_process_missing(dataframe):
    fill_na.process_missing(dataframe, target='TARGET')
 
def test_fill_na_auto(table):
    test_fill_na_method(table, fill_na.fill_nan_auto)
    
def test_fill_na_mean(table):
    test_fill_na_method(table, fill_na.fill_nan_mean)
    
def test_fill_na_knn(table):
    test_fill_na_method(table, fill_na.fill_nan_knn)    
    
def test_fill_nan_lgbm(table):
    test_fill_na_method(table, fill_na.fill_nan_lgbm)    
    
def test_fill_nan_bayes(table):
    test_fill_na_method(table, fill_na.fill_nan_bayes)    

def test_fill_na_decisiontree(table):
    test_fill_na_method(table, fill_na.fill_nan_decisiontree)   

def test_fill_na_decisiontree_cluster(table):
    test_fill_na_method(table, fill_na.fill_nan_decisiontree_cluster)

def test_fill_nan_ensemble_tree(table):
    test_fill_na_method(table, fill_na.fill_nan_ensemble_tree)    
    
   
    
def test_fill_na_method(table,method):
    dataframe = table
    #dataframe = table[['TARGET','CODE_GENDER','FLAG_OWN_CAR','AMT_CREDIT','AMT_ANNUITY','OWN_CAR_AGE','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','AMT_REQ_CREDIT_BUREAU_WEEK']]

    dataframe.ix[[0],['FLAG_OWN_CAR']] = np.NaN
    dataframe.ix[[0],['EXT_SOURCE_2']] = np.NaN
    dataframe.ix[[0],['OWN_CAR_AGE']] = np.NaN
    dataframe.ix[[0],['EXT_SOURCE_1']] = np.NaN
    dataframe.ix[[0],['CODE_GENDER']] = np.NaN
    dataframe.ix[[0],['AMT_CREDIT']] = np.NaN
    dataframe.ix[[0],['AMT_REQ_CREDIT_BUREAU_WEEK']] = np.NaN
    df = dataframe
    df,_11 = method(df, feature='LIVINGAREA_MEDI', target='TARGET')
    df,_10 = method(df, feature='NAME_TYPE_SUITE', target='TARGET')
    df,_9 = method(df, feature='NAME_TYPE_SUITE', target='TARGET')
    df,_1 = method(df, feature='FLAG_OWN_CAR', target='TARGET')
    df,_2 = method(df, feature='EXT_SOURCE_2', target='TARGET')
    df,_3 = method(df, feature='OWN_CAR_AGE', target='TARGET')
    df,_4 = method(df, feature='EXT_SOURCE_1', target='TARGET')
    df,_5 = method(df, feature='CODE_GENDER', target='TARGET')
    df,_6 = method(df, feature='AMT_CREDIT', target='TARGET')
    df,_7 = method(df, feature='AMT_REQ_CREDIT_BUREAU_WEEK', target='TARGET')
    df,_8 = method(df, feature='OCCUPATION_TYPE', target='TARGET')
    

test_fill_na_general()