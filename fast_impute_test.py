import pandas as pd
import fast_impute as impute
import numpy as np
import matplotlib.pyplot as plt # for plotting
import seaborn as sns 

 
def test_impute_general():
    file_path = 'E:/python/credit/input/'
    file_name = 'application_train_sample.csv'

    table = pd.read_csv(file_path+file_name)

    table = table.sample(10000)
    table.reset_index(drop=True,inplace=True)
    impute.setEnvInfo('E:/python/credit/input/','application_train.log')
    
#    test_process_missing(table)
#    test_impute_auto(table)
#    test_impute_mean(table)
    test_impute_knn(table)
    test_impute_lgbm(table)
    test_impute_random_forest(table)
#    test_imputen_ensemble_tree(table)
#    impute.check_inexistence_nan(table, 'LIVINGAREA_MEDI', 'TARGET')

def test_process_missing(dataframe):
    impute.process_missing(dataframe, target='TARGET')
 
def test_impute_auto(table):
    test_impute_method(table, impute.impute_auto)
    
def test_impute_mean(table):
    test_impute_method(table, impute.impute_mean)
    
def test_impute_knn(table):
    test_impute_method(table, impute.impute_knn)    
    
def test_impute_lgbm(table):
    test_impute_method(table, impute.impute_lgbm)    
    
def test_impute_random_forest(table):
    test_impute_method(table, impute.impute_random_forest)    

   
    
def test_impute_method(table,method):
    dataframe = table.copy(deep=True)
    #dataframe = table[['TARGET','CODE_GENDER','FLAG_OWN_CAR','AMT_CREDIT','AMT_ANNUITY','OWN_CAR_AGE','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','AMT_REQ_CREDIT_BUREAU_WEEK']]

    dataframe.ix[[0],['FLAG_OWN_CAR']] = np.NaN
    dataframe.ix[[0],['EXT_SOURCE_2']] = np.NaN
    dataframe.ix[[0],['OWN_CAR_AGE']] = np.NaN
    dataframe.ix[[0],['EXT_SOURCE_1']] = np.NaN
    dataframe.ix[[0],['CODE_GENDER']] = np.NaN
    dataframe.ix[[0],['AMT_CREDIT']] = np.NaN
    dataframe.ix[[0],['OCCUPATION_TYPE']] = np.NaN
    dataframe.ix[[0],['AMT_REQ_CREDIT_BUREAU_WEEK']] = np.NaN
    df = dataframe

    print('test 2')
    df,r_2 = method(df, feature='EXT_SOURCE_2', target='TARGET')
    print('test 1')
    df,r_1 = method(df, feature='FLAG_OWN_CAR', target='TARGET')
    print('test 3')
    df,r_3 = method(df, feature='OWN_CAR_AGE', target='TARGET')
    print('test 4')
    df,r_4 = method(df, feature='EXT_SOURCE_1', target='TARGET')
    print('test 5')
    df,r_5 = method(df, feature='CODE_GENDER', target='TARGET')
    print('test 6')
    df,r_6 = method(df, feature='AMT_CREDIT', target='TARGET')
    print('test 7')
    df,r_7 = method(df, feature='AMT_REQ_CREDIT_BUREAU_WEEK', target='TARGET')
    print('test 8')
    df,r_8 = method(df, feature='OCCUPATION_TYPE', target='TARGET')
    print('test 9')
    df,r_9 = method(df, feature='NAME_TYPE_SUITE', target='TARGET')
    print('test 11')
    df,r_11 = method(df, feature='LIVINGAREA_MEDI', target='TARGET')
    

test_impute_general()