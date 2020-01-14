#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author: Jiang Yitao
import os
import zipfile
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import model_selection

#%matplotlib inline
warnings.filterwarnings('ignore')
os.chdir(r'D:\Users\JIANGYITAO894\Desktop\JYT-IN-OC\01_working\coding\Tianchi_DCIC')


### read dataset from zip file
def read_zip_data(zip_file):
    tmp = []
    z = zipfile.ZipFile(zip_file + ".zip", "r")
    for filename in z.namelist():
        if filename.find('.csv') != -1 :
            f = z.open(filename)
            df = pd.read_csv(f)
            tmp.append(df)
    df = pd.concat(tmp)
    if df.shape[1] == 7:
        df.columns = ['ship_id','x','y','speed','direction','time','type']
    else:
        df.columns = ['ship_id','x','y','speed','direction','time']
    return df
    

def clean_data(df, train_flag=1):

    if train_flag == 1 :
        ### y labeling
        df_y = df[['ship_id','type']].drop_duplicates('ship_id')
        type_map = dict(zip(df_y['type'].unique(), np.arange(3)))
        df_y['type'] = df_y['type'].replace(type_map)
        df_y = df_y.reset_index(drop=True)
    else:
        df_y = df[['ship_id']].drop_duplicates('ship_id')

    ### deal with time 
    df['time'] = pd.to_datetime(df['time'], format='%m%d %H:%M:%S')
    df['date'] = df['time'].dt.date
    df['month'] = df['time'].apply(lambda d: d.month)
    df['day'] = df['time'].apply(lambda d: d.day)
    df['DayOfWeek'] = df['time'].apply(lambda d: d.dayofweek)
    df['DayName'] = df['time'].apply(lambda d: d.weekday_name)
    df['DayOfYear'] = df['time'].apply(lambda d: d.dayofyear)
    df['WeekOfYear'] = df['time'].apply(lambda d: d.weekofyear)
    df['Quarter'] = df['time'].apply(lambda d: d.quarter)
    df['Hour'] = df['time'].apply(lambda d: d.hour)

    cut_hour = [-1, 5, 11, 16, 21, 23]
    cut_labels = ['last night', 'morning', 'afternoon', 'evening', 'Night']
    df['Hour_cut'] = pd.cut(df['Hour'], bins=cut_hour, labels=cut_labels)
    map_dict = dict(zip(cut_labels, np.arange(5)))
    df['Hour_cut'] = df['Hour_cut'].replace(map_dict)

    cut_direction = [0, 90, 180, 270, 360]
    df['direction_cut'] = pd.cut(df['direction'], bins=cut_direction, labels=np.arange(len(cut_direction)-1))

    ### aggregation feature
    agg_dict = {}
    stat_list = ['min', 'max', 'mean', 'var', 'std', 'skew']
    agg_dict.update(dict.fromkeys(['x', 'y', 'speed', 'direction'], stat_list))
    agg_dict.update(dict.fromkeys(['DayOfWeek', 'Quarter', 'Hour', 'Hour_cut'], ['median']))

    df_agg = df.groupby('ship_id').agg(agg_dict)
    df_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in df_agg.columns.tolist()])
    df_agg = df_agg.reset_index()
    out_df = pd.merge(df_y, df_agg, on='ship_id', how='inner')

    list_col = ['Hour','Hour_cut','DayOfWeek','direction_cut']
    for col in list_col:
        dict_ = df.groupby('ship_id')[col].agg(lambda x:x.value_counts().index[0]).to_dict()
        new = col + '_MODE'
        out_df[new] = out_df['ship_id'].replace(dict_)

    list_col = ['Hour','DayOfWeek','day']
    for col in list_col:
        _nunique = df.groupby('ship_id')[col].nunique().to_dict()
        new = col + '_cnt'
        out_df[new] = out_df['ship_id'].map(_nunique)

    out_df['x_MAX_x_MIN'] = out_df['x_MAX'] - out_df['x_MIN']
    out_df['y_MAX_y_MIN'] = out_df['y_MAX'] - out_df['y_MIN']
    out_df['AREA'] = out_df['x_MAX_x_MIN'] * out_df['y_MAX_y_MIN']
    out_df['SLOPE'] = out_df['y_MAX_y_MIN'] / np.where(out_df['x_MAX_x_MIN']==0, 0.001, out_df['x_MAX_x_MIN'])

    out_df['dist_Manhattan'] =  out_df['x_MAX_x_MIN'].abs() + out_df['y_MAX_y_MIN'].abs()
    
    return out_df
def model_lgb(train, test, label='y'):
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score

    X_train = train.drop([label], axis=1)
    X_train = X_train.drop(['ship_id'], axis=1)
    y_train = np.array(train[label])
    dev_X, val_X, dev_y, val_y = train_test_split(X_train, y_train, test_size = 0.3, random_state = 233)
    params = {
        "objective" : "multiclass",
        "metric" : "multi_error",
        "num_class" : 3,
        "max_depth" : 5,
        "num_leaves" : 60,
        "learning_rate" : 0.05,
        "bagging_fraction" : 0.6,
        "feature_fraction" : 0.6,
        "bagging_frequency" : 6,
        "bagging_seed" : 42,
        "verbosity" : -1,
        "seed": 233
    }

    def lgb_f1_score(preds, dtrain):
        labels = dtrain.get_label()
        preds = preds.reshape(-1, 3)
        preds = preds.argmax(axis = 1)
        f_score = f1_score(preds, labels, average = 'macro')
        return 'f1_score', f_score, True

    lgtrain = lgb.Dataset(dev_X, label=dev_y)
    lgval = lgb.Dataset(val_X, label=val_y)

    X_test = test.drop(['ship_id'], axis=1)
    lgtest_X = lgb.Dataset(X_test)

    evals_result = {}
    model = lgb.train(params, lgtrain, 5000, 
                      valid_sets=[lgtrain, lgval], 
                      early_stopping_rounds=500, 
                      verbose_eval=100,
                      feval = lgb_f1_score,
                      evals_result=evals_result)
    
    pred_test_y = np.expm1(model.predict(X_test, num_iteration=model.best_iteration))
    pred_train = np.expm1(model.predict(X_train, num_iteration=model.best_iteration))
    lgb_F1 = f1_score(np.argmax(pred_train, axis=1), y_train, average = 'macro')
    print("The Light GBM Traning_Set's F1-Score is", lgb_F1)
    return pred_test_y, model, evals_result

if __name__ == '__main__':
    print("reading dataset...")
    train = read_zip_data('hy_round1_train_20200102')
    test = read_zip_data('hy_round1_testA_20200102')
    
    print("data processing...")
    train_df = clean_data(train)
    test_df = clean_data(test, 0)
    
    ### training model
    pred_test_y, model, evals_result = model_lgb(train_df, test_df, label='type')
    
    ### output submission
    type_map = dict(zip(train['type'].unique(), np.arange(3)))
    type_map_rev = {v:k for k,v in type_map.items()}
    pred = np.argmax(pred_test_y, axis=1)
    submission = test_df[['ship_id']]
    submission['pred'] = pred
    print("predicting...")
    submission['pred'] = submission['pred'].map(type_map_rev)
    submission.to_csv('result_14Jan.csv', index=None, header=None)
    
    print("All done")