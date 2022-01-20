#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             roc_curve, auc, confusion_matrix, mean_squared_error,
                             classification_report)
class preprocess:

    def __init__(self):
        pass

    def convertstringtonumber(self, df: pd.DataFrame, lst):
        """ 字符串转为数字型"""
        for n in range(len(lst)):
            df = df.replace(lst[n], n)
        return df

    def scalex(self, X):
        """ 数值标准化"""
        nmin, nmax = 0.0, 1.0
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (nmax - nmin) + nmin
        #return X_std
        return X_scaled

    def calcrmse(self, X_train: pd.DataFrame, gensamples: pd.DataFrame):
        """计算均方误差"""
        max_column = X_train.shape[1]
        rmse_lst = []
        for col in range(max_column):
            rmse_lst.append(np.sqrt(mean_squared_error(X_train[:, col], gensamples[:, col])))
        return np.sum(rmse_lst) / max_column

    def data_normalization(self,train_df,dest_feature):
        max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
        for name in dest_feature:
            train_df[name]=train_df[[name]].apply(max_min_scaler)
        return train_df
    


    
    def create_df(self, train, test):
        combined_data = pd.concat([train, test])#.drop(['id'],axis=1)
        # Contaminsation mean pollution (outliers) in data
        tmp = train.where(train['attack_cat'] == "Normal").dropna()
        contamination = round(1 - len(tmp)/len(train), 2)
        print("train contamination ", contamination)
    
        tmp = test.where(test['attack_cat'] == "Normal").dropna()
        print("test  contamination ", round(1 - len(tmp)/len(test),2),'\n')
    
        if contamination > 0.5:
            print(f'contamination is {contamination}, which is greater than 0.5. Fixing...')
            contamination = round(1-contamination,2)
            print(f'contamination is now {contamination}')
            
        le1 = LabelEncoder()
        le = LabelEncoder()
        vector = combined_data['attack_cat']
        print("attack cat:", set(list(vector))) # use print to make it print on single line 
        combined_data['attack_cat'] = le1.fit_transform(vector)
        combined_data['proto'] = le.fit_transform(combined_data['proto'])
        combined_data['service'] = le.fit_transform(combined_data['service'])
        combined_data['state'] = le.fit_transform(combined_data['state'])
        
        print(le1.inverse_transform([0,1,2,3,4,5,6,7,8,9]))
        
        lowSTD = list(combined_data.std().to_frame().nsmallest(6, columns=0).index)
        lowCORR = list(combined_data.corr().abs().sort_values('attack_cat')['attack_cat'].nsmallest(3).index) 
        drop = set( lowCORR + lowSTD)
        
        combined_data_reduced=combined_data.drop(drop,axis=1)
        #combined_data_reduced = combined_data_reduced[combined_data_reduced['attack_cat']==6]
        #['Analysis' 'Backdoor' 'DoS' 'Exploits' 'Fuzzers' 'Generic' 'Normal' 'Reconnaissance' 'Shellcode' 'Worms']
        #[0,1,2,3,4,5,6,7,8,9]

        df = combined_data_reduced
        print(len(df.columns.values))
        
        columns = np.delete(df.columns.values, [34,35])
        print("@@@@@@@@@@@@@@@@@")
        print(columns)
        print("@@@@@@@@@@@@@@@@@")
        combined_data_reduced = self.data_normalization(df, columns)



        remain_Worms = [0,1,2,3,4,5,6,7,8] #174
        remain_Shellcode = [0,1,2,3,4,5,6,7,9]#1511
        remain_Backdoor = [0,2,3,4,5,6,7,8,9]#2329
        remain_Analysis = [1,2,3,4,5,6,7,8,9]#2677
        
        remain_Dos = [0,1,3,4,5,6,7,8,9]#16353
        remain_Exploits = [0,1,2,4,5,6,7,8,9]#44525
        remain_Fuzzers= [0,1,2,3,5,6,7,8,9]#24246
        remain_Generic = [0,1,2,3,4,6,7,8,9]#215481
        remain_Reconnaissance = [0,1,2,3,4,5,6,8,9]#13987
        
        remain_Normal = [0,1,2,3,4,5,7,8,9]#2218761
        
        Worms = combined_data_reduced[~combined_data_reduced['attack_cat'].isin(remain_Worms)] 
        Shellcodes = combined_data_reduced[~combined_data_reduced['attack_cat'].isin(remain_Shellcode)] 
        Backdoor = combined_data_reduced[~combined_data_reduced['attack_cat'].isin(remain_Backdoor)] 
        Analysis = combined_data_reduced[~combined_data_reduced['attack_cat'].isin(remain_Analysis)] 
        
        Dos = combined_data_reduced[~combined_data_reduced['attack_cat'].isin(remain_Dos)] 
        Exploits = combined_data_reduced[~combined_data_reduced['attack_cat'].isin(remain_Exploits)] 
        Fuzzers = combined_data_reduced[~combined_data_reduced['attack_cat'].isin(remain_Fuzzers)] 
        Generic = combined_data_reduced[~combined_data_reduced['attack_cat'].isin(remain_Generic)] 
        Reconnaissance = combined_data_reduced[~combined_data_reduced['attack_cat'].isin(remain_Reconnaissance)] 
        
        Normal = combined_data_reduced[~combined_data_reduced['attack_cat'].isin(remain_Normal)] 
        return Worms,Shellcodes,Backdoor,Analysis,Normal,  Dos,Exploits,Fuzzers,Generic,Reconnaissance



    def gererated_preprocess(self, generated_data: pd.DataFrame):
        """为GAN生成的数据加上attack_type"""
        df = generated_data
        df.columns = self.lbl[:-1]
        df["attack_type"] = pd.Series(["1"]*len(df), index=df.index)

        return df

    def split_df(self, df):
        #将df拆分为X和y，为机器学习做准备数据
        X_df = df.drop('attack_type', 1)
        y_df = df.attack_type

        return X_df, y_df

    def merge_df(self,df1, df2):
        df = df1.append(df2)

        return  df
