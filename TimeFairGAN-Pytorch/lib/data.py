"""Reimplement TimeGAN-pytorch Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com)

-----------------------------

data.py

(0) MinMaxScaler: Min Max normalizer
(1) sine_data_generation: Generate sine dataset
(2) real_data_loading: Load and preprocess real data
  - stock_data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG
  - energy_data: http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
(3) load_data: download or generate data
(4): batch_generator: mini-batch generator
"""



# Necessary Packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import QuantileTransformer
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")


def sine_data_generation (no, seq_len, dim):
  # Initialize the output
  data = list()
  # Generate sine data
  for i in range(no):      
    # Initialize each time-series
    temp = list()
    # For each feature
    for k in range(dim):
      # Randomly drawn frequency and phase
      freq = np.random.uniform(0, 0.1)            
      phase = np.random.uniform(0, 0.1)       
      # Generate sine signal based on the drawn frequency and phase
      temp_data = [np.sin(freq * j + phase) for j in range(seq_len)] 
      temp.append(temp_data)   
    # Align row/column
    temp = np.transpose(np.asarray(temp))        
    # Normalize to [0,1]
    temp = (temp + 1)*0.5
    # Stack the generated data
    data.append(temp)             
  return data
    
def get_ohe_data(df):
  df_int = df.select_dtypes(['float', 'integer']).values
  continuous_columns_list = list(df.select_dtypes(['float', 'integer']).columns)
  scaler = QuantileTransformer(n_quantiles=2000, output_distribution='uniform')
  df_int = scaler.fit_transform(df_int)
  df_cat = df.select_dtypes('object')
  df_cat_names = list(df.select_dtypes('object').columns)
  numerical_array = df_int
  ohe = OneHotEncoder()
  ohe_array = ohe.fit_transform(df_cat)
  cat_lens = [i.shape[0] for i in ohe.categories_]
  discrete_columns_ordereddict = OrderedDict(zip(df_cat_names, cat_lens))
  final_array = np.hstack((numerical_array, ohe_array.toarray()))
  return df, ohe, scaler, final_array

def get_ohe_data_fair(df, S_under, Y_desire, S, Y):
  df_int = df.select_dtypes(['float', 'integer']).values
  continuous_columns_list = list(df.select_dtypes(['float', 'integer']).columns)
  scaler = QuantileTransformer(n_quantiles=2000, output_distribution='uniform')
  df_int = scaler.fit_transform(df_int)
  df_cat = df.select_dtypes('object')
  df_cat_names = list(df.select_dtypes('object').columns)
  numerical_array = df_int
  ohe = OneHotEncoder()
  ohe_array = ohe.fit_transform(df_cat)
  cat_lens = [i.shape[0] for i in ohe.categories_]
  discrete_columns_ordereddict = OrderedDict(zip(df_cat_names, cat_lens))
  S_start_index = len(continuous_columns_list) + sum(
                  list(discrete_columns_ordereddict.values())[:list(discrete_columns_ordereddict.keys()).index(S)])
  Y_start_index = len(continuous_columns_list) + sum(
                  list(discrete_columns_ordereddict.values())[:list(discrete_columns_ordereddict.keys()).index(Y)])
  if ohe.categories_[list(discrete_columns_ordereddict.keys()).index(S)][0] == S_under:
      underpriv_index = 0
      priv_index = 1
  else:
      underpriv_index = 1
      priv_index = 0
  if ohe.categories_[list(discrete_columns_ordereddict.keys()).index(Y)][0] == Y_desire:
      desire_index = 0
      undesire_index = 1
  else:
      desire_index = 1
      undesire_index = 0
  final_array = np.hstack((numerical_array, ohe_array.toarray()))
  return df, ohe, scaler, final_array, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index

def real_data_loading (args):
  assert args.df_name in ['stock','maintenance', 'vehicle']
  if args.df_name  == 'stock':
    df = pd.read_csv('data/stock.csv')
  elif args.df_name  == 'maintenance':
    df = pd.read_csv('data/maintenance.csv')
  elif args.df_name  == 'vehicle':
    df = pd.read_csv('data/vehicle.csv')
  if args.df_name  == 'vehicle':
    df_list = []
    unique_ids = set(df['Vehicle_ID'].values)
    for id in unique_ids:
      df_temp = df[df['Vehicle_ID'] == id]
      df_list.append(df_temp)
  if args.df_name  == 'maintenance':
    df_list = []
    unique_ids = set(df['Machine_ID'].values)
    for id in unique_ids:
      df_temp = df[df['Machine_ID'] == id]
      df_list.append(df_temp)
  ohes = []
  scalers = []
  datas = []
  if args.command == 'with_fairness':
    S = args.S
    Y = args.Y
    S_under = args.underprivileged_value
    Y_desire = args.desirable_value
    for ddf in df_list:
      ddf[S] = ddf[S].astype(object)
      ddf[Y] = ddf[Y].astype(object)
      ddf, ohe, scaler, df_transformed, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index = get_ohe_data_fair(ddf, S_under, Y_desire, S, Y)
      ohes.append(ohe)
      scalers.append(scaler)
      ori_data = df_transformed  
      # Flip the data to make chronological data
      ori_data = ori_data[::-1]
      # Preprocess the dataset
      temp_data = []    
      # Cut data by sequence length
      for i in range(0, len(ori_data) - args.seq_len):
        _x = ori_data[i:i + args.seq_len]
        temp_data.append(_x)     
      # Mix the datasets (to make it similar to i.i.d)
      idx = np.random.permutation(len(temp_data))    
      data = []
      for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])
      datas.append(data)
    return df_list, ohes, scalers, datas, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index
  elif args.command == 'no_fairness':
    for ddf in df_list:
      ddf, ohe, scaler, df_transformed = get_ohe_data(ddf)
      ohes.append(ohe)
      scalers.append(scaler)
      ori_data = df_transformed  
      # Flip the data to make chronological data
      ori_data = ori_data[::-1]
      # Preprocess the dataset
      temp_data = []    
      # Cut data by sequence length
      for i in range(0, len(ori_data) - args.seq_len):
        _x = ori_data[i:i + args.seq_len]
        temp_data.append(_x)     
      # Mix the datasets (to make it similar to i.i.d)
      idx = np.random.permutation(len(temp_data))    
      data = []
      for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])
      datas.append(data)
    return df_list, ohes, scalers, datas
def get_original_data(df_transformed, df_orig, ohe, scaler):
  df_transformed = np.array(df_transformed)
  df_ohe_int = df_transformed[:,:df_orig.select_dtypes(['float', 'integer']).shape[1]]
  df_ohe_int = scaler.inverse_transform(df_ohe_int)
  df_ohe_cats = df_transformed[:, df_orig.select_dtypes(['float', 'integer']).shape[1]:]
  df_ohe_cats = ohe.inverse_transform(df_ohe_cats)
  df_int = pd.DataFrame(df_ohe_int, columns=df_orig.select_dtypes(['float', 'integer']).columns)
  df_cat = pd.DataFrame(df_ohe_cats, columns=df_orig.select_dtypes('object').columns)
  return pd.concat([df_int, df_cat], axis=1)


def batch_generator(data, time, batch_size):
  """Mini-batch generator.

  Args:
    - data: time-series data
    - time: time information
    - batch_size: the number of samples in each batch

  Returns:
    - X_mb: time-series data in each batch
    - T_mb: time information in each batch
  """
  no = len(data)
  idx = np.random.permutation(no)
  train_idx = idx[:batch_size]

  X_mb = list(data[i] for i in train_idx)
  T_mb = list(time[i] for i in train_idx)

  return X_mb, T_mb