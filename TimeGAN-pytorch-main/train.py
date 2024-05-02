"""Reimplement TimeGAN-pytorch Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com)

-----------------------------

train.py

(1) Import data
(2) Generate synthetic data
(3) Evaluate the performances in three ways
  - Visualization (t-SNE, PCA)
  - Discriminative score
  - Predictive score
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from options import Options
from lib.data import real_data_loading, sine_data_generation, get_original_data
from lib.timegan import TimeGAN_Fair, TimeGAN
import pandas as pd


def train():
  """ Training
  """

  # ARGUMENTS
  opt_obj = Options()
  opt = opt_obj.parse()

  # LOAD DATA
  if opt.df_name in ['stock', 'maintenance', 'vehicle']:           
    if opt.command == 'no_fairness':
      df_list, ohes, scalers, datas = real_data_loading(opt)
    elif opt.command == 'with_fairness':
      df_list, ohes, scalers, datas, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index = real_data_loading(opt)
  print(opt.df_name + ' dataset is ready.')
  generated_datas = []
  l = len(df_list)
  for i in range(l):
    print(f'Training model for {i+1}th of {l} {opt.df_name}_ID\n\n\n****************************************')
    if opt.command == 'with_fairness':
      model = TimeGAN_Fair(opt, datas[i], S_start_index, Y_start_index, opt.lamda_val, underpriv_index, priv_index, desire_index)
    elif opt.command == 'no_fairness':
      model = TimeGAN(opt, datas[i])
    model.train()
    generated_data = model.generation()
    synthetic_data_reverse = []
    for row in generated_data:
      for batch in row:
        synthetic_data_reverse.append(list(batch))
    synthetic_data = synthetic_data_reverse[::-1]
    synthetic_data = synthetic_data[:opt.size_of_fake_data]
    fake_data = get_original_data(synthetic_data, df_list[i], ohes[i], scalers[i])
    generated_datas.append(fake_data)
  combined_df = pd.concat(generated_datas, axis=0)
  machine_ids = combined_df['Machine_ID'].unique()
  df_reordered = pd.DataFrame()
  machine_df = []
  for machine_id in machine_ids:
    machine_df.append(combined_df[combined_df['Machine_ID'] == machine_id])
  jump = 100
  chunks = []
  i = 0
  while i < len(machine_df[0]):
    for j in range (len(machine_df)): 
      df = machine_df[j]
      chunks.append(df[i:i+jump])
      i += jump
  df_reordered = pd.concat(chunks, ignore_index=True)
  df_reordered.to_csv('TimeFairGAN_'+opt.command+'_'+opt.fake_name+'_'+str(opt.iteration)+'.csv', index=False)

if __name__ == '__main__':
    train()
