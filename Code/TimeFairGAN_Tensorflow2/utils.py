import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import tensorflow as tf


def configure_gpu_memory_growth():
    """
    Configure TensorFlow to use GPU with memory growth. 
    If no GPU is available, it will default to CPU.
    """
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if gpu_devices:
        print('Using GPU')
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    else:
        print('Using CPU')


def get_real_data_array(hdf_store, datatype, seq_len):
    data_ = pd.read_hdf(hdf_store, datatype).sort_index()
    numerical_array = data_.select_dtypes(['float', 'integer']).values
    numeric_columns = list(data_.select_dtypes(['float', 'integer']).columns)
    scaler = MinMaxScaler()
    numerical_array = scaler.fit_transform(numerical_array)
    df_categoric = data_.select_dtypes('object')
    categorical_columns = list(data_.select_dtypes('object').columns)
    encoder = OneHotEncoder()
    ohe_array = encoder.fit_transform(df_categoric)
    final_array = np.hstack((numerical_array, ohe_array.toarray()))
    # ohe_feature_names = encoder.get_feature_names_out(categorical_columns)
    # all_columns = numeric_columns + list(ohe_feature_names)
    # final_df = pd.DataFrame(final_array, columns=all_columns)
    data = []
    for i in range(len(data_) - seq_len):
        data.append(final_array[i:i + seq_len])
    return data    

def plot_comparison_sequences(dfreal, dfsynthetic, args):
    """
    Plot real and synthetic ticker sequences.

    Parameters:
    data_preparation (object): An object containing the real data and sequence length.
    df_generated_data (DataFrame): A DataFrame containing the synthetic generated data.
    """
    # Define the ticker columns
    ticker_columns = list(dfreal.columns)
    n_tickers = len(ticker_columns)

    # Ensure all tickers are in the DataFrame
    missing_tickers = [t for t in ticker_columns if t not in dfreal.columns]
    if missing_tickers:
        raise KeyError(f"{missing_tickers} not in index")

    # Calculate number of rows and columns needed
    n_rows = int(np.ceil(n_tickers / 2))
    n_cols = 2

    # Create subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(14, n_rows * 3.5))
    axes = axes.flatten()

    # Select a random synthetic sequence from df_generated_data
    synthetic_idx = np.random.randint(len(dfsynthetic) - args.seq_len)
    synthetic_sequence = dfsynthetic.iloc[synthetic_idx: synthetic_idx + args.seq_len]

    # Select a random real sequence from data_preparation.data
    real_idx = np.random.randint(len(dfreal) - args.seq_len)
    real_sequence = dfreal.iloc[real_idx: real_idx + args.seq_len]

    # Plot each ticker column
    for j, ticker in enumerate(ticker_columns):
        if j < len(axes):
            ax = axes[j]
            real_series = real_sequence[ticker].values
            synthetic_series = synthetic_sequence[ticker].values
            
            ax.plot(real_series, label='Real', linestyle='-')
            ax.plot(synthetic_series, label='Synthetic', linestyle='--')
            ax.set_title(ticker)
            ax.legend()

    # Final plot adjustments
    sns.despine()
    fig.tight_layout()
    plt.show()
    if args.command == 'no_fairness':
        pic_name = args.fake_name + '_' + args.command + '_' + args.df_name + '_' + str(args.iteration) + '_' +'.png'
    else:
        pic_name = args.fake_name + '_' + args.command + '_' + args.df_name + '_' + str(args.iteration) + '_' + str(args.w_g_f)+'.png'
    fig.savefig(pic_name, bbox_inches='tight')

def balance_checking_real(df_name, sensitive_feature, Target, underprivileged, privileged):
    real_data = pd.read_csv('data/'+df_name+'.csv')
    sensitive_id = sensitive_feature
    Target_id = Target
    G1 = underprivileged
    G2 = privileged
    real_data[sensitive_id] = real_data[sensitive_id].map({G1: 0, G2: 1})
    real_data[Target_id] = real_data[Target_id].map({'No': 0, 'Yes': 1})
    pivot_sensitive = pd.pivot_table(real_data,values=Target_id, index = sensitive_id, aggfunc=['sum', 'count'])
    print(pivot_sensitive['sum']/pivot_sensitive['count'])
    bar_plot = (pivot_sensitive['sum']/pivot_sensitive['count']*100).plot.bar(title=f'% of {Target_id} for {G1}(0) vs {G2}(1) in '+df_name+' dataset')
    bar_plot.get_figure().savefig('Bar_plot_'+df_name+'.png')
    density_plot = sns.displot(real_data, x=sensitive_id, hue=Target_id, kind="kde")
    density_plot.set(title=f'Density plot of {sensitive_id} split by {Target_id}')
    density_plot.savefig('Density_plot_'+df_name+'.png')

def balance_checking_synthetic_no_fairness(file, df_name, sensitive_feature, Target, underprivileged, privileged):
    real_data = pd.read_csv(df_name)
    sensitive_id = sensitive_feature
    Target_id = Target
    G1 = underprivileged
    G2 = privileged
    real_data[sensitive_id] = real_data[sensitive_id].map({G1: 0, G2: 1})
    real_data[Target_id] = real_data[Target_id].map({'No': 0, 'Yes': 1})
    pivot_sensitive = pd.pivot_table(real_data,values=Target_id, index = sensitive_id, aggfunc=['sum', 'count'])
    print(pivot_sensitive['sum']/pivot_sensitive['count'])
    bar_plot = (pivot_sensitive['sum']/pivot_sensitive['count']*100).plot.bar(title=f'% of {Target_id} for {G1}(0) vs {G2}(1) in '+file+' synthetic dataset')
    bar_plot.get_figure().savefig('Bar_plot_'+df_name+'.png')
    density_plot = sns.displot(real_data, x=sensitive_id, hue=Target_id, kind="kde")
    density_plot.set(title=f'Density plot of {sensitive_id} split by {Target_id}')
    density_plot.savefig('Density_plot_'+df_name+'.png') 

def balance_checking_synthetic_with_fairness(file, df_name, sensitive_feature, Target, underprivileged, privileged):
    real_data = pd.read_csv(df_name)
    sensitive_id = sensitive_feature
    Target_id = Target
    G1 = underprivileged
    G2 = privileged
    real_data[sensitive_id] = real_data[sensitive_id].map({G1: 0, G2: 1})
    real_data[Target_id] = real_data[Target_id].map({'No': 0, 'Yes': 1})
    pivot_sensitive = pd.pivot_table(real_data,values=Target_id, index = sensitive_id, aggfunc=['sum', 'count'])
    print(pivot_sensitive['sum']/pivot_sensitive['count'])
    bar_plot = (pivot_sensitive['sum']/pivot_sensitive['count']*100).plot.bar(title=f'% of {Target_id} for {G1}(0) vs {G2}(1) in '+file+' synthetic dataset')
    bar_plot.get_figure().savefig('Bar_plot_'+df_name+'.png')
    density_plot = sns.displot(real_data, x=sensitive_id, hue=Target_id, kind="kde")
    density_plot.set(title=f'Density plot of {sensitive_id} split by {Target_id}')
    density_plot.savefig('Density_plot_'+df_name+'.png') 



