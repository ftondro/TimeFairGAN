import argparse 
from pathlib import Path
import seaborn as sns
import numpy as np
import pandas as pd
from timegantf2 import TimeGAN
from data_processing  import DataPreparation
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.metrics import AUC
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from matplotlib.ticker import FuncFormatter
from utils import plot_comparison_sequences, configure_gpu_memory_growth, balance_checking_real, balance_checking_synthetic_no_fairness, balance_checking_synthetic_with_fairness, get_real_data_array
import random
np,random.seed(0)

def main(args):
    configure_gpu_memory_growth()
    sns.set_style('white')  
    results_path = Path('Result_TimeFairGAN')
    if not results_path.exists():
        results_path.mkdir()
    experiment = 0
    log_dir = results_path / f'experiment_{experiment:02}_itr{args.iteration}_btcs{args.batch_size}_sqlen{args.seq_len}'
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    hdf_store = results_path / 'TimeGAN.h5'    
    data_preparation = DataPreparation(args)
    prepared_data_train, prepared_data_test, real_train, real_test, args.n_seq = data_preparation.get_data()
    data_preparation.data.to_hdf(hdf_store, 'data/real')
    data_preparation.test_data.to_hdf(hdf_store, 'data/real_test')
    data_preparation.train_data.to_hdf(hdf_store, 'data/real_train')
    if args.command == 'with_fairness':
        args.S_start_index = data_preparation.S_start_index
        args.Y_start_index = data_preparation.Y_start_index
        args.underpriv_index = data_preparation.underpriv_index
        args.priv_index = data_preparation.priv_index
        args.undesire_index = data_preparation.undesire_index
        args.desire_index = data_preparation.desire_index
    balance_checking_real(args.df_name, args.S, args.Y, args.underprivileged, args.privileged)
    timegan = TimeGAN(args, args.n_seq, log_dir)
    timegan.train(prepared_data_train)
    generated_data = timegan.generate_data()
    synthetic_data_reverse = []
    for row in generated_data:
        for batch in row:
            synthetic_data_reverse.append(list(batch))
    synthetic_data = synthetic_data_reverse[::-1]
    fake_data = data_preparation.inverse_transform(synthetic_data)
    if args.command == 'no_fairness':
        fake_name = args.fake_name + '_' + args.command + '_' + args.df_name + '_' + str(args.iteration) + '.csv'
        fake_data.to_csv(fake_name, index = False)
        balance_checking_synthetic_no_fairness(args.df_name, fake_name, args.S, args.Y, args.underprivileged, args.privileged) 
    else:
        fake_name = args.fake_name + '_' + args.command + '_' + args.df_name + '_' + str(args.iteration) + '_' + str(args.w_g_f)+'.csv'
        fake_data.to_csv(fake_name, index = False)
        balance_checking_synthetic_with_fairness(args.df_name, fake_name, args.S, args.Y, args.underprivileged, args.privileged)
    print('Synthetic dataset is:\n')
    print(fake_data.head())
    plot_comparison_sequences(real_train, fake_data, args)
    with pd.HDFStore(hdf_store) as store:
        store.put('data/synthetic', fake_data)
    print(hdf_store)
    # Data loading
    real_data_array_ori = get_real_data_array(hdf_store, 'data/real_train', args.seq_len)
    synthetic_data_array= np.load(log_dir / 'generated_data.npy')
    real_data_classification = np.array(real_data_array_ori)[:len(synthetic_data_array)]
    train_data_array = np.asarray(real_data_classification)
    real_data_array = train_data_array
    test_data_array = np.array(get_real_data_array(hdf_store, 'data/real_test', args.seq_len))
    # Prepare samples
    sample_size = 250
    idx = np.random.permutation(len(real_data_array))[:sample_size]
    real_sample = np.asarray(real_data_array)[idx]
    synthetic_sample = np.asarray(synthetic_data_array)[idx]
    real_sample_2d = real_sample.reshape(-1, args.seq_len)
    synthetic_sample_2d = synthetic_sample.reshape(-1, args.seq_len)
    # Run PCA
    pca = PCA(n_components=2)
    pca.fit(real_sample_2d)
    pca_real = (pd.DataFrame(pca.transform(real_sample_2d)).assign(Data='Real'))
    pca_synthetic = (pd.DataFrame(pca.transform(synthetic_sample_2d)).assign(Data='Synthetic'))
    pca_result = pd.concat([pca_real, pca_synthetic]).rename(columns={0: '1st Component', 1: '2nd Component'})
    # Rnn t-SNE
    tsne_data = np.concatenate((real_sample_2d, synthetic_sample_2d), axis=0)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40)
    tsne_result = tsne.fit_transform(tsne_data)
    tsne_result = pd.DataFrame(tsne_result, columns=['X', 'Y']).assign(Data='Real')
    tsne_result.loc[sample_size*args.n_seq:, 'Data'] = 'Synthetic'
    # Plot results
    fig, axes = plt.subplots(ncols=2, figsize=(14, 5))
    sns.scatterplot(x='1st Component', y='2nd Component', data=pca_result, hue='Data', style='Data', ax=axes[0])
    sns.despine()
    axes[0].set_title('PCA Result')
    sns.scatterplot(x='X', y='Y', data=tsne_result, hue='Data', style='Data', ax=axes[1])
    sns.despine()
    for i in [0, 1]:
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    axes[1].set_title('t-SNE Result')
    fig.suptitle('Assessing Diversity: Qualitative Comparison of Real and Synthetic Data Distributions', fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=.88)
    plt.show()
    if args.command == 'no_fairness':
        pic_name = 't-SNE Result'+'_'+args.fake_name + '_' + args.command + '_' + args.df_name + '_' + str(args.iteration) + '.png'
    else:
        pic_name = 't-SNE Result'+'_'+args.fake_name + '_' + args.command + '_' + args.df_name + '_' + str(args.iteration) + '_' + str(args.w_g_f)+'.png'
    fig.savefig(pic_name, bbox_inches='tight')
    # Time Series Classification: A quantitative Assessment of Fidelity
    n_series_c = real_data_array.shape[0]
    idx_c = np.arange(n_series_c)
    n_train_c = int(0.9*n_series_c)
    train_idx_c = idx_c[:n_train_c]
    test_idx_c = idx_c[n_train_c:]
    train_idx_c = np.array(train_idx_c)
    test_idx_c  = np.array(test_idx_c)
    train_data_c = np.vstack((real_data_classification[train_idx_c], synthetic_data_array[train_idx_c]))
    test_data_c = np.vstack((real_data_classification[test_idx_c], synthetic_data_array[test_idx_c]))
    n_train, n_test = len(train_idx_c), len(test_idx_c)
    train_labels = np.concatenate((np.ones(n_train),np.zeros(n_train)))
    test_labels = np.concatenate((np.ones(n_test),np.zeros(n_test)))
    # Create classifier
    ts_classifier = Sequential([GRU(args.n_seq, input_shape=(args.seq_len, args.n_seq), name='GRU'), Dense(1, activation='sigmoid', name='OUT')], name='Time_Series_Classifier')
    ts_classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=[AUC(name='AUC'), 'accuracy'])
    print(ts_classifier.summary())
    result = ts_classifier.fit(x=train_data_c, y=train_labels, validation_data=(test_data_c, test_labels), epochs=args.num_epoch, batch_size=args.batch_size, verbose=0)
    print(ts_classifier.evaluate(x=test_data_c, y=test_labels))
    history = pd.DataFrame(result.history)
    print(history.info())
    # Assessing Fidelity: Time Series Classification Performance
    sns.set_style('white')
    fig, axes = plt.subplots(ncols=2, figsize=(14,4))
    history[['AUC', 'val_AUC']].rename(columns={'AUC': 'Train', 'val_AUC': 'Test'}).plot(ax=axes[1], title='ROC Area under the Curve', style=['-', '--'], xlim=(0, 250))
    history[['accuracy', 'val_accuracy']].rename(columns={'accuracy': 'Train', 'val_accuracy': 'Test'}).plot(ax=axes[0], title='Accuracy', style=['-', '--'], xlim=(0, 250))
    for i in [0, 1]:
        axes[i].set_xlabel('Epoch')
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
    axes[0].set_ylabel('Accuracy (%)')
    axes[1].set_ylabel('AUC')
    sns.despine()
    fig.suptitle('Assessing Fidelity: Time Series Classification Performance', fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=.85)
    plt.show()
    if args.command == 'no_fairness':
        pic_name = 'Performance Result'+'_'+args.fake_name + '_' + args.command + '_' + args.df_name + '_' + str(args.iteration) + '.png'
    else:
        pic_name = 'Performance Result'+'_'+args.fake_name + '_' + args.command + '_' + args.df_name + '_' + str(args.iteration) + '_' + str(args.w_g_f)+'.png'
    fig.savefig(pic_name, bbox_inches='tight')
    # Assessing usefulness
    real_train_data = real_data_array[:, :args.n_seq-1, :]
    real_train_label = real_data_array[:, -1, :]
    real_test_data = test_data_array[:, :args.n_seq-1, :]
    real_test_label = test_data_array[:, -1, :]
    fake_data_data_array = synthetic_data_array[:, :args.n_seq-1, :]
    fake_data_data_label = synthetic_data_array[:, -1, :]
    def get_model():
        model = Sequential([GRU(12, input_shape=(args.seq_len-1, args.n_seq)), Dense(args.n_seq)])
        model.compile(optimizer=Adam(), loss=MeanAbsoluteError(name='MAE'))
        return model
    ts_regression_s = get_model()
    synthetic_result = ts_regression_s.fit(x=fake_data_data_array, y=fake_data_data_label, validation_data=(real_test_data, real_test_label), epochs=args.num_epoch, batch_size=args.batch_size, verbose=0)
    ts_regression_r = get_model()
    real_result = ts_regression_r.fit(x=real_train_data, y=real_train_label, validation_data=(real_test_data, real_test_label), epochs=args.num_epoch, batch_size=args.batch_size, verbose=0)
    synthetic_result = pd.DataFrame(synthetic_result.history).rename(columns={'loss': 'Train', 'val_loss': 'Test'})
    real_result = pd.DataFrame(real_result.history).rename(columns={'loss': 'Train', 'val_loss': 'Test'})
    fig, axes = plt.subplots(ncols=2, figsize=(14, 4), sharey=True)
    synthetic_result.plot(ax=axes[0], title='Train on Synthetic, Test on Real', logy=True, xlim=(0, 100))
    real_result.plot(ax=axes[1], title='Train on Real, Test on Real', logy=True, xlim=(0, 100))
    for i in [0, 1]:
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Mean Absolute Error (log scale)')
    sns.despine()
    fig.suptitle('Assessing Usefulness: Time Series Prediction Performance', fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=.85)
    plt.show()
    if args.command == 'no_fairness':
        pic_name = 'Usefulness Result'+'_'+args.fake_name + '_' + args.command + '_' + args.df_name + '_' + str(args.iteration) + '.png'
    else:
        pic_name = 'Usefulness Result'+'_'+args.fake_name + '_' + args.command + '_' + args.df_name + '_' + str(args.iteration) + '_' + str(args.w_g_f)+'.png'
    fig.savefig(pic_name, bbox_inches='tight')
if __name__ == "__main__":
    # Inputs for the main function
    parser = argparse.ArgumentParser(description="Script Description")
    subparser = parser.add_subparsers(dest='command')
    with_fairness = subparser.add_parser('with_fairness')
    no_fairness = subparser.add_parser('no_fairness')
    # with_fairness
    with_fairness.add_argument('--df_name', type=str, help='Dataframe name', default='Robot')
    with_fairness.add_argument('--batch_size', type=int, help='The batch size', default= 128)
    with_fairness.add_argument('--seq_len', type=int, help='The sequence length', default= 24)
    with_fairness.add_argument('--hidden_dim', type=int, help='The hidden dimensions', default= 10)
    with_fairness.add_argument('--num_layer', type=int, help='Number of layers', default= 3)
    with_fairness.add_argument('--iteration', type=int, help='Number of training iterations', default=400000)
    with_fairness.add_argument('--num_epoch', type=int, help='Number of epochs', default=250)
    with_fairness.add_argument('--S', type=str, help='Protected attribute', default='Region')
    with_fairness.add_argument('--Y', type=str, help='Label (decision)', default='Failure')
    with_fairness.add_argument('--underprivileged', type=str, help='Value for underprivileged group', default='Urban')
    with_fairness.add_argument('--privileged', type=str, help='Value for privileged group', default='Rural')
    with_fairness.add_argument('--desirable_value', type=str, help='Desired label (decision)', default='Yes')
    with_fairness.add_argument('--fake_name', type=str, help='Name of the produced csv file', default='TimeFairGAN_Synthetic')
    with_fairness.add_argument('--w_gamma', type=float, default=1, help='Gamma weight')
    with_fairness.add_argument('--w_es', type=float, default=0.1, help='Encoder loss weight')
    with_fairness.add_argument('--w_e0', type=float, default=10, help='Encoder loss weight')
    with_fairness.add_argument('--w_g_s', type=float, default=100, help='Generator supervised loss weight')
    with_fairness.add_argument('--w_g_v', type=float, default=100, help='Generator moment loss weight')
    with_fairness.add_argument('--w_g_f', type=float, default=100, help='Generator fair loss weight') 
    # no_fairness
    no_fairness.add_argument('--df_name', type=str, help='Dataframe name', default='Robot')
    no_fairness.add_argument('--batch_size', type=int, help='The batch size', default= 128)
    no_fairness.add_argument('--seq_len', type=int, help='The sequence length', default= 24)
    no_fairness.add_argument('--hidden_dim', type=int, help='The hidden dimensions', default= 10)
    no_fairness.add_argument('--num_layer', type=int, help='Number of layers', default= 3)
    no_fairness.add_argument('--iteration', type=int, help='Number of training iterations', default=400000)
    no_fairness.add_argument('--num_epoch', type=int, help='Number of epochs', default=250)
    no_fairness.add_argument('--S', type=str, help='Protected attribute', default='Region')
    no_fairness.add_argument('--Y', type=str, help='Label (decision)', default='Failure')
    no_fairness.add_argument('--underprivileged', type=str, help='Value for underprivileged group', default='Urban')
    no_fairness.add_argument('--privileged', type=str, help='Value for privileged group', default='Rural')
    no_fairness.add_argument('--fake_name', type=str, help='Name of the produced csv file', default='TimeFairGAN_Synthetic')
    no_fairness.add_argument('--w_gamma', type=float, default=1, help='Gamma weight')
    no_fairness.add_argument('--w_es', type=float, default=0.1, help='Encoder loss weight')
    no_fairness.add_argument('--w_e0', type=float, default=10, help='Encoder loss weight')
    no_fairness.add_argument('--w_g_s', type=float, default=100, help='Generator supervised loss weight')
    no_fairness.add_argument('--w_g_v', type=float, default=100, help='Generator moment loss weight')
    args = parser.parse_args()
    main(args)