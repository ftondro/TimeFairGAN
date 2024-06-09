import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from collections import OrderedDict
from sklearn.model_selection import train_test_split

class DataPreparation:
    def __init__(self, args):
        self.data_path = args.df_name
        self.seq_len = args.seq_len
        self.command = args.command
        if self.command == 'with_fairness':
            self.S = args.S
            self.Y = args.Y
            self.S_under = args.underprivileged
            self.Y_desire = args.desirable_value
            self.S_start_index = None
            self.Y_start_index = None
            self.underpriv_index = None
            self.priv_index = None
            self.undesire_index = None
            self.desire_index = None
        self.n_seq = 0 
        self.scaler = MinMaxScaler()
        self.encoder = None  # Initialize encoder
        self.categorical_columns = []
        self.encoded_categorical_columns = []
        self.numeric_columns = []
        self.n_windows = 0

    def load_data(self):
        # Load the data from a CSV or any other format
        self.data = pd.read_csv('data/'+self.data_path+'.csv')
        train_data, test_data = train_test_split(self.data, test_size=0.2, random_state=42)
        self.train_data = pd.DataFrame(train_data, columns=self.data.columns)
        self.test_data = pd.DataFrame(test_data, columns=self.data.columns)
        print('Real dataset is:\n')
        print(self.data.head())
        print("Initial columns:", self.data.columns)
        if self.command == 'with_fairness':
            self.data[self.S] = self.data[self.S].astype(object)
            self.data[self.Y] = self.data[self.Y].astype(object)       
        # Identify numeric columns
        self.numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = self.data.select_dtypes(exclude=[np.number]).columns.tolist()
        print("Numeric columns:", self.numeric_columns)
        print("Categorical columns:", self.categorical_columns)

    def get_ohe_data(self, data_):
        numerical_array = data_.select_dtypes(['float', 'integer']).values
        self.numeric_columns = list(data_.select_dtypes(['float', 'integer']).columns)
        numerical_array = self.scaler.fit_transform(numerical_array)
        df_categoric = data_.select_dtypes('object')
        self.categorical_columns = list(data_.select_dtypes('object').columns)
        self.encoder = OneHotEncoder()
        ohe_array = self.encoder.fit_transform(df_categoric)
        cat_lens = [i.shape[0] for i in self.encoder.categories_]
        self.encoded_categorical_columns = OrderedDict(zip(self.categorical_columns, cat_lens))
        if self.command == 'with_fairness':
            self.S_start_index = len(self.numeric_columns) + sum(
                            list(self.encoded_categorical_columns.values())[:list(self.encoded_categorical_columns.keys()).index(self.S)])
            self.Y_start_index = len(self.numeric_columns) + sum(
                            list(self.encoded_categorical_columns.values())[:list(self.encoded_categorical_columns.keys()).index(self.Y)])
            if self.encoder.categories_[list(self.encoded_categorical_columns.keys()).index(self.S)][0] == self.S_under:
                self.underpriv_index = 0
                self.priv_index = 1
            else:
                self.underpriv_index = 1
                self.priv_index = 0
            if self.encoder.categories_[list(self.encoded_categorical_columns.keys()).index(self.Y)][0] == self.Y_desire:
                self.desire_index = 0
                self.undesire_index = 1
            else:
                self.desire_index = 1
                self.undesire_index = 0
        final_array = np.hstack((numerical_array, ohe_array.toarray()))
        ohe_feature_names = self.encoder.get_feature_names_out(self.categorical_columns)
        all_columns = self.numeric_columns + list(ohe_feature_names)
        final_df = pd.DataFrame(final_array, columns=all_columns)
        self.n_seq = len(final_df.columns)
        print("Columns after encoding:", final_df.columns)
        return final_df

    def inverse_transform(self, data_):
        df_transformed = np.array(data_)
        df_numeric = df_transformed[:,:self.data.select_dtypes(['float', 'integer']).shape[1]]
        df_numeric_inverse = self.scaler.inverse_transform(df_numeric)
        df_int = pd.DataFrame(df_numeric_inverse, columns=self.data.select_dtypes(['float', 'integer']).columns)
        if self.categorical_columns:
            df_ohe_categoric = df_transformed[:, self.data.select_dtypes(['float', 'integer']).shape[1]:]
            df_ohe_categoric_inverse = self.encoder.inverse_transform(df_ohe_categoric)   
            df_cat = pd.DataFrame(df_ohe_categoric_inverse, columns=self.data.select_dtypes('object').columns)
            df_inverse = pd.concat([df_int, df_cat], axis=1)
            df_inverse = df_inverse[self.data.columns]
            return df_inverse
        else:
            df_int = df_int[self.data.columns]
            return df_int
        
    def create_sequences(self, data_):
        # Create sequences for TimeGAN
        data_sequences = []
        for i in range(len(data_) - self.seq_len):
            data_sequences.append(data_.iloc[i:i + self.seq_len].values)
        data_sequences = np.array(data_sequences)
        self.n_windows = len(data_sequences)
        print(f"Number of windows: {self.n_windows}")
        return data_sequences
    
    def get_data(self):
        self.load_data()
        data_train = self.get_ohe_data(self.train_data)
        data_test = self.get_ohe_data(self.test_data)
        print('Encoded real train dataset is:\n')
        print(data_train.head())
        data_train_ = self.create_sequences(data_train)
        data_test_ =  self.create_sequences(data_test) 
        return data_train_, data_test_, self.train_data, self.test_data, self.n_seq



