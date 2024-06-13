from flask import Flask, request,  flash, jsonify, Response, send_from_directory
import os
import warnings
import gc
from typing import Tuple, Union, List, Dict
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('agg')

import sklearn
sklearn_version = '1.0'
# Check to make sure you have the right version of sklearn
assert sklearn.__version__  > sklearn_version, f'sklearn version is only {sklearn.__version__} and needs to be > {sklearn_version}'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from io import BytesIO, StringIO
import base64

np.set_printoptions(suppress=True)


def returnColumn(file):
    #print(file)

    # we get the data from the request body and it should be in bytes
    # next we convert the byte file into a string which is called s 
    s=str(file,'utf-8')

    #here the string is read and coverted to csv 
    data = StringIO(s) 
 
    #here we make pandas reads the csv file 
    df=pd.read_csv(data)
    # numerical_cols_df = df.select_dtypes([np.number])

    # we just want the columns 
    return df.columns.tolist() 


def run_neural_network(target_area, hidden_neurons, epochs, target_df):

    s=str(target_df,'utf-8')

    #here the string is read and coverted to csv 
    data = StringIO(s) 
 
    #here we make pandas reads the csv file 
    data_frame=pd.read_csv(data)
    
    df = data_frame
    
    # Data prep
    data = data_prep(target_area, df, target_area, return_array=True)
    X_trn, y_trn, X_vld, y_vld, X_tst, y_tst = data

    # making a neural network class
    nn = NeuralNetwork(hidden_neurons=hidden_neurons, g_hidden=Sigmoid(), g_output=Tanh(), epochs=epochs)

    # fitting the data prep
    nn.fit(X_trn, y_trn, X_vld=X_vld, y_vld=y_vld)
    vld_mse = mse(y_vld, nn.predict(X_vld))

    y_hat_trn = nn.predict(X_trn)

    trn_sse, trn_mse, trn_rmse, photo, target_array, predictions, x_values = analyze(y_trn, y_hat_trn,
                                        title="Training Predictions Log Transform",
                                        dataset="Training",
                                        xlabel="Data Sample Index",
                                        ylabel="Predicted Log Area");            
    
    y_hat_vld = nn.predict(X_vld)

    trn_sse, trn_mse, trn_rmse, photo, target_array, predictions, x_values = analyze(y_vld, y_hat_vld,
            title="Validation Predictions Log Transform",
            dataset="Validation",
            xlabel="Data Sample Index",
            ylabel="Log Area")
    
    y_hat_tst = nn.predict(X_tst)

    analyze(y_tst, y_hat_tst,
            title="Test Predictions Log Transform",
            dataset="Test",
            xlabel="Data Sample Index",
            ylabel="Log Area")

    print(trn_mse)
    print(trn_rmse)
    print(trn_sse)

    return trn_sse, trn_mse, trn_rmse, photo, target_array, predictions, x_values


# this helps clean our data by finding outliers
def outlier_locations(z, threshold):
    abs_z = np.abs(z)
    outlier_locs = np.where(abs_z > threshold)
    return outlier_locs[0]

# this helps clean our data by dropping outliers
def drop_outliers(target_area, df, threshold):
    
    numerical_cols_df = df.select_dtypes([np.number])
    numerical_cols_df = numerical_cols_df.drop([target_area], axis=1)

    z = numerical_cols_df.apply(zscore)

    outlier_locs = outlier_locations(z, threshold=threshold)

    new_df = df.drop(outlier_locs, axis=0)

    return new_df

#this is for splitting the data 
def feature_label_split(df: pd.DataFrame,
                        label_name: str) -> Tuple[pd.DataFrame]:

    X = df.drop(label_name, axis=1)
    y = df[[label_name]].copy()

    return X, y

# we are training the data 
def train_valid_test_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    seed: int = 42
) -> Tuple[pd.DataFrame]:
    
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=.2, random_state=seed)
    X_trn, X_vld, y_trn, y_vld = train_test_split(X_trn, y_trn, test_size=.2, random_state=seed)

    return X_trn, y_trn, X_vld, y_vld, X_tst, y_tst

class DataFrameColumnTransformer(TransformerMixin):
    def __init__(self, stages: List[Tuple]):
        self.col_trans = ColumnTransformer(stages, remainder='passthrough')

    def fit(self, X: pd.DataFrame):
        self.col_trans.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        output_arr = self.col_trans.transform(X)

        return self.to_dataframe(output_arr)

    def to_dataframe(self, arr: np.ndarray) -> pd.DataFrame:
        feature_names = self.col_trans.get_feature_names_out()

        for i, name in enumerate(feature_names):
            if '__' in name:
                feature_names[i] = name.split('__', 1)[-1]

        df = pd.DataFrame(arr, columns=feature_names)
        return df
    
   
def target_pipeline(y_trn, y_vld, y_tst):
    t = Pipeline([('one_hot', LogTransformer())])

    y1 = t.fit_transform(y_trn)
    y2 = t.transform(y_vld)
    y3 = t.transform(y_tst)

    return y1, y2, y3


def feature_pipeline(df: pd.DataFrame,
                     X_trn: pd.DataFrame,
                     X_vld: pd.DataFrame,
                     X_tst: pd.DataFrame) -> List[pd.DataFrame]:

    #finds all of the columns that contains strings which we dont want 
    print(df.select_dtypes([np.chararray]).columns)

    d = DataFrameColumnTransformer([('OneHotEncoding',  OneHotEncoding(), df.select_dtypes([np.chararray]).columns)])
    t = Pipeline([('DataFrameColumnTransformer', d),
                  ('Standardization', Standardization())])


    x1 = t.fit_transform(X_trn)

    x2 = t.transform(X_vld)
    x3 = t.transform(X_tst)
    return x1, x2, x3



# IMPORTANT data_prep puts everything we did upstairs together 
def data_prep(target_area,
              df: pd.DataFrame,
              label_name: str,
              *,
              seed: int = 42,
              return_array: bool = False,
              outlier_threshold: int = 5) -> Tuple[pd.DataFrame]:

    
    dropped_df = drop_outliers(target_area, df, threshold=outlier_threshold)
    
    X, y = feature_label_split(dropped_df, label_name=label_name)
    
    X_trn, y_trn, X_vld, y_vld, X_tst, y_tst = train_valid_test_split(X, y, seed)
    
    y_trn, y_vld, y_tst = target_pipeline(y_trn, y_vld, y_tst)
    
    X_trn, X_vld, X_tst = feature_pipeline(df, X_trn, X_vld, X_tst)

    
    X_trn.reset_index(inplace=True, drop=True)
    y_trn.reset_index(inplace=True, drop=True)
    X_vld.reset_index(inplace=True, drop=True)
    y_vld.reset_index(inplace=True, drop=True)
    X_tst.reset_index(inplace=True, drop=True)
    y_tst.reset_index(inplace=True, drop=True)

    
    if return_array:
        X_trn, y_trn, X_vld, y_vld, X_tst, y_tst = (X_trn.values,
                                                    y_trn.values,
                                                    X_vld.values,
                                                    y_vld.values,
                                                    X_tst.values,
                                                    y_tst.values)


    return X_trn, y_trn, X_vld, y_vld, X_tst, y_tst    



def reshape_labels(y):
    if len(y.shape) != 2:
        y = y.reshape(-1, 1)
    return y

def sse(y: np.ndarray, y_hat: np.ndarray):

    y = reshape_labels(y)
    y_hat = reshape_labels(y_hat)
    sqrd_err = (y_hat - y)**2
    sse_ = np.sum(sqrd_err)

    return sse_


def mse(y: np.ndarray, y_hat: np.ndarray):
    y = reshape_labels(y)
    y_hat = reshape_labels(y_hat)
    sqrd_err = (y_hat - y)**2

    mse_ = np.mean(sqrd_err)

    return mse_



def rmse(y: np.ndarray, y_hat: np.ndarray):
    y = reshape_labels(y)
    y_hat = reshape_labels(y_hat)
    sqrd_err = (y_hat - y)**2
    mse_ = np.mean(sqrd_err)
    rmse_ = np.sqrt(mse_)
    return rmse_    


def performance_measures(y: np.ndarray, y_hat: np.ndarray) -> Tuple[np.ndarray]:
    sse_ = sse(y=y, y_hat=y_hat)
    mse_ = mse(y=y, y_hat=y_hat)
    rmse_ = rmse(y=y, y_hat=y_hat)
    return sse_, mse_, rmse_


def analyze(
    y: np.ndarray,
    y_hat: np.ndarray,
    title: str,
    dataset: str,
    xlabel: str = None,
    ylabel: str = None
) -> Tuple[np.ndarray, float, float, float]:
    
    y = reshape_labels(y)
    y_hat = reshape_labels(y_hat)

    sse_, mse_, rmse_ = performance_measures(y=y, y_hat=y_hat)

    fig, axs = plt.subplots(1, 2, figsize=(15,7))
    fig.suptitle(title, fontsize=15)
    
    
    print(range(y.shape[0]))
    
    x = np.array(range(0, y.shape[0])).reshape((y.shape[0],1))
    
    
    axs[0].plot(x, y, 'ob', label='Target')
    axs[0].plot(x, y_hat, 'xr', label='Prediction')
    axs[0].set_xlabel("xlabel")
    axs[0].set_ylabel("ylabel")
    axs[0].legend()

    
    axs[1].plot(x, y_hat, 'xr', label='Prediction')
    axs[1].set_xlabel("xlabel")
    axs[1].set_ylabel("ylabel")
    axs[1].legend()
    
    return sse_, mse_, rmse_, fig, y, y_hat, x

def forward(
    X: np.ndarray,
    W: Dict[str, np.ndarray],
    b:  Dict[str, np.ndarray],
    g_hidden: object,
    g_output: object) -> Tuple[np.ndarray, dict, dict]:

    Zs = {}
    As = {}

    As['A0'] = X.T

    Zs['Z1'] = W['W1'] @ As['A0'] + b['b1']

    As['A1'] = Sigmoid.activation(Zs['Z1'])

    Zs['Z2'] = W['W2'] @ As['A1'] + b['b2']

    As['A2'] = Linear.activation(Zs['Z2'])
   
    y_hat = As['A2'].T

    return y_hat, Zs, As
    

class Linear():
    @staticmethod
    def activation(z):
        return z

    @staticmethod
    def derivative(z):
        return np.ones(z.shape)
    

class Sigmoid():
    @staticmethod
    def activation(z):
        return (np.exp(z) / (1+ np.exp(z)))

    @staticmethod
    def derivative(z):
        return (np.exp(z) / (1+ np.exp(z)))*( 1- (np.exp(z) / (1+ np.exp(z))))
    

class Tanh():
    @staticmethod
    def activation(z):
        return np.tanh(z)

    @staticmethod
    def derivative(z):
        return 1 - np.tanh(z)**2
    
class OneHotEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names='auto'):
        self.feature_names = feature_names
        self.encoder = OneHotEncoder(categories=feature_names, sparse_output=False)

    def fit(self, X: pd.DataFrame):

        self.encoder.fit(X)

    
        self.feature_names = self.encoder.get_feature_names_out()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        one_hot =  self.encoder.transform(X)

        return pd.DataFrame(one_hot, columns=self.get_feature_names_out())

    def get_feature_names_out(self, name=None)-> pd.Series:
        return self.feature_names
    
class Standardization(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names = None

    def fit(self, X: pd.DataFrame) -> pd.DataFrame:

        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.feature_names = X.columns
        return (X  - self.mean) / self.std

    def get_feature_names_out(self, name=None) -> pd.Series:
        return self.feature_names
    

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names = None

    def fit(self, y: pd.DataFrame):
        return self

    def transform(self, y: pd.DataFrame) -> pd.DataFrame:
        self.feature_names = y.columns

        return np.log1p(y)

    def get_feature_names_out(self, name=None) -> pd.Series:
        return self.feature_names
 


class NeuralNetwork():
    def __init__(self,
        hidden_neurons: int,
        g_hidden: object,
        g_output: object,
        output_neurons: int = 1,
        batch_size: int = 32,
        alpha: float = .01,
        epochs: int = 1
    ):
        
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.g_hidden = g_hidden
        self.g_output = g_output
        self.batch_size = batch_size
        self.alpha = alpha
        self.epochs = epochs
        self.W = None
        self.b = None
        self.epoch_losses = None
        self.vld_epoch_losses = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        X_vld: np.ndarray = None,
        y_vld:np.ndarray = None,
        seed: int = 0,
    ):
        
        np.random.seed(seed)
        m = len(X)
        self.epoch_losses = []
        self.vld_epoch_losses = []

        # Initialize weights and biases
        weights = {}
        bias = {}

        rng = np.random.RandomState(seed)

        weights['W1'] = rng.uniform(low=-0.5, high=0.5, size=(self.hidden_neurons, X.shape[1]))
        weights['W2'] = rng.uniform(low=-0.5, high=0.5, size=(self.output_neurons, self.hidden_neurons))

        bias['b1'] = np.ones([self.hidden_neurons, 1])
        bias['b2'] = np.ones([self.output_neurons, 1])

        self.W = weights
        self.b = bias

        for e in range(self.epochs):

            X_idx = np.arange(m)
            np.random.shuffle(X_idx)
            batches = [X_idx[i:i+self.batch_size] for i in range(0, m, self.batch_size)]

            epoch_sse = 0
            
            for mb in batches:

                y_hat, Zs, As = forward(X[mb], W=self.W, b=self.b, g_hidden=self.g_hidden, g_output=self.g_output)
         
                delta_mse_A2 = y_hat - y[mb]
                delta_A2_Z2 = self.g_output.derivative(Zs['Z2'])
                delta_Z2_W2 = As['A1']
                delta_Z2_b2 = np.ones([1, len(y[mb])])

                delta_mse_W2 = (delta_mse_A2.T * delta_A2_Z2) @ delta_Z2_W2.T
                W2_avg_grad = delta_mse_W2 / len(y[mb])
                delta_mse_b2 = (delta_mse_A2.T * delta_A2_Z2) @ delta_Z2_b2.T
                b2_avg_grad = delta_mse_b2 / len(y[mb])

                # ------- this part is for the hidden layer gradient 
                delta_mse_A2 = y_hat - y[mb]
                delta_A2_Z2 = self.g_output.derivative(Zs['Z2'])
                delta_Z2_A1 = self.W['W2']
                delta_A1_Z1 = self.g_hidden.derivative(Zs['Z1'])
                delta_Z1_W1 = As['A0']
                delta_Z1_b1 = np.ones([1, len(y[mb])])


            
                delta_mse_A1 = delta_Z2_A1.T @ (delta_mse_A2.T * delta_A2_Z2)
                

                delta_mse_W1 = ((delta_mse_A1 * delta_A1_Z1) @ delta_Z1_W1.T)
                W1_avg_grad = delta_mse_W1 / len(y)
                delta_mse_b1 = (delta_mse_A1 * delta_A1_Z1) @ delta_Z1_b1.T
                b1_avg_grad = delta_mse_b1 / len(y)

    

                self.W['W2'] -= self.alpha * W2_avg_grad
                self.b['b2'] -=  self.alpha * b2_avg_grad

                self.W['W1'] -= self.alpha * W1_avg_grad
                
                self.b['b1'] -= self.alpha * b1_avg_grad

                y_2 = reshape_labels(y[mb])
                y_hat_2 = reshape_labels(y_hat)
                sqrd_err = (y_hat_2 - y_2)**2
                batch_sse = np.sum(sqrd_err)


                epoch_sse += batch_sse

            epoch_mse = epoch_sse / m
            self.epoch_losses.append(epoch_mse)
           
            if X_vld is not None and y_vld is not None:
                y_hat, _, _ = forward(
                                X=X_vld,
                                W=self.W,
                                b=self.b,
                                g_hidden=self.g_hidden,
                                g_output=self.g_output,
                            )
                vld_epoch_mse = mse(X_vld, y_hat)
                self.vld_epoch_losses.append(vld_epoch_mse)

    def predict(self, X: np.ndarray):
       
        y_hat, _, _  = forward(X, W=self.W, b=self.b, g_hidden=self.g_hidden, g_output=self.g_output)

        return y_hat
    
