import tensorflow as tf
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pandas_datareader import data as pdr


@dataclass 
class DataHandler:
    companies: list[str]
    start: str
    end: str
    test_start: str
    
    def __post_init__(self):
        df = self.get_data()
        self.df = df
        self.scaled_data = ...
        train_data, test_data =self._split(df)
        self.train_data = train_data
        self.test_data = test_data
        self.mode = None
    
    def get_data(self):
        df = pdr.get_data_yahoo(self.companies, self.start, self.end)
        return df
    
    def _split(self, df) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_data = df.loc[self.start:self.test_start]
        test_data = df.loc[self.test_start:]
        return train_data, test_data
    
    def set_mode(self, mode: str):
        assert mode in ["train", "test"]
        self.mode = mode
    
    def windowed_dataset(self, df, window_size=30, batch_size=32, shuffle_buffer=1000):
        self.window_size = window_size
        self.batch_size = batch_size
        # Create dataset from the series
        dataset = tf.data.Dataset.from_tensor_slices(df.values)
        # Slice the dataset into the appropriate windows
        dataset = dataset.window(window_size, shift=1, drop_remainder=True)
        # Flatten the dataset
        dataset = dataset = dataset.flat_map(lambda window: window.batch(window_size))
        # Shuffle it
        dataset = dataset.shuffle(shuffle_buffer)
        # Split it into the features and labels
        #dataset = dataset = dataset.map(lambda window: (window, window))
        # Batch it
        dataset = dataset.batch(batch_size).prefetch(1)

        return dataset
    
    def __next__(self):
        if self.mode:
            if self.mode == "train":
                data = self.train_data.as_numpy_iterator().next()
            else:
                data = self.test_data.as_numpy_iterator().next()
            data = data.reshape(self.batch_size, self.window_size, len(self.companies), 4)
            return data
    
    def get_close_price(self, date: str):
        return self.df["Adj Close"].loc[date]
    
    def get_open_price(self, date: str):
        return self.df["Open"].loc[date]
    
    def get_high_price(self, date: str):
        return self.df["High"].loc[date]
    
    def get_low_price(self, date: str):
        return self.df["Low"].loc[date]
    
    def get_n_past(self, date: str, window: int):
        df = self.df.loc[:date].iloc[-(window+1):-1]
        return df
    