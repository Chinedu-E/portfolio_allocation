import tensorflow as tf
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass 
class DataHandler:
    companies: list[str]
    
    def __post_init__(self):
        df = self.get_data()
        self.df = df
        self.scaled_data = ...
        train_data, test_data =self._split(df)
        self.train_data = self.windowed_dataset(train_data)
        self.test_data = self.windowed_dataset(test_data)
        self.mode = None
    
    def get_data(self):
        df = pd.DataFrame()
        labels = []
        for c in self.companies:
            labels += [c]*4
            c_df = pd.read_csv(f"data/{c}.csv")
            c_df = c_df.set_index("Date")
            df = pd.concat([c_df, df], axis = 1)
        df.columns = pd.MultiIndex.from_arrays([labels, df.columns])
        return df
    
    def _split(self, df, train_date: str = "2012-01-05",
               test_date: str = "2019-01-02") -> tuple[pd.DataFrame, pd.DataFrame]:
        train_data = df.loc[train_date:test_date]
        test_data = df.loc[test_date:]
        return train_data, test_data
    
    def set_mode(self, mode: str):
        assert mode in ["train", "test"]
        self.mode = mode
    
    def windowed_dataset(self, df, window_size=60, batch_size=32, shuffle_buffer=1000):
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
        close_price = np.zeros(len(self.companies))
        for i, company in enumerate(self.companies):
            close_price[i] = self.df.loc[date][company, "Close"]
        return close_price
    
    def get_open_price(self, date: str):
        open_price = np.zeros(len(self.companies))
        for i, company in enumerate(self.companies):
            open_price[i] = self.df.loc[date][company, "Open"]
        return open_price
    
    def get_high_price(self, date: str):
        high_price = np.zeros(len(self.companies))
        for i, company in enumerate(self.companies):
            high_price[i] = self.df.loc[date][company, "High"]
        return high_price
    
    def get_low_price(self, date: str):
        low_price = np.zeros(len(self.companies))
        for i, company in enumerate(self.companies):
            low_price[i] = self.df.loc[date][company, "Low"]
        return low_price
    
    def get_n_past(self, date: str, window: int):
        return self.df.loc[:date].iloc[-(window+1):-1]
    