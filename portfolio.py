from dataclasses import dataclass
from typing import Union
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from data_handler import DataHandler


@dataclass
class Portfolio:
    asset_names: list[str]
    amount: int
    weights: NDArray[np.float32]
    window: int
    train_split: float
    mode: str
    
        
    def __post_init__(self):    
        assert self.mode in ["train", "test"]
        if np.sum(self.weights) > 1:
            self.weights = tf.nn.softmax(self.weights)
            
        self.num_assets = len(self.asset_names)
        self.values = [self.amount]
        self.data_handler = DataHandler(self.asset_names)
        self.dates = self.data_handler.df.index.values
        self.train_dates, self.test_dates = self._train_test_split(self.dates)
        if self.mode == "train":
            self.dates = self.train_dates
        else:
            self.dates = self.test_dates
        
        self._cur_date_idx = self.window

        self.past_dates = [self.current_date]
        prices = self.data_handler.get_close_price(self.current_date)
        self.prices = prices
        self.n_shares = (self.weights*self.amount)/prices
        self.past_prices = [prices]
        
        
    def _train_test_split(self, dates):
        idx = int(len(dates)*self.train_split)
        train_dates = dates[:idx]
        test_dates = dates[idx:]
        return train_dates, test_dates
        
    def get_proportions(self):
        return self.weights
    
    def get_data(self, date: str):
        return self.data_handler.get_n_past(date, self.window)
    
    def set_mode(self, mode: str) -> None:
        assert mode in ["train", "test"]
        if mode == "train":
            self.dates = self.train_dates
        if mode == "test":
            self.dates = self.test_dates
        
    def update_allocation(self, weights: NDArray[np.float32]) -> None:
        date = self.current_date
        prices = self.data_handler.get_close_price(date)
        m = self.total_portfolio_value
        self.n_shares = (m*weights)/prices
        self.weights = weights
        
    def tick(self, weights: NDArray[np.float32] = None) -> None:
        if weights is not None:
            self.update_allocation(weights)
        self._cur_date_idx += 1
        if self._cur_date_idx >= len(self.dates):
            self._cur_date_idx = len(self.dates)-1
        date = self.current_date
        prices = self.data_handler.get_close_price(date)
        self.update_prices(prices)
        
        self.past_dates.append(date)
        self.values.append(self.total_portfolio_value)
        self.past_prices.append(prices)
        
    def update_prices(self, prices: NDArray[np.float32]) -> None:
        assert len(prices) == self.num_assets
        self.prices = prices
        
    def get_info(self) -> dict[str, Union[float, int]]:
        info = {}
        info["Date"] = self.current_date
        info["Profit"] = self.total_portfolio_value - self.amount
        info["Profit %"] = ((self.total_portfolio_value - self.amount)/self.amount) *100
        info["Initial value"] = self.amount
        info["Total value"] = self.total_portfolio_value
        return info
        
    def plot(self):
        plt.plot(self.past_dates, self.values)
        plt.xticks(self.past_dates[::(len(self.values)//5)])
        plt.xlabel("Dates")
        plt.ylabel("Portfolio Value")
        plt.show()
        
    @property
    def relative_price(self):
        done = False
        try:
            next_date = self.dates[self._cur_date_idx+1]
        except IndexError:
            next_date = self.dates[self._cur_date_idx]
            done = True
        next_prices = self.data_handler.get_close_price(next_date)
        curr_prices = self.prices + 0.01
        return next_prices/curr_prices, done
    
    @property    
    def performance_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.past_prices, columns=self.asset_names, index=self.past_dates)
        # ADD % change df
        # ADD Moving average
        return df
    
    @property
    def total_portfolio_value(self) -> float:
        return np.sum(self.n_shares*self.prices)
    
    @property
    def current_date(self) -> str:
        return self.dates[self._cur_date_idx]
