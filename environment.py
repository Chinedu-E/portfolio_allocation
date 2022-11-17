from gym import Env
from portfolio import *

class StockEnv(Env):
    transaction_cost = 0.2
    
    def __init__(self, portfolio: Portfolio, data_handler: DataHandler, mode: str = "train"):
        self.portfolio = portfolio
        self.mode = mode
        self.data_handler = data_handler
        self.action_shape = (self.portfolio.num_assets,)
        self.observation_shape = [(self.portfolio.num_assets, self.portfolio.num_assets, 3),
                                  self.action_shape]
        
        self.data_handler.set_mode(mode)
        
    def step(self, action):
        state = self.get_state()
        reward = self.reward_function(action)
        done = False
        self.portfolio.tick(action)
        date = self.portfolio.current_date
        if date not in self.portfolio.dates:
            done = True

        info = self.portfolio.get_info()
        return state, reward, done, info
        
    def reset(self) -> list:
        return self.get_state()
    
    def get_state(self) -> list:
        date = self.portfolio.current_date
        state1 = np.expand_dims(self.data_handler.get_n_past(date, self.portfolio.window).values, axis=0)
        state2 = np.expand_dims(self.portfolio.performance_df.pct_change().cov().values, axis=0)
        state3 = np.expand_dims(self.portfolio.get_proportions(), axis=0)
        return [state1, state2, state3]
        
    def reward_function(self, action):
        port_weights = self.portfolio.get_proportions()
        change = np.sum(action-port_weights)
        rel_price = self.portfolio.relative_price
        reward = np.log((action*rel_price) - self.transaction_cost*change)
        return np.sum(reward)