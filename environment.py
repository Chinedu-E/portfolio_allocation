from gym import Env
from portfolio import *

class StockEnv(Env):
    transaction_cost = 0.2
    
    def __init__(self, portfolio: Portfolio, mode: str):
        self.portfolio = portfolio
        self.mode = mode
        self.action_shape = (self.portfolio.num_assets,)
        self.observation_shape = [(self.portfolio.num_assets, self.portfolio.num_assets, 3),
                                  self.action_shape]
        
        
    def step(self, action):
        state = self.get_state()
        reward, done = self.reward_function(action)
        self.portfolio.tick(action)
        info = self.portfolio.get_info()
        return state, reward, done, info
        
    def reset(self) -> list:
        return self.get_state()
    
    def get_state(self) -> list:
        date = self.portfolio.current_date
        state1 = np.expand_dims(self.portfolio.get_data(date).values, axis=0)
        state2 = np.expand_dims(self.portfolio.get_data(date).pct_change().cov().values, axis=0)
        state3 = np.expand_dims(self.portfolio.get_proportions(), axis=0)
        return [state1, state2, state3]
        
    def reward_function(self, action):
        port_weights = self.portfolio.get_proportions()
        change = np.sum(np.abs(action-port_weights))
        rel_price, done = self.portfolio.relative_price
        reward = np.log(np.dot(action, rel_price) - self.transaction_cost*change)
        return reward, done