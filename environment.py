from gym import Env
from portfolio import *

class StockEnv(Env):
    transaction_cost = 0.2
    
    def __init__(self, portfolio: Portfolio, mode: str):
        self.portfolio = portfolio
        self.mode = mode

        
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
        state1 = self.portfolio.get_data(date)
        state1 = self._process_state(state1)
        state2 = np.expand_dims(self.portfolio.get_data(date)["Close"].pct_change().cov().values, axis=0)
        state3 = np.expand_dims(self.portfolio.get_proportions(), axis=0)
        return [state1, state2, state3]
        
    def reward_function(self, action):
        port_weights = self.portfolio.get_proportions()
        change = np.sum(np.abs(action-port_weights))
        rel_price, done = self.portfolio.relative_price
        reward = np.log(np.dot(action, rel_price) - self.transaction_cost*change)
        return reward, done
    
    def _process_state(self, obs):
        high_normal = np.expand_dims(obs["High"]/obs["Close"].iloc[-1], axis=-1)
        low_normal = np.expand_dims(obs["Low"]/obs["Close"].iloc[-1], axis=-1)
        close_normal = np.expand_dims(obs["Close"]/obs["Close"].iloc[-1], axis=-1)
        final_state = np.concatenate([close_normal, high_normal, low_normal], axis=-1)
        final_state = np.expand_dims(final_state, axis=0)
        return final_state
        