import random
from itertools import permutations




class AssetHandler:
    markets = []
    
    def __init__(self, num_assets: int, eval_metric: str, num_markets: int = 5):
        assert eval_metric in ["volume"]
        self.num_assets = num_assets
        self.selected_markets = random.choices(self.markets, k=num_markets)
  
    def __next__(self) -> list[str]:
        ...
        
    def load_data(self):
        ...
        
