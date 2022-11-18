import random


class AssetHandler:
    markets = []
    
    def __init__(self, num_assets: int, num_markets: int = 5):
        self.num_assets = num_assets

        self.all_companies = self.get_companies()
        random.shuffle(self.all_companies)

        self.p1 = -self.num_assets
        self.p2 = 0
  
    def __next__(self) -> list[str]:
        self.p1 += self.num_assets
        self.p2 += self.num_assets
        return self.all_companies[self.p1: self.p2]
        
    def load_data(self):
        ...
        
    def get_companies(self, n=500):
        with open("companies.txt", "r") as f:
            companies = f.read().split("\n")
        companies = [company for company in companies if len(company)>1]
        return companies[:n]