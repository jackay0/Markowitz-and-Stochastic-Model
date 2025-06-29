import yfinance as yf
import numpy as np
from typing import List, Dict

class FinancialSecuritiesAnalysis:
    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
    
    def get_stock_data(self, ticker: str, start_date: str, end_date: str):
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        return df['Adj Close']
    
    def calculate_log_returns(self):
        log_returns = {}
        for ticker in self.tickers:
            prices = self.get_stock_data(ticker, self.start_date, self.end_date)
            log_returns[ticker] = np.log(prices / prices.shift(1)).dropna().to_numpy()
        return log_returns
    
    def estimate_drift(self, log_returns: np.ndarray) -> float:
        return np.mean(log_returns)
    
    def estimate_volatility(self, log_returns: np.ndarray, drift: float = None) -> float:
        if drift is None:
            drift = self.estimate_drift(log_returns)
        return np.mean((log_returns - drift)**2)
    
    def estimate_covolatility(self, ticker1: str, ticker2: str) -> float:
        log_returns_dict = self.calculate_log_returns()
        log_returns1 = log_returns_dict[ticker1]
        log_returns2 = log_returns_dict[ticker2]
        
        min_length = min(len(log_returns1), len(log_returns2))
        log_returns1 = log_returns1[:min_length]
        log_returns2 = log_returns2[:min_length]
        
        drift1 = self.estimate_drift(log_returns1)
        drift2 = self.estimate_drift(log_returns2)
        
        covolatility = np.mean((log_returns1 - drift1) * (log_returns2 - drift2))
        return covolatility
    
    def analyze_securities(self) -> dict:
        results = {}
        log_returns_dict = self.calculate_log_returns()
        
        for ticker in self.tickers:
            log_returns = log_returns_dict[ticker]
            drift = self.estimate_drift(log_returns)
            volatility = self.estimate_volatility(log_returns, drift)
            
            results[ticker] = {
                'drift': drift,
                'volatility': volatility
            }
        
        covolatility_matrix = {}
        for i in range(len(self.tickers)):
            for j in range(i+1, len(self.tickers)):
                ticker1 = self.tickers[i]
                ticker2 = self.tickers[j]
                covolatility = self.estimate_covolatility(ticker1, ticker2)
                covolatility_matrix[f'{ticker1}-{ticker2}'] = covolatility
        
        results['covolatility_matrix'] = covolatility_matrix
        return results