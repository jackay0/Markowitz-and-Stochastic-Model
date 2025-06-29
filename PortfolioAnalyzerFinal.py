import numpy as np
import pandas as pd
import yfinance as yf


class PortfolioAnalyzer:
    #initializes all of the values that we will need throughout the whole class
    def __init__(self, tickers, start_date, end_date, risk_free_rate):
        self.risk_free_rate= risk_free_rate
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.returns = self._get_stock_data()
        self._run_analysis()


    #This gets the stock data from yahoo finance and uses the pandas Datafram Structure to store it
    def _get_stock_data(self):
        data = {}
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=self.start_date, end=self.end_date)
                data[ticker] = hist['Close'].pct_change()
            except Exception as e:
                print("Error getting data")
        #uses .dropna to ensure there is no blank indices
        return pd.DataFrame(data).dropna()

    #calculates pearson standard deviation recommended in the rubric
    def _pearson_std(self, returns):
        n = len(returns)
        return np.sqrt(np.sum((returns - returns.mean())**2) / (n-1))

    #initializes statistics that utilize the pearson standard deviation
    def _initialize_parameters(self):
        
        #252 trading days
        self.mean_returns = self.returns.mean() * 252
        
        #this transposes and multiplies each vector to form a "standard deviaton" matrix,
        #then multiplies it by a correlation matrix, and then scales it to be an annual basis
        pearson_std = self.returns.apply(self._pearson_std)
        pearson_corr = self.returns.corr()
        self.cov_matrix = (pearson_std.values.reshape(-1, 1) @ pearson_std.values.reshape(1, -1)) * pearson_corr * 252
        
        self.num_assets = len(self.returns.columns)

        self.weights = np.zeros(self.num_assets)
        self.lambda_val = 0.1  # Lagrange multiplier
        self.mu_vals = np.zeros(self.num_assets)  # Multipliers for the KKT conditions (for no short selling)


    #calculates the expected return and volatility of the portfolio
    def _portfolio_performance(self, weights):
        portfolio_return = np.sum(self.mean_returns * weights)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return portfolio_std, portfolio_return
    
    #optimizes the portfolio utilizing gradient decent and lagrange multipliers to find the gradient each iteration
    def _optimize_portfolio(self):
        n = self.cov_matrix.shape[0]
    
        #initializes lambda, weights, mu, and parameters
        x = np.ones(n) / n
        lambda_val = 0.0
        mu = np.zeros(n)
        max_iter = 10000
        tolerance = 1e-6
        
        #parameters specific to gradient descent
        alpha = 0.01  # step for weights
        beta = 0.01   # step for lambda
        gamma = 0.01  # step for mu
    
        for _ in range(max_iter):
            #computes gradients 
            grad_x = 2 * (self.cov_matrix @ x) - lambda_val * np.ones(n) - mu
            grad_lambda = 1 - np.sum(x)
            grad_mu = -x
        
            #updates variables based upon gradient descent formula
            x = x - alpha * grad_x
            lambda_val = lambda_val + beta * grad_lambda
            mu = np.maximum(0, mu + gamma * grad_mu)  # mu should be non-negative
        
            #checks if it is converging
            if np.max(np.abs(grad_x)) < tolerance and np.abs(grad_lambda) < tolerance:
                break
    
        return x
    #methods for Series I Savings Bond, ETF, added;

    #getter for values for the CML
    def find_market_portfolio(self):

        optimal_weights = self._optimize_portfolio()

        #retrieves the new optimal returns and std
        market_std, market_ret = self._portfolio_performance(optimal_weights)

        #calculates sharpe
        market_sharpe = (market_ret - self.risk_free_rate) / market_std
        
        return optimal_weights, market_std, market_ret, market_sharpe

    def capital_market_line(self, std_range):

        # _ in first and fourth parameters to ignore those values
        _ , market_std, market_ret, _ = self.find_market_portfolio()

        #gets the slope of the CML based on CML formula
        slope = (market_ret - self.risk_free_rate) / market_std

        
        cml_returns = self.risk_free_rate + slope * std_range
        
        return cml_returns











    #runs all of the previous methods for analysis and prints results
    def _run_analysis(self):
        
        self._initialize_parameters()
        
        optimal_weights = self._optimize_portfolio()
        
        optimal_std, optimal_ret = self._portfolio_performance(optimal_weights)

        print("\nPortfolio Analysis Results:")
        print(f"Time period: {self.start_date} to {self.end_date}")

        print("\nAnnualized Returns for Individual Stocks:")
        for ticker in self.returns.columns:
            
            ticker_index = list(self.returns.columns).index(ticker)
            std = np.sqrt(self.cov_matrix.values[ticker_index, ticker_index])
            
            #prints each asset and its risk with return
            print(f"{ticker}:")
            print(f"  Return: {self.mean_returns[ticker]:.4f}")
            print(f"  Risk (Pearson Std): {std:.4f}")

        print("\nOptimal Portfolio Weights:")
        for ticker, weight in zip(self.returns.columns, optimal_weights):
            print(f"{ticker}: {weight:.4f}")

        print(f"\nOptimal Portfolio Performance:")
        print(f"Expected Annual Return: {optimal_ret:.4f}")
        print(f"Annual Volatility (Pearson): {optimal_std:.4f}")
        print(f"Sharpe Ratio: {(optimal_ret) / optimal_std:.4f}")

#test code
if __name__ == "__main__":
    tickers = ['AMZN', 'META', 'JNJ', 'PYPL', 'DAL', 'XOM']
    start_date = '2023-01-01'
    end_date = '2024-10-10'
    risk_free_rate = .013 #bond
    analysis = PortfolioAnalyzer(tickers, start_date, end_date, .013)





