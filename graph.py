import numpy as np
import matplotlib.pyplot as plt
import PortfolioAnalyzerFinal

class EfficientFrontierPlotter:
    def __init__(self, portfolio_analyzer):
        self.pa = portfolio_analyzer
        self.returns = self.pa.returns
        self.mean_returns = self.pa.mean_returns
        self.cov_matrix = self.pa.cov_matrix
        self.num_assets = self.pa.num_assets

    def random_portfolios(self, num_portfolios=10000):
        results = np.zeros((3, num_portfolios))
        weights_record = []
        for i in range(num_portfolios):
            weights = np.random.random(self.num_assets)
            weights /= np.sum(weights)
            weights_record.append(weights)
            portfolio_std, portfolio_return = self.pa._portfolio_performance(weights)
            results[0, i] = portfolio_std
            results[1, i] = portfolio_return
            results[2, i] = (portfolio_return) / portfolio_std  # Sharpe Ratio
        return results, weights_record

    def plot_efficient_frontier(self):
        results, _ = self.random_portfolios()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(results[0, :], results[1, :], c= 'b', cmap = None , marker='o', s=10, alpha=0.3)
        
        ax.set_xlabel('Annualized Risk')
        ax.set_ylabel('Annualized Return')
        ax.set_title('Efficient Frontier')

         # Plotting the Capital Market Line
        x_range = np.linspace(0, max(results[0, :]), 100)
        cml = self.pa.capital_market_line(x_range)
        ax.plot(x_range, cml, 'g--', label='Capital Market Line')

        # Plotting the optimal portfolio from PortfolioAnalyzer
        opt_weights = self.pa._optimize_portfolio()
        opt_std, opt_ret = self.pa._portfolio_performance(opt_weights)
        ax.scatter(opt_std, opt_ret, marker='o', color='r', s=200, label='Optimal portfolio')
        ax.legend(loc='upper left', ncol=1, fontsize='small')
        plt.tight_layout()
        plt.show()

def plot_efficient_frontier(portfolio_analyzer):
    ef_plotter = EfficientFrontierPlotter(portfolio_analyzer)
    ef_plotter.plot_efficient_frontier()





if __name__ == "__main__":
    tickers = ['AMZN', 'META', 'JNJ', 'PYPL', 'DAL', 'XOM']
    start_date = '2023-01-01'
    end_date = '2024-10-10'
    
    analysis = PortfolioAnalyzerFinal.PortfolioAnalyzer(tickers, start_date, end_date, .013)
    
    
    plot_efficient_frontier(analysis)