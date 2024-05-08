import numpy as np

class PortfolioCritic:
    def __init__(self, num_stocks):
        self.num_stocks = num_stocks
        self.weights = np.ones(num_stocks) / num_stocks  # setting equal weights for all stocks initally
        self.alpha = 0.1  # Learning rate 
        self.gamma = 0.9  # Discount factor 

        
        self.Q = np.zeros((num_stocks, 3)) # Initially the Q function is set to 0
         # 3 actions: buy, sell, hold

    def update_weights(self, action, returns):
        self.weights += returns / np.sum(returns) # Updating weights
        state = np.argmax(self.weights) # Updating Q function
        if action == 'buy':
            next_state = state + 1
        elif action == 'sell':
            next_state = state - 1
        else:
            next_state = state
        reward = returns[state]
        self.Q[state, :] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state, :]) - self.Q[state, :])

    def get_portfolio(self):
        return self.weights

    def get_policy(self):
        return np.argmax(self.Q, axis=1)

def simulate_trading(critic, num_iterations, stock_returns):
    for i in range(num_iterations): 
        portfolio = critic.get_portfolio() # Fetching portfolio weights from the Critic function 
        policy = critic.get_policy()  # Fetching policy from the Critic Function 
        action = [] # Executing the policy
        for p in policy:
            if p == 0:
                action.append('buy')
            elif p == 1:
                action.append('sell')
            else:
                action.append('hold')
        returns = np.dot(portfolio, stock_returns[i]) # Updating the Critic function with action and returns
        critic.update_weights(action, returns)

def main():
    # Setting Parameters
    num_stocks = 3  
    num_iterations = 100  
    stock_returns = np.random.randn(num_iterations, num_stocks)  # Random returns for demonstration
    critic = PortfolioCritic(num_stocks) # Initializing critic
    simulate_trading(critic, num_iterations, stock_returns)
    final_portfolio = critic.get_portfolio() # Obtaining the final weights from the critic
    print("Final Portfolio Weights:", final_portfolio)

if __name__ == "__main__":
    main()
