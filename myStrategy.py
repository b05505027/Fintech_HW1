import time
start = time.time()
import pandas as pd
import torch
import random
from agent import Critic

critic =  Critic(7)
critic.load_state_dict(torch.load('batch_size_256_buffer_size_20000_gamma_0.95.pt'))
critic.eval()



def myStrategy(pastPriceVec, currentPrice):
    # Convert pastPriceVec to a DataFrame
    pastPriceVec  = pastPriceVec[-26:]
    if len(pastPriceVec) < 26:
        action = random.randint(0,1)
        if action == 0:
            action = -1
        return action


    df = pd.DataFrame(pastPriceVec, columns=['Adj Close'])

    # Simple Moving Average (SMA)
    df['SMA_12'] = df['Adj Close'].rolling(window=12).mean()
    df['SMA_26'] = df['Adj Close'].rolling(window=26).mean()

    # Exponential Moving Average (EMA)
    df['EMA_12'] = df['Adj Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Adj Close'].ewm(span=26, adjust=False).mean()

    # Moving Average Convergence Divergence (MACD)
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Compute the 14-day RSI
    delta = df['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['Rolling_Mean'] = df['Adj Close'].rolling(window=20).mean()

    # Fetch the most recent data point for each indicator and convert to tensor
    recent_data = df[['Adj Close','SMA_12','SMA_26','MACD','Signal','RSI','Rolling_Mean']].iloc[-1].values
    input_tensor = torch.tensor(recent_data, dtype=torch.float32).reshape(1, 7)

    with torch.no_grad():
        q_values = critic(input_tensor)
        # Select the action with the highest Q-value
        action = torch.argmax(q_values).item()
        if action == 0:
            action = -1
    return action

# #Sample usage:
# pastPrices = [10, 11, 12, 13, 1
# , 15, 16, 15, 14, 13, 12, 11, 10, 11, 12, 13, 14,
#  15, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 10, 11, 12, 
#  13, 14, 15, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6,
#  13, 14, 15, 16, 15, 14, 13, 12, 11, 10, 9, 8, ]

# import time
# start = time.time()
# print(myStrategy(pastPrices, 7))
# end = time.time()
# print('time to run: ', end - start, 'seconds')