import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from agent import DQNAgent
import sys
# from agent import DQNAgent



def process_data():
    # Reads the CSV file using pandas.
    df = pd.read_csv("0050.TW.csv")


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
    # drop nan
    df = df.dropna()

    return df[['Adj Close','SMA_12','SMA_26','MACD','Signal','RSI','Rolling_Mean']]





def create_dataset(df):
    # Convert DataFrame Columns to PyTorch Tensors:
    data = df.values # numpy.ndarray
    # Convert data to PyTorch tensors
    tensor_data = torch.tensor(data, dtype=torch.float32)
    # Create the TensorDataset with only features
    dataset = TensorDataset(tensor_data)
    return dataset

def create_dataloader(dataset, batch_size=1):
    # Create a DataLoader to handle batching
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader

if __name__ == '__main__':
    # parse 3 arguments: batch_size, buffer_size, gamma
    batch_size = int(sys.argv[1])
    buffer_size = int(sys.argv[2])
    gamma = float(sys.argv[3])

    session_name = f'batch_size_{batch_size}_buffer_size_{buffer_size}_gamma_{gamma}'
    


    df = process_data() # len(df) = 3802
    # Assuming df is already defined
    subset1 = df.iloc[:3000]
    subset2 = df.iloc[3071:]

    df_train = pd.concat([subset1, subset2], ignore_index=True)
    df_test = df.iloc[3000:3071]


    dataset_train = create_dataset(df_train)
    loader_train = create_dataloader(dataset_train)
    dataset_test = create_dataset(df_test)
    loader_test = create_dataloader(dataset_test)
    agent = DQNAgent(input_dim=7,dataloader_train=loader_train,dataloader_test=loader_test,
                    batch_size=batch_size, buffer_size=buffer_size, gamma=gamma)
    agent.train_dqn(episodes=100000, session_name=session_name)
    
