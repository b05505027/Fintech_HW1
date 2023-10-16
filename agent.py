import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from itertools import islice
class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.ln1 = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, 32)
        self.ln2 = nn.LayerNorm(32)
        self.fc3 = nn.Linear(32, 2)
    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x
  
import torch

class StockTradingEnv:
    def __init__(self, dataloader, initial_capital=1000):
        self.dataloader = dataloader
        self.initial_capital = initial_capital  # Starting capital
        self.capital = self.initial_capital  # Current available capital
        self.stock_price = 0  # Current stock price
        self.stocks_held = 0  # Number of stocks currently held

    def reset(self):
        """Resets the environment to its initial state and returns the initial state."""
        
        self.capital = self.initial_capital  # Reset capital
        self.stocks_held = 0  # Reset stocks held
        length = len(self.dataloader)
        starting = np.random.randint(0, length-70)
        #self.iterator = iter(self.dataloader)  # Reset the dataloader iteratorc
        self.iterator = iter(islice(self.dataloader, starting, starting + 70))  # Reset the dataloader iteratorc
        initial_state = next(self.iterator)[0]  # Get the initial state
        self.stock_price = initial_state[0][0].item()  # Set the initial stock price
        return initial_state # Return the initial state as a numpy array

    def step(self, action):
        """Takes an action (0 or 1) and returns the next state, reward, and whether the episode is done."""
        
        prev_capital = self.capital  # Store previous capital to calculate reward
        prev_stock_price = self.stock_price  # Store previous stock price to calculate reward

        # Buy stocks
        if action == 1 and self.capital >= self.stock_price:
            self.stocks_held = self.capital / self.stock_price  # Spend all capital to buy stocks
            self.capital = 0  # Update capital after buying

        # Sell stocks
        elif action == 0 and self.stocks_held > 0:
            self.capital += self.stocks_held * self.stock_price  # Sell all stocks and update capital
            self.stocks_held = 0  # Reset stocks held after selling
        
        next_state = None
        done = False  # Flag to check if the episode is over

        try:
            # Try to get the next state from the dataloader
            next_state = next(self.iterator)[0]
            self.stock_price = next_state[0][0].item()  # Update the stock price
        except StopIteration:
            # If StopIteration is raised, we've reached the end of the dataloader
            done = True
            # If stocks are held at the end of the episode, sell them
            if self.stocks_held > 0:
                self.capital += self.stocks_held * self.stock_price
                self.stocks_held = 0
            
            increment = self.capital - prev_capital
            
            return None, 0 ,done, increment  # Return the next state, reward, and whether the episode is done
        
        # Calculate reward as the change in capital
        # print('prev_capital', prev_capital, 'capital', self.capital)

        increment = self.capital - prev_capital

        reward1 = 0.05* self.stocks_held * (self.stock_price - prev_stock_price)
        reward2 = 10*(self.stocks_held >0)*np.log(self.stock_price/prev_stock_price + 0.0001)

        reward = reward1 + reward2         
        # print('reward', reward)
        # print('current stock held', self.stocks_held)
        # print('stock price increment', self.stock_price - prev_stock_price)
        # input()

        return next_state, reward, done, increment  # Return the next state, reward, and whether the episode is done

class DQNAgent:
    def __init__(self, input_dim, dataloader_train, dataloader_test, batch_size=128, buffer_size=2000, gamma=0.995):
        self.critic = Critic(input_dim)
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.initial_capital = 1000
        self.criterion = nn.MSELoss()
        self.env = StockTradingEnv(dataloader_train, initial_capital=self.initial_capital)
        self.test_env = StockTradingEnv(dataloader_test, initial_capital=self.initial_capital)
        self.buffer = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = 0.995
        
    
    def get_action(self, state):
        # Get the Q-value for each action
        q_values = self.critic(state)
        # print('q_values', q_values)
        
        # Select the action with the highest Q-value
        action = torch.argmax(q_values).item()
        # print('action', action)
        # input()
        return action
    
    def train(self, state, action, target_reward):
        
        # here action is a form of [[0],[1],[0]]
        # and critic(state) is a form of [[14.2449,  2.1441],[18.9334,  2.2573],[35.9309,  2.1355]]
        # now we need to get the corresponding reward for each action
        # for example, if action is [[0],[1],[0]], then we need to get [[14.2449],[2.2573],[35.9309]]
        loss = self.criterion(self.critic(state).gather(1, action), target_reward)
        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping
        for param in self.critic.parameters():
            param.grad.data.clamp_(-5, 5)
        self.optimizer.step()
        return loss.item()
    
    def train_dqn(self, episodes, session_name='default'):
        test_returns = []
        losses = []
        train_returns = []
        exploration_rate = 1

        for episode in tqdm(range(episodes)):
            state = self.env.reset()
            total_reward = 0
            done = False
            steps = 0
            while not done:
                if np.random.rand() < exploration_rate:
                    # Select a random action
                    action = np.random.randint(2)
                else:
                    action = self.get_action(state/1000)
                next_state, reward, done , increment = self.env.step(action)
                
                if next_state is not None:
                    # Update the Q-value using Bellman equation
                    target_reward = reward + self.gamma * torch.max(self.critic(next_state))
                    target_reward = target_reward.reshape(1, -1)
                else:
                    target_reward = torch.tensor([[reward]], dtype=torch.float32)

                # self.train(state, action, target_reward)
                # stor the transition in buffer
                self.buffer.append((state.detach(), action, target_reward.detach()))
                if len(self.buffer) > (self.buffer_size * 70):
                    # drop the oldest transition
                    self.buffer.pop(0)
                state = next_state
                total_reward += reward
                steps += 1

            # sample a batch of random batch from buffer
            if len(self.buffer) < self.batch_size:
                batch_size = len(self.buffer)
            else:
                batch_size = self.batch_size
            batch = np.random.choice(len(self.buffer), batch_size, replace=False)
            states = []
            actions = []
            rewards = []
            for i in batch:
                states.append(self.buffer[i][0])
                actions.append(self.buffer[i][1])
                rewards.append(self.buffer[i][2])
            states = torch.cat(states)
            actions = torch.tensor(actions, dtype=torch.int64).reshape(-1,1)
            rewards = torch.cat(rewards)

            loss = self.train(states, actions, rewards)
            losses.append(loss)

            exploration_rate *= 0.9999
            exploration_rate = max(0.01, exploration_rate)
            
            train_returns.append(total_reward)

            # print(f"Episode {episode+1}/{episodes} - Total Reward: {total_reward}")

            if (episode + 1) % 200 == 0:
                
                test_return = self.test_dqn()

                # print(f"Test Return: {test_return}")
                # print('exploration_rate', exploration_rate)

                test_return=(test_return)/self.initial_capital
                test_returns.append(test_return)

                if test_return == max(test_returns):
                    torch.save(self.critic.state_dict(), f'{session_name}.pt')

            if (episode + 1) % 1000 == 0:
                # plot loss, testing return and training return all in one figure and different subplots
                fig, axs = plt.subplots(3)
                fig.set_size_inches(24, 20)
                fig.suptitle('Loss, Testing Return and Training Return')
                # set text size = 14
                plt.rcParams.update({'font.size': 14})
                axs[0].plot(np.convolve(losses, np.ones(1000)/1000, mode='valid'), color = '#2a9d8f', linewidth=4)
                axs[0].set_title('Training Loss')
                axs[0].yaxis.set_label_text('Loss')
                axs[0].xaxis.set_label_text('Episode')
                # also plot the average line (the most recent 1000 data) with value labeled
                axs[0].axhline(y=np.mean(losses[-1000:]), color='#2a9d8f', linestyle='-', label='Average Loss')
                # write down the value
                axs[0].text(0, np.mean(losses[-1000:]), f'{np.mean(losses[-1000:]):.4f}', fontsize=18)

                axs[1].plot(test_returns, color='#e76f51', linewidth=4)
                axs[1].set_title('Testing ROI')
                axs[1].yaxis.set_label_text('ROI')
                axs[1].xaxis.set_label_text('Episode')
                # also plot the average line (the most recent 1000 data) with value labeled
                axs[1].axhline(y=np.mean(test_returns[-1000:]), color='#e76f51', linestyle='-', label='Average ROI')
                # write down the value
                axs[1].text(0, np.mean(test_returns[-1000:]), f'{np.mean(test_returns[-1000:]):.4f}', fontsize=18)

                axs[2].plot(np.convolve(train_returns, np.ones(1000)/1000, mode='valid'),  color = '#e9c46a', linewidth=4)
                axs[2].set_title('Training Return')
                axs[2].yaxis.set_label_text('Return')
                axs[2].xaxis.set_label_text('Episode')
                # also plot the average line (the most recent 1000 data) with value labeled
                axs[2].axhline(y=np.mean(train_returns[-1000:]), color='#e9c46a', linestyle='-', label='Average Return')
                # write down the value
                axs[2].text(0, np.mean(train_returns[-1000:]), f'{np.mean(train_returns[-1000:]):.4f}', fontsize=18)

                plt.savefig(f'{session_name}.png')
                plt.close()

                

    def test_dqn(self):
        state = self.test_env.reset()
        done = False
        total_increment = 0
        while not done:
            action = self.get_action(state)
            #cprint('action', action)
            next_state, reward, done, increment = self.test_env.step(action)
            state = next_state
            total_increment += increment
            # print('total_reward', total_reward)
        return total_increment