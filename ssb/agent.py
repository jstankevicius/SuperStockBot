from keras.models import load_model
from keras import optimizers
from keras.models import Sequential
import numpy as np
import random

class replay_memory:
    
    def __init__(self, memsize=300, gamma=0.9):
        self.memsize = memsize
        self.memory = []
        self.gamma = gamma

    def remember(self, states, session_over):
        self.memory.append([states, session_over])

        # Keep fixed amount of states
        if len(self.memory) > self.memsize:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]

        # maybe have something to calculate this automatically lol
        env_dim = 16

        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))

        # Perform a random sampling of states:
        for i, index in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
            state_t, action_t, reward_t, next_state = self.memory[index][0]

            session_over = self.memory[index][1]

            # Add the new state to our inputs:
            inputs[i:i+1] = state_t

            targets[i] = model.predict(state_t)[0]
            Q = np.max(model.predict(next_state)[0])

            # Once our session is over, the expected reward should be the final reward:
            # TODO: somehow pass in session_over.
            if session_over:
                targets[i, action_t] = reward_t
            else:
                #print(action_t, reward_t, self.gamma, Q)
                targets[i, action_t] = reward_t + self.gamma * Q
                
        return inputs, targets

class agent:

    def __init__(self, nn_model, *, epsilon=0.1):
        # Should we pass in a file path for nn_model to avoid any value/reference wonkiness?
        self.nn = nn_model
        self.inventory = []
        self.epsilon = epsilon

        # Some stats:
        self.buys_attempted = 0
        self.sells_attempted = 0
        self.buys_successful = 0
        self.sells_successful = 0
        self.total_actions_successful = 0
        self.loss = 0
        self.min_return = 0
        self.max_return = 0
        self.total_return  = 0
        self.wins = 0
        
        # Default memory settings (maybe change later)
        self.memory = replay_memory(650)
        
    def get_nn(self):
        return self.nn
        
    def get_stats(self):
        """Returns statistics about the last training run. reset() should be called after calling this function."""
        
        # The rewards should hopefully be consistent. If the minimum reward and maximum reward are far
        # apart, that means the agent is mostly playing a guessing game instead of playing intelligently.
        min_return = self.min_return
        max_return = self.max_return

        # Loss just tells us how well the agent is doing in estimating its future reward.
        loss = self.loss

        # Ugly and hardcoded. Similarly to activity, this gives us an idea of the model's strategy.
        # For example, if activity is high, loss is low, and average reward is low, the bot has most likely
        # adopted a short-term strategy and favors short-term rewards.
        avg_return = self.total_return / self.sells_successful

        # Success rate is important because it will tell us whether or not the NN actually understands
        # the relationship between having a share (or not) and the possible actions that it can take.
        # Ideally, when the bot has 0 shares, the value at the index for selling should be zero. In the
        # same vein, if the bot is limited to only one share and possesses it at a particular iteration,
        # the value corresponding to buying more shares should be zero.
        buy_success = self.buys_successful / self.buys_attempted
        sell_success = self.sells_successful / self.sells_attempted

        # How often does the bot make a winning trade?
        win_rate = self.wins / self.sells_successful

        # What proportion of the bot's total successful actions are spent either buying or
        # selling shares? I.E., how frequently does the bot purchase and sell shares? Does
        # it look for long-term rewards or short-term? This is mostly for performance.
        activity = (self.buys_successful + self.sells_successful) / self.total_actions_successful

        stats = {
            "loss": loss,
            "min_return": min_return,
            "max_return": max_return,
            "avg_return": avg_return,
            "buy_success": buy_success,
            "sell_success": sell_success,
            "win_rate": win_rate,
            "activity": activity
            }
        
        return stats

    def reset(self):
        """Completely resets the agent's state for the next training session."""
        self.inventory = []
        self.buys_attempted = 0
        self.sells_attempted = 0
        self.buys_successful = 0
        self.sells_successful = 0
        self.total_actions_successful = 0
        self.loss = 0
        self.min_return = 0
        self.max_return = 0
        self.total_return  = 0
        self.wins = 0
        
    def act(self, current_price, state, next_state):

        # Normalized reward value
        reward = 0

        # Dollar amount
        true_reward = 0

        # We change the placeholder value in our network to a useful value.
        num_shares = len(self.inventory)
        state[0, 15] = num_shares
        next_state[0, 15] = num_shares
        
        # Decide if we want to perform a random action (explore) or if we're
        # relying on the NN.
        action_id = 0
        action_matrix = self.nn.predict(state)

        if np.random.rand() <= self.epsilon:
            action_id = random.randint(0, 2)
        else:
            action_id = np.argmax(action_matrix)

        # BUY:
        # if action_id == 0: # unlimited shares owned
        if action_id == 0 and num_shares == 0: # limits this to one share
            self.inventory.append(current_price)
            next_state[0, 15] += 1            
            self.total_actions_successful += 1
            self.buys_successful += 1

        # SELL:
        elif action_id == 1 and num_shares > 0:
            bought_price = self.inventory.pop(0)

            # Reward is % change from last price to current price.
            percent_return = (current_price - bought_price)/bought_price*100
            reward = np.max(percent_return, 0)
            true_reward = current_price - bought_price
            
            # Remove a share from the next state.
            next_state[0, 15] -= 1

            # Update stats:
            self.sells_successful += 1
            self.total_actions_successful += 1
            self.total_return += percent_return

            if percent_return > self.max_return:
                self.max_return = percent_return

            if percent_return < self.min_return:
                self.min_return = percent_return
                
            if percent_return >= 0:
                self.wins += 1
        
        # HOLD:
        elif action_id == 2:
            self.total_actions_successful += 1

        # Update attempted actions:
        if action_id == 0:
            self.buys_attempted += 1
        elif action_id == 1:
            self.sells_attempted += 1

        # TODO: fix sess_over? Right now it's always set to False.
        self.memory.remember([state, action_id, reward, next_state], False)
        
        return reward, true_reward

    def train(self):
        inputs, targets = self.memory.get_batch(self.nn, batch_size=10)
        batch_loss = self.nn.train_on_batch(inputs, targets)
        self.loss = batch_loss
        return batch_loss
        
