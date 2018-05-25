from behaviors import classical_brownian, geometric_brownian, merton, heston
from agent import agent
from keras.models import load_model
from keras.optimizers import sgd
from keras.models import Sequential
from keras.layers import Dense
import h5py
import numpy as np
import finance as f
import csv

# Dictionary containing every agent's name and the corresponding object that
# gets instantiated.
agent_config = {
    "classical_brownian": classical_brownian,
    "geometric_brownian": geometric_brownian,
    "merton_jump_diffusion": merton,
    "heston_stochastic": heston
    }

# What periods are we looking at?
periods = [5, 8, 13]

# We define what inputs we want here:
input_config = (
    (f.roc, periods),
    (f.rsi, periods),
    (f.sma, periods),
    (f.wma, periods),
    (f.ema, periods)
)

def get_symbols():
    """Returns all stock symbols in data//symbols.txt as a list."""
    symbol_file = open("data//symbols.txt", "r")
    symbols = symbol_file.readlines()

    corrected = []

    # Correcting for newline delimiters.
    for symbol in symbols:
        corrected.append(symbol.rstrip())

    return corrected

def get_inputs(prices):
    # We append cut and formatted lists here:
    input_lists = []
    
    # GET INDICATORS
    for indicator_tuple in input_config:
        indicator = indicator_tuple[0]
        period_list = indicator_tuple[1]

        for period in period_list:
            indicator_list = indicator(prices, period)
            input_lists.append(indicator_list)

    # After this, we begin to cut all the arrays down to size.
    # max_length is simply an integer defining the maximum size (and therefore the
    # actual size) of each array. The maximum size of ALL arrays needs to be the
    # size of the smallest array so that all elements "line up."
    max_length = len(prices[periods[len(periods) - 1]:])

    # We resize everything in input_lists:
    for input_list in input_lists:
        input_list = input_list[len(input_list) - max_length:]

    # We cut the price list:
    prices = prices[len(prices) - max_length:]

    # This is NOT input_lists! "inputs" contains the actual input matrices that are
    # passed to the agent.
    inputs = []
    
    # Iterate through all input lists, take element at i, and create a numpy array:
    for i in range(max_length):
        a = []
        for input_list in input_lists:
            a.append(input_list[i])

        # This is a placeholder for the number of shares the agent owns.
        a.append(0)
        
        # Now, convert to a numpy array and reshape it so it is acceptable as input.
        npa = np.array(a)
        npa = npa.reshape(1, 16)

        inputs.append(npa)

    return inputs

class training_session:
    
    def __init__(self):

        # "name": behavior instance
        self.agents = {}

        # Internal episode counter.
        self.epcount = 1
        
        # Go through agent_config and create an agent with the corresponding model.
        for key in agent_config.keys():
            
            # Create NN:
            # This is a bad model. I just want to see if instantiating this works.
            # I should also probably clean this up. A lot.
            nn = Sequential()
            nn.add(Dense(20, input_shape=(16,), activation="relu"))
            nn.add(Dense(20, activation="tanh"))
            nn.add(Dense(20, activation="tanh"))
            nn.add(Dense(3, activation="linear"))
            nn.compile(sgd(lr=0.01), "mse")
            
            agent_instance = agent(nn)

            # Warm up the agent and immediately reset:
            # TODO: maybe fix act()? During training we might not necessarily want
            # to act on an actual stock.
            agent_instance.act(1, np.ones((1, 16)), np.ones((1, 16)))
            agent_instance.reset()
            
            behavior_instance = agent_config[key]()
            self.agents[key] = (agent_instance, behavior_instance)

    # Use this for benchmarking.
    def save_agents(self, rewards, *, prefix=""):
        for agent in self.agents.keys():
            agent_instance = self.agents[agent][0]
            agent_instance.get_nn().save("models//" + agent + "//" + prefix + agent + "{reward: " + str(round(rewards[agent], 2)) + "}.h5")
            
    def get_agents(self):
        return self.agents
        
    def run(self):
        """Performs a single simulation of all behaviors and has agents act correspondingly."""
        
        for key in self.agents.keys():
            agent_instance = self.agents[key][0]
            behavior_instance = self.agents[key][1]
            
            prices = behavior_instance.step(690)
            inputs = get_inputs(prices)
            
            # Now, we iterate through all the inputs and make the model act on the input and
            # receive information about the next state. Internally, the agent also updates
            # the number of shares it has as one of the inputs to the NN.
            for i in range(len(inputs) - 1):
                state = inputs[i]
                next_state = inputs[i + 1]
                agent_instance.act(prices[i], state, next_state)

            # Get batch loss and add that to the string too.
            batch_loss = agent_instance.train()
        
        # Update episode count.
        self.epcount += 1

    def get_stats(self):
        stats = {}
        for key in self.agents.keys():
            agent_instance = self.agents[key][0]
            stats[key] = agent_instance.get_stats()

        return stats

    def reset(self):
        for key in self.agents.keys():
            agent_instance = self.agents[key][0]
            agent_instance.reset()

    def test(self, *, symbol_list=get_symbols()):
        """Iterates through the test dataset and collects the rewards of each agent."""
        
        symbols = symbol_list
        total_rewards = {}

        print("Checking files...")
        # Test pass:
        for symbol in symbols:
            file = open("data//intraday//" + symbol + ".csv", "r")

        print("File check passed")

        # Header:
        header = ""

        # We create a list to keep consistent iteration order.
        agent_list = []

        for agent in self.agents.keys():
            total_rewards[agent] = 0
            header += "\t" + agent
            agent_list.append(agent)

        print(header)
        for symbol in symbols:
            line = symbol + "\t"
            prices = []
            
            with open("data//intraday//" + symbol + ".csv", "r") as file:
                reader = csv.reader(file)
                next(file)

                for row in reader:
                    prices.append(float(row[5]))

            prices = np.array(prices)
            inputs = get_inputs(prices)

            # REALLY GROSS
            agent_reward_list = [0, 0, 0, 0]

            for i in range(len(inputs) - 1):
                for k in range(len(agent_list)):
                    agent = agent_list[k]
                    agent_instance = self.agents[agent][0]
                    
                    state = inputs[i]
                    next_state = inputs[i + 1]
                    
                    reward, true_reward = agent_instance.act(prices[i], state, next_state)
                    total_rewards[agent] += true_reward
                    agent_reward_list[k] += true_reward
                    
            for i in range(len(agent_reward_list)):
                line += str(round(agent_reward_list[i], 2)).rjust(len(agent_list[i])) + "\t"
                
            print(line)
                
            for agent in self.agents.keys():
                self.agents[agent][0].reset()

        # Print the total rewards
        for agent in self.agents.keys():
            print(agent + ":\t" + str(round(total_rewards[agent], 2)))

        return total_rewards

tr = training_session()
print("FIRST FULL TEST:")
first_rewards = tr.test()
tr.reset()
tr.save_agents(first_rewards, prefix="first_")

for i in range(100):

    for k in range(200):
        tr.run()
        tr.reset()
        
    rewards = tr.test(symbol_list=["AAPL"])
    tr.reset()
    
    for key in rewards.keys():
        with open("data//test_results//" + key + ".csv", "a") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(str(rewards[key]))
    
    tr.save_agents(rewards, prefix=str(i) + "_")

print("FINAL FULL TEST:")
final_rewards = tr.test()
tr.save_agents(final_rewards, prefix="final_")
            
    
