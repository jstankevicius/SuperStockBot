# Contains 4 classes of stock price "behaviors":
# classical_brownian
# geometric_brownian
# jump_diffusion
# heston_stochastic
# Each of these is an instantiable object, requiring only a single argument
# that represents the stock symbol of a company. Immediately upon
# initialization, the object calculates the necessary parameters defining
# the behavior of the stock. One is then able to call a step() function
# to provide the next price point of a company's price as computed
# via the 4 different models.
import numpy as np
import csv
from matplotlib import pyplot as plt
import statistics

# This is a test class to integrate into a working skeleton. No one would ever
# actually use this in finance, but that's not the point; we simply want
# a proof-of-concept object that returns values.
# YOU SHOULD ALMOST DEFINITELY NOT FOLLOW THIS FORMAT.
class test_behavior:

    # Minute volatility of the stock. Almost definitely calculated incorrectly.
    sigma = 0

    # Last price of the stock.
    last_price = 0
    
    def __init__(self, symbol):
        """Calculates the parameters necessary to perform later calculations
        with the step function."""
        prices = []
        file = open("data//intraday//" + symbol + ".csv", "r")
        reader = csv.reader(file)

        date = "2018-03-16"

        # Get a single day's prices:
        for row in reader:
            if row[0] == date:
                prices.append(float(row[5]))
        
        self.last_price = prices[len(prices) - 1]
        print("Initial price: $" + str(self.last_price))
        deltas = []
        for i in range(1, len(prices)):
            deltas.append((prices[i] - prices[i - 1])/prices[i - 1]*100)

        # For whatever reason I decided to divide by the square root of 690,
        # the amount of minutes in a trading day. I don't even remember why
        # I did this. Whatever. At least this gives nice enough numbers.
        self.sigma = np.std(deltas)/26.27

    def step(self):
        mu = 0
        n = 690
        dt = 0.0001
        step = np.exp((mu-self.sigma**2/2)*dt)*np.exp(self.sigma*np.random.normal(0, self.sigma, 1))

        next_price = self.last_price * step
        self.last_price = next_price
        
        return next_price
        
# Below here we have a list of classes that can be used to produce differing
# simulations of stock behavior. Each takes a starting price and a standard
# deviation as parameters in its constructor (those values can also be computed
# using actual stock data via the calc_stock method). We then use the step
# method to calculate the value of an asset some number of minutes into the future.

# Simple brownian motion
class classical_brownian:

    # Constructor that initializes instance variables.
    def __init__(self,start=1000,sigma=6):
        self.sigma = sigma
        self.start = start
    
    # Method to randomly generate num data points to simulate stock behavior.
    # The behavior is based on the starting price and standard deviation
    # stored in the object's instance variables, which can be adjusted using
    # calc_stock.
    def step(self,num):
        z = np.sqrt(1/num) * np.random.standard_normal(num)
        z *= self.sigma
        output = np.cumsum(z)
        output += np.expand_dims(self.start,axis=-1)
        return output

    # Method that takes intraday data from a single stock on a given date
    # and sets the sigma and start values of the object based on that data.
    # Nothing is returned - we just make changes to instance variables.
    def calc_stock(self,symbol,date):

        # Read data from the intraday folder
        file = open("data//intraday//" + symbol + ".csv","r")
        reader = csv.reader(file)
        closings = []
        for row in reader:
            if row[0] == date:
                closings.append(float(row[5]))

        # Set sigma
        sd = statistics.stdev(closings)
        self.sigma = sd

        # Set start
        self.start = closings[0]

# Geometric brownian motion
class geometric_brownian:

    # Constructor that initializes instance variables.
    def __init__(self,start=1000,sigma=6):
        self.sigma = sigma
        self.start = start

    # Method to randomly generate num data points to simulate stock behavior.
    # The behavior is based on the starting price and standard deviation
    # stored in the object's instance variables, which can be adjusted using
    # calc_stock.
    def step(self,num):
        z = np.sqrt(1/num) * np.random.standard_normal(num) 
        z *= self.sigma/self.start
        output = np.cumsum(z)
        output = np.exp(output)
        output *= self.start
        return output

    # Method that takes intraday data from a single stock on a given date
    # and sets the sigma and start values of the object based on that data.
    # Nothing is returned - we just make changes to instance variables.
    def calc_stock(self,symbol,date):

        # Read data from the intraday folder
        file = open("data//intraday//" + symbol + ".csv","r")
        reader = csv.reader(file)
        closings = []
        for row in reader:
            if row[0] == date:
                closings.append(float(row[5]))

        # Set sigma
        sd = statistics.stdev(closings)
        self.sigma = sd

        # Set start
        self.start = closings[0]

# Heston stochastic volatility model
class heston:

    # Constructor that initializes instance variables.
    def __init__(self,start=1000,sigma=6):
        self.sigma = sigma
        self.start = start

    # Method to randomly generate num data points to simulate stock behavior.
    # The behavior is based on the starting price and standard deviation
    # stored in the object's instance variables, which can be adjusted using
    # calc_stock.
    def step(self,num):

        # Calculation of v, which is a volatility based on a Weiner process.
        z1 = np.sqrt(1/num) * np.random.standard_normal(num)
        z1 = np.cumsum(z1)
        rtv = (self.sigma * z1) # this should probably be /2 but whatever

        # Computing the stock price
        z2 = np.sqrt(1/num) * np.random.standard_normal(num)
        output = np.multiply(rtv,z2)/self.start

        ### An alternate implementation that produces less volatility than the
        ### line above but has far more iterations than using numpy does. 
        ##output = np.empty(num)
        ##for i in range(num):
        ##   output[i] = z2[i] * rtv[i]
        ##output /= self.start

        output = np.cumsum(output)
        output = np.exp(output)
        output *= self.start
        return output

    # Method that takes intraday data from a single stock on a given date
    # and sets the sigma and start values of the object based on that data.
    # Nothing is returned - we just make changes to instance variables.
    def calc_stock(self,symbol,date):

        # Read data from the intraday folder
        file = open("data//intraday//" + symbol + ".csv","r")
        reader = csv.reader(file)
        closings = []
        for row in reader:
            if row[0] == date:
                closings.append(float(row[5]))

        # Set sigma
        sd = statistics.stdev(closings)
        self.sigma = sd

        # Set start
        self.start = closings[0]

# Merton-jump diffusion model
class merton:

    # Constructor that initializes instance variables.
    def __init__(self,start=1000,sigma=6):
        self.sigma = sigma
        self.start = start

    # Method to randomly generate num data points to simulate stock behavior.
    # The behavior is based on the starting price and standard deviation
    # stored in the object's instance variables, which can be adjusted using
    # calc_stock.
    def step(self,num):

        # Instantiate a geometric brownian model. This is the basis for the
        # merton model, which will simply add a jump array to the output.
        gb = geometric_brownian(self.start,self.sigma)
        base = gb.step(num)

        # Instantiate the jump array
        jump = np.zeros(num)

        # For loop to create the jumps
        for i in range(num):
            temp = 0

            # Introduce a probability factor to determine if a jump occurs.
            # The tanh function mostly stops jumps in the first hour of
            # trading and then has very little impact.
            if np.random.random() < np.tanh(i/69):
                
                for j in range(int((np.random.poisson(1)))):
                    temp += (np.random.lognormal() * ((-1) ** np.random.randint(0,2)))
                    
            jump[i] = temp * self.sigma * np.sqrt(1/num)/2
                
        jump = np.cumsum(jump)
        
        # Add the arrays
        return np.add(base,jump)

    # Method that takes intraday data from a single stock on a given date
    # and sets the sigma and start values of the object based on that data.
    # Nothing is returned - we just make changes to instance variables.
    def calc_stock(self,symbol,date):

        # Read data from the intraday folder
        file = open("data//intraday//" + symbol + ".csv","r")
        reader = csv.reader(file)
        closings = []
        for row in reader:
            if row[0] == date:
                closings.append(float(row[5]))

        # Set sigma
        sd = statistics.stdev(closings)
        self.sigma = sd

        # Set start
        self.start = closings[0]

# Test of the four random simulations using pyplot.
t = range(690)

##for i in range(1):
##    cb = classical_brownian()
##    cb.calc_stock("MSFT","2018-03-16")
##    out = cb.step(690)
##    plt.plot(t,out)
##    plt.title("Classical")

##for i in range(100):
##    gb = geometric_brownian()
##    gb.calc_stock("GE","2018-03-16")
##    out = gb.step(690)
##    plt.plot(t,out)
##    plt.title("Geometric")

##for i in range(100):
##    h = heston()
##    h.calc_stock("GOOG","2018-03-16")
##    out = h.step(690)
##    plt.plot(t,out)
##    plt.title("Heston")

##for i in range(100):
##    m = merton()
##    m.calc_stock("GOOG","2018-03-16")
##    out = m.step(690)
##    plt.plot(t,out)
##    plt.title("Merton")
##
##plt.xlabel("Minutes")
##plt.ylabel("Stock Price")
##plt.show()

##for i in range(20):
##    print(np.random.lognormal())
