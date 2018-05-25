# This is a data loader class that helps with maintaining an active database
# of stock data. The "manager" instance allows the user to create an initial
# database via init_write() and to then continuously update it with fetch().

# To update the dataset, simply do this in the terminal:
# >>>from data_loader import manager
# >>>m = manager()
# >>>m.fetch()

# This is a time-expensive process due to AV's throttling of requests. If you
# call fetch(), be sure that you have enough time for the process to complete.
# It should also be noted that since safe writing was removed (i.e. being able
# to view the changes that will be written before "pushing" them onto the files
# via push() [kind of like GitHub]), one must be absolutely sure that calling
# fetch() is necessary, since once the method is called, the files will be
# edited immediately.

# Future features: automatic fetching of macroeconomic data.

from alpha_vantage.timeseries import TimeSeries
import time
import csv
import datetime
import finance

def fetch_data(index, row):
    """Reformats pandas-format data from AV into a list."""
    timedata = index.split()

    # Format: yyyy-mm-dd
    date = timedata[0]
    timeofday = timedata[1]

    # Monetary data
    o = row["1. open"]
    h = row["2. high"]
    l = row["3. low"]
    c = row["4. close"]
    v = row["5. volume"]
    
    return [date, timeofday, o, h, l, c, v]

def get_symbols():
    """Returns all stock symbols in data//symbols.txt as a list."""
    symbol_file = open("data//symbols.txt", "r")
    symbols = symbol_file.readlines()

    corrected = []

    # Correcting for newline delimiters.
    for symbol in symbols:
        corrected.append(symbol.rstrip())

    return corrected

class manager:
    """Class to help maintain a database of intraday stock data for multiple companies."""
    ts = TimeSeries("J4XLT1RK0S2QK5X0", output_format="pandas")
    
    def __init__(self):
        pass

    def read_as_list(self, stock_symbol):
        l = []
        with open("data//intraday//" + stock_symbol + ".csv", "r+") as file:
            reader = csv.reader(file)
            for row in reader:
                l.append(row)
                
        return l


    def init_write(self):
        """Creates files for each symbol in symbols.txt and populates with as much data as possible from AV."""
        for stock_symbol in get_symbols():
            intraday_data, meta_data = self.ts.get_intraday(symbol=stock_symbol, outputsize="full", interval="1min")

            with open("data//intraday//" + stock_symbol + ".csv", "w") as intraday_file:
                writer = csv.writer(intraday_file, lineterminator="\n")

                # Writing column titles:
                writer.writerow(["date", "ToD", "open", "high", "low", "close", "volume"])

                for index, row in intraday_data.iterrows():
                    writer.writerow(fetch_data(index, row))

            # To prevent AV from throttling our requests completely, we need to
            # minimize our call frequency. Around half a call per second is good
            # enough to prevent AV from throwing an error.
            time.sleep(2)

    def fetch(self, *, start=get_symbols()[0], append=False):
        """While iterating through symbols.txt, checks for updates to be written to each symbol."""
        symbols = get_symbols()

        # Human-readable index.
        n = 1

        # Iterate up to start and update n accordingly.
        while symbols[n - 1] != start:
            n += 1
        
        print("#\tTime\t\t\tSymbol\tLast update\t\tDeltas")

        for stock_symbol in symbols[(n-1):]:
            intraday_data, metadata = self.ts.get_intraday(symbol=stock_symbol, outputsize="full", interval="1min")

            # Ugly formatting. Oh well.
            print(str(n) + ".\t" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\t" + stock_symbol, end="")

            # String denoting the file that the open() method should access.
            filepath = "data//intraday//" + stock_symbol + ".csv"

            # Empty string fields that denote the date and time of day of the last index.
            lastdate = ""
            lasttime = ""

            # This is not a pretty solution at all. To fetch the last row of the csv file, it is
            # apparently necessary to read over the entire thing, which slows down the updating
            # process significantly.
            with open(filepath, "r") as file:
                reader = csv.reader(file)
                
                for row in reader:
                    lastdate = row[0]
                    lasttime = row[1]
                    
            print("\t" + lastdate + " " + lasttime, end="")
            
            # While this is False, all data read is ignored. Once the last date and time match
            # the date and time being read, flag is flipped to True, and every subsequent row
            # is written to the csv file.
            flag = False

            # Number of changes written to file.
            deltas = 0
            
            with open(filepath, "a") as file:
                writer = csv.writer(file, lineterminator="\n")

                # Iterate through stock data:
                for index, row in intraday_data.iterrows():
                    array = index.split()
                    date = array[0]
                    timeofday = array[1]
                    if (flag == True) or (flag == False and append == True):
                        writer.writerow(fetch_data(index, row))
                        deltas += 1 
                    
                    if flag == False:
                        if date == lastdate and timeofday == lasttime:
                            flag = True
                            
            print("\t" + str(deltas))
            n += 1

            # Hardcoded wait time to avoid AV throttling.
            time.sleep(2)
