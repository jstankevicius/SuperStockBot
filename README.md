# stockbot

This is a project I built for my Software Engineering class in my junior year of high school. It's mostly an experiment - I wanted to find out if it was possible for a neural network to develop a legitimate stock trading strategy by using models such as Black-Scholes to generate training data.

The back end of the project creates four agents and assigns each a different model to use as training data. The agents train for some number of sessions and are then tested against real-world data. (Which I decided to upload to GitHub for whatever reason. Free dataset for you, I guess.)

The front end was *supposed* to be a nice GUI interface to monitor several metrics for each bot, like loss or average percent return. In reality, trying to run gui.py somehow spawns another set of agents whose data is displayed on the interface, but not the ones whose progress is monitored through the shell. The GUI was built using Dash.
