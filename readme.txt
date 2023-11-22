Play_GridWorld.py let's you play the game yourself

Statistics.py is the statistics program that generated our statistics. The epsilon values it tests are defined in the list in the main for loop (right now only zero: "for epsilon in reversed([0]):")

statistics.txt is the our resulting statistics with confidence intervals, means, raw data for given epsilon values. (Note that many have 0 in mean, this simply means the epsilon led to cases where it took more than 10000 steps to get a succesfull algorithm, so we stopped simulating.)

AI_GridWorld.py a training algorithm for the game with the specific epsilon-value 0.1. This is what is simulated in Statistics.py with many different epsilon values and many different times

person.png, key.png, door.png, death.png, board.txt and GridWorld.py are necessary game files that do not need to be opened manually
