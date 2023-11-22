# Grid World: Statistics

# Control:
#    arrows  : Merge up, down, left, or right
#    s       : Toggle slow play
#    d       : Toggle rendering 
#    r       : Restart game
#    q / ESC : Quit

from GridWorld import GridWorld
import numpy as np
import pygame
import random
from scipy.stats import beta

# Initialize the environment
env = GridWorld()
env.reset()
x, y, has_key = env.get_state()

# Definitions and default settings
actions = ['left', 'right', 'up', 'down']
exit_program = False
action_taken = False
slow = True
runai = True
render = True
done = False
steps=0
ME=0
std=0
mean=0
waste_of_time=False

# Game clock
clock = pygame.time.Clock()

# INSERT YOUR CODE HERE (1/2)
# Define data structure for q-table
state_action_dictionary ={}
actions = ['left', 'right', 'up', 'down']
reward = 0

def find_list_of_best(dic: dict) -> list: 
    """takes a dictionary and returns the list of keys that have the highest values associated with them"""
    best = max(dic.values())
    bestmoves= []
    for i in dic:
        if dic[i] == best:
            bestmoves.append(i)
    return bestmoves

def clopper_pearson_interval(confidence_level: float, num_successes: int, num_trials: int) -> tuple:
    """Returns a tuple with the lower and upper bound respectively, for the exact confidence interval (clopper pearson) 
    for a given bernoulli experiment with a given confidence level"""
    lower = beta.ppf((1 - confidence_level) / 2, num_successes, num_trials - num_successes + 1)
    upper = beta.ppf((1 + confidence_level) / 2, num_successes + 1, num_trials - num_successes)
    if np.isnan(lower):
        lower=0
    if np.isnan(upper):
        upper=1
    return lower, upper

def is_success_rate_n_percent(dic: dict, n: float) -> bool:
    """Return True or False respectively if the successrate of the dictionary is above or below n*100%
    
    :param dic: state_action_dictionary scoring the possible moves in a certain gamestate
    :param n: how large a fraction of the time the dictionary has to lead to a win for the function to return True
    """
    # Initialization of a new simulation game environment
    sim = GridWorld()
    sim.reset()
    x, y, has_key = sim.get_state()
    wins=0
    total=0

    # Simulation
    done = False
    loop_preventer=0 #When the algorithm only takes the best move according to itself, it may become stuck in a loop. This is used to prevent that
    while True:
        if done or loop_preventer>100: # If the algorithm takes longer than 100 steps at reaching the door, it is counted as a fail
            loop_preventer=0
            # Count every time algorithm wins and also the total amount of games 
            if sim.won(x, y, has_key, sim.board):
                wins+=1
            total+=1

            # Every 10 simulations, check whether algorithm wins the game above or below n*100% of the time with a 99.99% confidence level, 
            # terminate while loop and return True or False accordingly
            if total%10==0: 
                lower, upper = clopper_pearson_interval(0.9999, wins, total)
                if lower>n:
                    if render:
                        print(wins,total,lower,upper) # for debugging, is active when rendering is active
                    return True # Terminates function and returns True
                elif upper<n:
                    if render:
                        print(wins,total,lower,upper) # for debugging, is active when rendering is active
                    return False # Terminates function and returns False
                if total>100000: # If the true proportion is very close to 95% it may continue indefinitely, therefore this if statement stops it after 100000 games and returns True if the proportion estimate is above n, and False if it isn't
                    percent=wins/total
                    if percent>n:
                        return True
                    else:
                        return False
                
            # reset sim game if function doesn't terminate (if we are not yet sure how good the algorithm is). Keep in mind this is under an "if done:" statement
            sim.reset()
            x, y, has_key = sim.get_state()

        # Below is the main loop that will take the decisions in the sim game, 
        # notice that there is no randomization of moves, it only takes the best move according to its dictionary/knowledge
        # 1. choose an action
        placement = (x, y, has_key)
        action_dict = dic.setdefault(placement, {'up':0, 'down':0, 'left':0, 'right':0})
        action_list = find_list_of_best(action_dict)
        action = action_list[np.random.randint(len(action_list))]
        
        # 2. step the environment
        (x, y, has_key), reward, done = sim.step(action)
        loop_preventer+=1

# Do statistics for a whole bunch of epsilon values
for epsilon in [0,0.005,0.01,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    results=[]
    waste_of_time=False # is explained later
    while (not exit_program and ME>0.05*mean) or len(results)<30: # For every instance of this loop, and entire q-learning process is done over with the same epsilon value. This continues until we have a reasonable confidence interval for the amount of steps before the game is won 95% of the time.
        if waste_of_time:
            break
        state_action_dictionary=dict() # reset dictionary to reset learning process
        steps=0 
        while not exit_program: # This is the loop that runs for a single learning process. We made this together, though some things have been stripped
            if render:
                env.render()
            
            # Slow down rendering to 5 fps
            if slow and runai:
                clock.tick(5)
                
            # Automatic reset environment in AI mode
            if done and runai:
                env.reset()
                x, y, has_key = env.get_state()
                
            # Process game events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit_program = True
                if event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                        exit_program = True
                    if event.key == pygame.K_r:
                        env.reset()   
                    if event.key == pygame.K_d:
                        render = not render
                    if event.key == pygame.K_s:
                        slow = not slow
            
            # 1. choose an action
            placement = (x, y, has_key)
            action_dict = state_action_dictionary.setdefault(placement, {'up':0, 'down':0, 'left':0, 'right':0})
            if random.random() < epsilon: # take a random action epsilon*100% of the time
                action = actions[np.random.randint(4)] 
            else: 
                action_list = find_list_of_best(action_dict)
                action = action_list[np.random.randint(len(action_list))] # else take a random action of the best actions available
            
            
            # 2. step the environment
            (x, y, has_key), reward, done = env.step(action)
            steps+=1 # the statistic we are measuring
            
            # 3. update q table
            action_dict = state_action_dictionary.setdefault((x, y, has_key), {'up':0, 'down':0, 'left':0, 'right':0})
            state_action_dictionary[placement][action]= reward + max(action_dict.values())


            # Gets successrate of current dictionary
            if steps%10==0: # We only check the success criteria every 10th step, otherwise it takes too long
                if is_success_rate_n_percent(state_action_dictionary,0.95):
                    results.append(steps)
                    break # Restart q-learning process
                if steps==10000: # If it takes more than 10000 steps for the algorithm to get good even just once, we stop simulating that epsilon value, as it is definitely worse than the others, and it takes too much time
                    print(f'Skipping {epsilon} as it took more than 10000 steps to get a good dictionary')
                    results=[0]
                    waste_of_time=True # Used to break out of the loops
                    break # Restart q-learning process

        # Calculate statistic values after every q-learning process, so we know when the confidence-interval is small enough to continue to next epsilon value.
        mean=np.mean(results)
        std=np.sqrt(sum([(_-mean)**2/(len(results)-1) for _ in results]))
        ME=2.947775*std/np.sqrt(len(results)) # 99,68% konfidensinterval
        print(results[-1], " ; ", ME, "|", 0.05*mean) 

    # When confidence-interval is small enough, write results to a file called "statistics.txt, in the same folder as this code file. After this the algorithm continues to next epsilon value"
    print(f'Epsilon: {epsilon}; Mean: {mean}, Confidence Interval: {mean - ME} - {ME + mean}')
    with open(r"./statistics.txt","a") as f:
        f.write((f'Epsilon: {epsilon}; Mean: {mean}; Confidence Interval: {mean - ME} - {ME + mean}; Raw: {results}\n'))

env.close()
