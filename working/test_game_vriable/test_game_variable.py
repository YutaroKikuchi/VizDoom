#!/usr/bin/env python

#####################################################################
# This script presents how to use the most basic features of the environment.
# It configures the engine, and makes the agent perform random actions.
# It also gets current state and reward earned with the action.
# <episodes> number of episodes are played. 
# Random combination of buttons is chosen for every action.
# Game variables from state and last reward are printed.
#
# To see the scenario description go to "../../scenarios/README.md"
#####################################################################

from __future__ import print_function
from vizdoom import *

from random import choice
from time import sleep

def initialize_game():
    game = DoomGame()
    game.load_config("./config/basic.cfg")
    game.set_doom_map("map01")
    #game.add_available_game_variable(GameVariable.USER1)
    #game.add_available_game_variable(GameVariable.USER2)
    #game.add_available_game_variable(GameVariable.USER3)
    game.init()

    return game

actions = [[True, False, False], [False, True, False], [False, False, True]]

game = initialize_game()

# Run this many episodes
episodes = 1

# Sets time that will pause the engine after each action (in seconds)
# Without this everything would go too fast for you to keep track of what's happening.
sleep_time = 1.0 / DEFAULT_TICRATE # = 0.028

for i in range(episodes):
    print("Episode #" + str(i + 1))

    # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
    game.new_episode()

    while not game.is_episode_finished():

        # Gets the state
        state = game.get_state()

        pos_x = game.get_game_variable(GameVariable.USER1)
        pos_y = game.get_game_variable(GameVariable.USER2)
        pos_z = game.get_game_variable(GameVariable.USER3)
        test = game.get_game_variable(GameVariable.USER4)

        pos_x_ = doom_fixed_to_double(pos_x)
        pos_y = doom_fixed_to_double(pos_y)
        pos_z = doom_fixed_to_double(pos_z)

        # Which consists of:
        n = state.number
        vars = state.game_variables
        screen_buf = state.screen_buffer
        depth_buf = state.depth_buffer
        labels_buf = state.labels_buffer
        automap_buf = state.automap_buffer
        labels = state.labels

        # Makes a random action and get remember reward.
        r = game.make_action(choice(actions))

        # Makes a "prolonged" action and skip frames:
        # skiprate = 4
        # r = game.make_action(choice(actions), skiprate)

        # The same could be achieved with:
        # game.set_action(choice(actions))
        # game.advance_action(skiprate)
        # r = game.get_last_reward()

        # Prints state's game variables and reward.
        print("State #" + str(n))
        print("test:",test)
        print("Game variables:", vars)
        print("X{0}, Y:{0}, X:{0}".format(pos_x_, pos_y, pos_z))
        print(type(pos_x))
        print("Reward:", r)
        print("=====================")

        if sleep_time > 0:
            sleep(sleep_time)

    # Check how the episode went.
    print("Episode finished.")
    print("Total reward:", game.get_total_reward())
    print("************************")

# It will be done automatically anyway but sometimes you need to do it in the middle of the program...
game.close()
