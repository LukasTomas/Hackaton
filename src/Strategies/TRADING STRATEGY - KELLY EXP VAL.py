#TRADING STRATEGY - KELLY EXP VAL
from random import randint
from math import comb
import matplotlib.pyplot as plt

## Pro všechny sázkařské příležitosti z Opps DataFrame
# počet příležitostí
#for betting_opportunity in Opps:

# input from the neural network
our_prob_h = 0.6 # probability of home team winning

# input from Opps DataFrame
odds_H = 1.930235
odds_A = 1.916195

# input from summary dataframe
bankroll = 1000
# if you bet -> min bet = 1
# max bet = 100
min_bet = 1
max_bet = 100

# Kelly's criterion

def kelly_criterion(win_prob, multi):
    # win_prob: probability of winning
    # multi: odds multiplier (decimal odds - 1)
    multi -= 1
    loss_prob = 1 - win_prob
    f_star = (multi * win_prob - loss_prob) / multi
    return f_star



def strat_kelly_exp_value(our_prob_h, odds_H, odds_A, bankroll):
    # calculated values
    our_prob_a = 1 - our_prob_h

    #Expected value
    EV_h = our_prob_h * odds_H
    EV_a = our_prob_a * odds_A

    if EV_h > 1:
        investment = kelly_criterion(our_prob_h, odds_H) * bankroll
        betA = 0
        betH = investment
    elif EV_a > 1:
        investment = kelly_criterion(our_prob_a, odds_A) * bankroll
        betA = investment
        betH = 0
    else:
        betA = 0
        betH = 0
        

    # output
    # send to Bets DataFrame

    return betH, betA


