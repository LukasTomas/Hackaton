from math import comb
from datetime import datetime
import pandas as pd

# Kelly's criterion
def kelly_criterion(win_prob, multi):
    # win_prob: probability of winning
    # multi: odds multiplier (decimal odds - 1)
    multi -= 1
    loss_prob = 1 - win_prob
    f_star = (multi * win_prob - loss_prob) / multi
    return f_star

def strat_kelly_exp_value(summary, opps):
    # print(opps.to_string())
    summary = summary.iloc[0]
    bankroll = summary['Bankroll']
    total_invested = 0  # Track total investment amount
    min_bet = summary['Min_bet']
    max_bet = summary['Max_bet']
    max_investment_total = bankroll - min_bet*5  # Total investment limit at x% of bankroll
    if max_investment_total < 0:
        max_investment_total = 0
    # print(bankroll, max_investment_total, min_bet, max_bet)

    # Process each opportunity
    opps_new_bets = pd.DataFrame(columns=['NewBetH', 'NewBetA'])
    for index, opp in opps.iterrows():
        # Extract odds, probabilities, and historical bets
        odds_H = opp['OddsH']
        odds_A = opp['OddsA']
        our_prob_h = opp['ProH']
        if pd.isna(our_prob_h):
            new_betH = 0
            new_betA = 0       
            opps_new_bets.loc[index] = [new_betH, new_betA]
            continue

        our_prob_a = 1 - our_prob_h
        hist_betH = opp.get('BetH', 0)  # Previous bets on Home (0 if not present)
        hist_betA = opp.get('BetA', 0)  # Previous bets on Away (0 if not present)

        # Calculate expected values for Home and Away bets
        EV_h = our_prob_h * odds_H
        EV_a = our_prob_a * odds_A
        # print("Exp Val",EV_h, EV_a)

        # Calculate Kelly-based investment for bets with positive EV and within bankroll cap
        new_betH = 0
        new_betA = 0
        if EV_h > 1:
            investment = kelly_criterion(our_prob_h, odds_H) * max_bet
            investment = max(min(investment, max_bet), min_bet)  # Enforce min and max bet limits
            # Adjust for remaining allowable bankroll to avoid breaching cap
            investment = min(investment, max_investment_total - total_invested)
            new_betH = investment
            new_betA = 0
        elif EV_a > 1:
            investment = kelly_criterion(our_prob_a, odds_A) * max_bet
            investment = max(min(investment, max_bet), min_bet)
            investment = min(investment, max_investment_total - total_invested)
            new_betH = 0
            new_betA = investment
        else:
            # If no profitable bets, set investment to zero
            new_betH = 0
            new_betA = 0


        # print("BET", new_betH, new_betA)

        # Update total invested amount and save bet details if a bet is placed
        if new_betH > 0 or new_betA > 0:
            total_invested += new_betH + new_betA

        opps_new_bets.loc[index] = [new_betH, new_betA]
    
    opps = pd.concat([opps, opps_new_bets], axis=1)
    # print("Total Invested", total_invested)
    # print()
    return opps
