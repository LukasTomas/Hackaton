# Here is template

def Strategie(summary, opps):
    # summary is unchanged from the original doc
    # summary with columns [Bankroll, Date, Min_bet, Max_bet]
    # opps with additional column that  contains our probability
    # the column is name ProH (Probability H)
    # opps with columns [ID, Date, OddsH, OddsA, BetH, BetA, ProH]

    # This method will be called from "model" so think of it as a main that gets parameters above as arguments
    ...
    bets = ...
    # retrun dataset with columsn [ID, BetH, BetA]
    return bets