import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Data provided
p = np.array([0.5, 0.5, 0.6, 0.4, 0.7, 0.3, 0.65, 0.35,0.8,0.2,0.7,0.3,0.67,0.23])
o = np.array([1.930235, 1.916195, 1.418280, 3.048582, 1.520244, 2.650582, 1.491004, 2.74833,1.302708,3.808664,1.471073,2.821692,1.505497,2.695813])
m = 0.1  # Maximum allocation per asset


def Strategie(summary, opps):
    # summary is unchanged from the original doc
    odds = []
    prob = []

    Bankroll = summary["Bankroll"]
    Max_bet = summary["Max_bet"]
    Rel_M = Max_bet/Bankroll
    # summary with columns [Bankroll, Date, Min_bet, Max_bet]
    
    for row in opps.rows():
        OddH = opps.loc([row, "OddsH"])
        OddA = opps.loc([row, "OddsH"])
        ProH = opps.loc([row, "ProH"])
        odds.append(OddH,OddA)
        prob.append(ProH,1-ProH)


    p = np.array(prob)
    o = np.array(odds)
    n = len(p)
    m = Rel_M
    # opps with additional column that  contains our probability
    # the column is name ProH (Probability H)
    # opps with columns [ID, Date, OddsH, OddsA, BetH, BetA, ProH]
    
    A_coeff = p * o - 1  # Expected return coefficients for risky assets
    C_coeff = (1 - p) * p * o  # Variance coefficients for risky assets

    def compute_E(x):
        b = x[:-1]
        w = x[-1]
        E = np.dot(A_coeff, b)
        # Riskless return is assumed to be zero
        return E

    def compute_V(x):
        b = x[:-1]
        V = np.dot(C_coeff, b ** 2)
        return V

    def objective(x):
        E = compute_E(x)
        V = compute_V(x)
        if V <= 0:
            return np.inf  # Penalize non-positive variance
        return -E / np.sqrt(V)  # Negative Sharpe ratio (since we minimize)

    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Total allocation equals 1
        {'type': 'ineq', 'fun': lambda x: x[-1]}  # w >= 0
    ]

    # Bounds for each variable b_i and w
    bounds = [(0, m) for _ in range(n)] + [(0, None)]  # w has no upper bound

    # Initial guess
    x0 = np.ones(n + 1) / (n + 1)  # Evenly distribute initial guess

    # Optimize
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    # Check if the optimization was successful
    if not result.success:
        print("Optimization failed:", result.message)
    else:
        # Get the optimal allocations
        x_optimal = result.x
        b_optimal = x_optimal[:-1]
        w_optimal = x_optimal[-1]

        # Multiply the allocations by 200
        b_scaled = Bankroll * b_optimal
        b_scaled = b_scaled.round()

    # This method will be called from "model" so think of it as a main that gets parameters above as arguments
    BetH = []
    BetA = []

    for n in range(0,len(p)/2):
        BetH.append(b_scaled[2*n])
        BetA.append(b_scaled[2*n+1])
    
    odds["BetH"] = BetH
    odds["BetA"] = BetA


    bets = odds[["ID","BetH","BetA"]]
    # retrun dataset with columsn [ID, BetH, BetA]
    return bets

