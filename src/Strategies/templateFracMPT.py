import numpy as np
import pandas as pd
from scipy.optimize import minimize

def Strategie(summary, opps):
    # Příprava setů pro pravěpodobnosti
    odds = []
    prob = []
    # Fraction portfolia ve výpočtu - určeno pro risk managment
    frac = 0.7

    # Získáme basic info ze Summary - [Bankroll, Date, Min_bet, Max_bet]
    Bankroll = summary["Bankroll"][0]
    Max_bet = summary["Max_bet"][0]
    Date = summary["Date"][0]
    Rel_M = Max_bet/Bankroll

    # Iterujeme přes řádky, vykrademe podstatné info
    for index, row in opps.iterrows():
        OddH = row["OddsH"]
        OddA = row["OddsA"]
        ProH = row["ProH"]
        odds.append(OddH)
        odds.append(OddA)
        prob.append(ProH)
        prob.append(1-ProH)

    # Chci z toho np arraye
    p = np.array(prob)
    o = np.array(odds)
    n = len(p)
    m = Rel_M

    # Opps - [ID, Date, OddsH, OddsA, BetH, BetA, ProH]
    
    A_coeff = p * o - 1  # EV
    C_coeff = (1 - p) * p * o  # Variace

    def compute_E(x):
        b = x[:-1]
        w = x[-1]
        E = np.dot(A_coeff, b)
        # Kalkulace střední hodnoty
        return E

    def compute_V(x):
        b = x[:-1]
        V = np.dot(C_coeff, b ** 2)
        # Kalkulace variace
        return V

    def objective(x):
        E = compute_E(x)
        V = compute_V(x)
        if V <= 0:
            return np.inf  # Nechci zápornou variaci
        return -E / np.sqrt(V)  # Chci minimálního Sharpa (od toho minus), což je EV/Sqrt(Var)

    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Total allocation equals 1
        {'type': 'ineq', 'fun': lambda x: x[-1]}  # w >= 0
    ]

    # Bounds for each variable b_i and w
    bounds = [(0, m) for _ in range(n)] + [(0, None)]  # w - cash - nemá hranici, zbytek je limitován max. betem

    # Initial guess
    x0 = np.ones(n + 1) / (n + 1)  # Evenly distribute initial guess

    # Optimize
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
         # Check if the optimization was successful
        if not result.success:
            print("Optimization failed:", result.message)
            b_scaled = [0 for i in range(0,len(p))]
        else:
            # Bereme optimální řeš.
            x_optimal = result.x
            b_optimal = x_optimal[:-1]
            w_optimal = x_optimal[-1]

            # a našklujeme je bankrolem - původní b jsou poměry portfolia
            b_scaled = frac*Bankroll * b_optimal
            b_scaled = b_scaled.round()
    except:
        b_scaled = [0 for i in range(0,len(p))]
   

    # Výsledky nahážeme do předurčených množin
    BetH = []
    BetA = []

    for n in range(0,int(len(p)/2)):
        BetH.append(b_scaled[2*n])
        BetA.append(b_scaled[2*n+1])
    
    opps["BetH"] = BetH
    opps["BetA"] = BetA

    bets = opps[["ID","BetH","BetA"]]
    # a vyplyvneme požadovaný DFrame
    return bets