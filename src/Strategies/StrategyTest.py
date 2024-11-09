import numpy as np
import pandas as pd
from random import randint
import matplotlib.pyplot as plt
from templateMPT import Strategie
import warnings

warnings.warn = lambda *args,**kwargs: None

# OppsCount udává počet sázek v tabulce. OddmakerOff je offset bookmakera v procentech, tj. jak moc maximálně se může minout.
# OurOff udává jak moc se můžeme minout my od pravdivé pravděpodobnosti. AvgSpread udává průměr spreadu. SpreadRange udává o kolik procentních
# bodů se může spread měnit.

def CreateData(OppsCount, BookmakerOff, OurOff, AvgSpread, SpreadRange, Bankroll):
    # Zatím chybí datum
    summary = {"Bankroll":Bankroll,
               "Min_bet": 1,
               "Date": "1975-11-06",
               "Max_bet": 100}
    summary = pd.DataFrame(summary, index=[0])
    ID = [i for i in range(0,OppsCount)]
    OurOddlist =[]
    KurzyA = []
    KurzyB = []

    for p in range(0,OppsCount):

        #Generuji náhodnou pravděpodobnost, a psti naše a bookmakera v rámci předem dané odchylky.
        TrueOdds = randint(2000,8000)
        OurOdds = (TrueOdds + (-1)**randint(1,2)*randint(1,int((OurOff*100)+1)))/10000
        BookmakerOdds = (TrueOdds + (-1)**randint(1,2)*randint(1,int((BookmakerOff*100)+1)))/10000

        #Kalkulace spreadu a odchylky
        TrueSpread = (AvgSpread + (-1)**(randint(1,2))*randint(1,int((SpreadRange*1000)+1))/1000)/100
        KurzA = 1/(BookmakerOdds + TrueSpread/2)
        KurzB = 1/(1-BookmakerOdds + TrueSpread/2)

        KurzyA.append(KurzA)
        KurzyB.append(KurzB)
        OurOddlist.append(OurOdds)
    

    opps = {"ID":ID,
            "OddsH": KurzyA,
            "OddsA": KurzyB,
            "ProH": OurOddlist}
    opps = pd.DataFrame(opps, index = opps["ID"])
    return(summary, opps, TrueOdds)


def TestStrategy(Days, OppsCount, BookmakerOff, OurOff, AvgSpread, SpreadRange):
    BankrollSet = [1000]

    for d in range(0,Days):

        if d == 0:
            bankroll = 1000
        
        summary, opps, sance = CreateData(OppsCount, BookmakerOff, OurOff, AvgSpread, SpreadRange, bankroll)
        
        bets = Strategie(summary,opps)
        for index, row in bets.iterrows():
            diceroll = randint(1,10000)

            # Položíme obě sázky z každého řádku
            bankroll -= row["BetH"]
            bankroll -= row["BetA"] 

            # a vyplatíme je podle daných kurzů
            if diceroll <= sance:
                bankroll += row["BetH"]*opps["OddsH"][index]
            else:
                bankroll += row["BetA"]*opps["OddsA"][index]
            #print(bankroll)
        
        BankrollSet.append(bankroll)

    plt.plot(BankrollSet)

for r in range(0,10):
    # Dny, Sázek za den, Range bookmakera, Náš range, Spread bookmakera, Range spreadu
    TestStrategy(300, 10, 3.2, 2.5, 3, 1)

plt.show()
