import numpy as np

class cBandit:
    def __init__(self, tCash=100, pRatio=.1):
        self.pot = 0
        self.bank = 0
        self.tCash = tCash
        self.pRatio = pRatio

    def play(self, amt=1):
        self.pot += amt
        if np.random.poisson(self.pot, 1) > self.tCash:
            profit = np.ceil(self.pot * self.pRatio)
            cashout = self.pot - profit
            self.bank += profit
            self.pot = 0
            return cashout
        else:
            return 0

class cCasino:
    def __init__(self, tCash,pRatio=.1):
        self.bandits=[]

        for c in tCash:
            self.bandits.append(cBandit(c,pRatio))

    def play(self, amt=1):
        p=[]
        for b in self.bandits:
           p.append(b.play(amt))
        return p

