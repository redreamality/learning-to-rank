import random

from TeamDraft import TeamDraft


class TdiVa(TeamDraft):
    """ Basis class for TdiVa1 and TdiVa2 """

    @staticmethod
    def sampleSmoothly(a, b, maxVal):
        if a > b:
            a, b = b, a
        if a > 0 and b < maxVal:
            randVal = random.randint(a, b + 1)
            if randVal == b + 1:
                return a - 1 if random.randint(0, 1) == 0 else b + 1
            else:
                return randVal
        elif a == 0 and b == maxVal:
            return random.randint(a, b)
        else:   # a > 0 or b < maxVal
            randVal = random.randint(0, 2 * (b - a) + 2)
            if randVal == 2 * (b - a) + 2:
                return (a - 1) if a > 0 else b + 1
            else:
                return a + randVal // 2
