

class LivingLabsEval:
    
    __wins_list__ = None
    
    
    def __init__(self):
        self.__wins_list__ = []
    
    
    
    def update_score(self, wins):
        self.__wins_list__.append(wins)
    
    
    def get_win(self):
        return self.__wins_list__[len(self.__wins_list__)-1]
    
    
    def get_performance(self):
        total_wins = 0
        total_losses = 0
        for i in self.__wins_list__:
            if i[0]>i[1]:
                total_wins += 1
            if i[0]<i[1]:
                total_losses += 1
        if total_wins > 0:
            return (float(total_wins) / (total_losses + total_wins) )
        