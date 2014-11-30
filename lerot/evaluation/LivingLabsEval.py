

class LivingLabsEval:
    
    __wins_list__ = None
    __performance_list__ = None
    
    
    def __init__(self):
        self.__wins_list__ = []
        self.__performance_list__ = []
    
    
    
    def update_score(self, wins_list):
        if True:
            self.__wins_list__.append(1)
            self.__performance_list__.append(sum(self.__wins_list__)/float(len(self.__wins_list__)))
        else:
            self.__wins_list__.append(0)
    
    
    
    def get_performance(self):
        return self.__performance_list__[len(self.__performance_list__)-1]