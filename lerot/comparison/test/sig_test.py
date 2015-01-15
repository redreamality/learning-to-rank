import numpy as np
import evaluateData

def get_significance(mean_1, mean_2, std_1, std_2, n):
   significance = ""
   ste_1 = std_1 / np.sqrt(n)
   ste_2 = std_2 / np.sqrt(n)
   t = (mean_1 - mean_2) / np.sqrt(ste_1 ** 2 + ste_2 ** 2)
   if mean_1 > mean_2:
       # treatment is worse than baseline
       # values used are for 120 degrees of freedom
       # (http://changingminds.org/explanations/research/analysis/
       # t-test_table.htm)
       if abs(t) >= 2.62:
           significance = "\dubbelneer"
       elif abs(t) >= 1.98:
           significance = "\enkelneer"
   else:
       if abs(t) >= 2.62:
           significance = "\dubbelop"
       elif abs(t) >= 1.98:
           significance = "\enkelop"
   return significance
 
 
def readallData():
    errors = readData()
    errors_informational = errors[0]
    errors_navigational = errors[1]
    errors_perfect = errors[2]
    return errors_informational, errors_navigational, errors_perfect
     
def test_all_significance(testPoint = 4999):
    errors = evaluateData.readData()
    for x in range(len(errors)):
        PM_mean = np.mean(np.array([errors[x][i][j][testPoint][1] for i in range(5) for j in range(5)]))
        PM_std = np.std(np.array([errors[x][i][j][testPoint][1] for i in range(5) for j in range(5)]))
        PI_mean = np.mean(np.array([errors[x][i][j][testPoint][-1] for i in range(5) for j in range(5)]))
        PI_std = np.std(np.array([errors[x][i][j][testPoint][-1] for i in range(5) for j in range(5)]))
        TDM_mean = np.mean(np.array([errors[x][i][j][testPoint][2] for i in range(5) for j in range(5)]))
        TDM_std = np.std(np.array([errors[x][i][j][testPoint][2] for i in range(5) for j in range(5)]))
        
        sig_PMvPI = get_significance(PM_mean,PI_mean,PM_std,PI_std,25)
        sig_PMvTDM = get_significance(PM_mean,TDM_mean,PM_std,TDM_std,25)
        print '============'
        print "PM " + str(PM_mean) + str(sig_PMvPI) + '\n' + "PI" + str(PI_mean)
        print "PM " + str(PM_mean) + str(sig_PMvTDM) + '\n' + "TDM" + str(TDM_mean)
        print '============'
    
    
def make_table():
    pass