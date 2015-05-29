thepath = 'C:\Users\Spyros\OneDrive\workspace\lerot\output_data\listwise_LL_evaluation_data\\Fold1\\data.csv'
data = []
with open(thepath, 'r') as f:
    datastring = f.read()
data = datastring.strip().split('\n')

performance = []
my_wins = 0.0
site_wins = 0.0

for i in data:
    current_run = i.split(',')
    if int(current_run[0]) > int(current_run[1]):
        my_wins += 1
    if int(current_run[0]) < int(current_run[1]):
        site_wins += 1
    if my_wins > 0:
        performance.append(my_wins / (my_wins + site_wins))
    else:
        performance.append(0)

outString = ''
for i in performance:
    outString += ''.join((str(i), '\n'))
print outString
with open('C:\Users\Spyros\OneDrive\workspace\lerot\output_data\listwise_LL_evaluation_data\\Fold1\\out.csv', 'w') as f:
    f.write(outString)

'''
thepath = 'C:\Users\Spyros\OneDrive\workspace\lerot\output_data\pairwise_local_evaluation_data\\'
data = []
for fold in range(1,6):
    for iteration in range(5):
        with open(thepath+'fold'+str(fold)+'_'+str(iteration), 'r') as f:
            string = f.read()
            data.append(string.split('\n'))
            
string = ''

for i in range(1000):
    for j in range(25):
        string += data[j][i]+','
    string = string[:-1]
    string += '\n'
    
with open(thepath+'finalOut.csv', 'w') as f:
    string = f.write(string)'''
