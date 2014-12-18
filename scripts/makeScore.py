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
    print current_run
    if int(current_run[0]) > int(current_run[1]):
        my_wins += 1
    if int(current_run[0]) < int(current_run[1]):
        site_wins += 1
    if my_wins > 0:
        performance.append(my_wins / (my_wins + site_wins))
    else:
        performance.append(0)

print performance