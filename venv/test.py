from collections import Counter

#example of negetive list indexes
finishers = ['kai', 'abe', 'ada', 'gus', 'zoe']
finishers  = finishers[ : -2]
x = finishers[-2:]
print(x)


# example for counters
chars = ['r','r','r','r','r','k','k','k']
print(Counter(chars).most_common(1)[0][0])