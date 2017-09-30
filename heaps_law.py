from pandas import HDFStore, DataFrame, Panel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import powerlaw
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as offline
from numpy import sqrt, pi, exp, linspace, loadtxt
from lmfit import  Model
import pylab


#Getting data frames from HDF files that we need to calculate entropy
file_name = 'file.hdf'
store =  pd.HDFStore(file_name)

adjMat = store['df_info']
emoji_list = adjMat['emoji_List']
emojis = pd.DataFrame(emoji_list)

#Counting emojis and not seen previously
k = []
beta = []
iam = 0
for cont in range(30):
    iam += 1
    print ('I am on cycle %s'%iam)
    set_of_emojis_seen = set()
    counter_of_emojis = 0
    counter_of_not_seen = 0
    count = 0
    emojis_seen = []
    emojis_not = []
    n = len(emojis) * 1.0
    emojis = emojis.iloc[np.random.permutation(len(emojis))]
    emojis = emojis.sample(n = int(n))


    #Going through the tweets and finding each emoji
    #If they are have not been seen before I put them in a set and check the set every iteration
    for row in range(len(emojis)):
        count += 1
        #print ('I am on tweet %s' %count)
        emoji_list = emojis.iloc[row][0]
        for emoji in range(len(emoji_list)):
            counter_of_emojis += 1
            emojis_seen.append(counter_of_emojis)
            check = emoji_list[emoji]
            if check not in set_of_emojis_seen:
                counter_of_not_seen += 1
                set_of_emojis_seen.add(check)
                emojis_not.append(counter_of_not_seen)
            else:
                emojis_not.append(counter_of_not_seen)

    #Changing list to an array to plot in matplotlib
    data = np.asarray(emojis_not)
    x_values = np.asarray(emojis_seen)

    #Creating best fit line
    def heaps_law(n, k, beta):
        "heaps law"
        return k*n**beta

    gmodel = Model(heaps_law)
    result = gmodel.fit(data, n=x_values, k=11, beta = 0.4 )
    dictonary = result.best_values
    for value in range(len(dictonary)):
        if value == 0:
            k.append(dictonary.values()[0])
        else:
            beta.append(dictonary.values()[1])


# print beta
# print k
beta = np.asarray(beta)
k = np.asarray(k)
plt.hist(beta)
plt.title('Beta Histogram')
plt.xlabel('Beta')
plt.ylabel('Count')
plt.savefig('beta_histo_orga.pdf')
plt.show()

plt.hist(k)
plt.title('K Histogram')
plt.xlabel('K')
plt.ylabel('Count')
plt.savefig('k_histo_beta.pdf')
plt.show()

k = np.average(k)
beta = np.average(beta)
x = np.linspace(0,500000) # 100 linearly spaced numbers
#y = numpy.sin(x)/x # computing the values of sin(x)/x

# compose plot
pylab.plot(x,k*x**beta, label = 'Organs ( K = %s, b = %s)'%(k,beta))
pylab.legend(loc = 'lower right')
pylab.xlabel('Number Of Emojis Seen')
pylab.ylabel('Number Of New Emojis Seen')
pylab.title('Heaps Law Random')
pylab.savefig('hp_test_random.pdf')
#pylab.plot(x,y,'co') # same function with cyan dots
#pylab.plot(x,2*y,x,3*y) # 2*sin(x)/x and 3*sin(x)/x
pylab.show()
\
