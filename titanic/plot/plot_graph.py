import pandas as pd
import numpy as np
import csv as csv
import math
import matplotlib.pyplot as plt


# Data cleanup
# TRAIN DATA
train_df = pd.read_csv('../train.csv', header=0)        # Load the train file into a dataframe
# TEST DATA
test_df = pd.read_csv('../test.csv', header=0)        # Load the test file into a dataframe

train_set = train_df.drop("Survived", axis = 1)
df_combo = pd.concat((train_set, test_df), axis = 0, ignore_index = True)

'''
plt.plot([1,2,3,4])
plt.ylabel('some numbers')
plt.show()
'''

train_df['family']=train_df["SibSp"] + train_df["Parch"] + 1
x = train_df.loc[ (train_df.Survived == 1),'family' ]

survived_family_array = np.array(train_df.loc[ (train_df.Survived == 1),'family' ].value_counts(), dtype=pd.Series)
died_family_array = np.array(train_df.loc[ (train_df.Survived == 0),'family' ].value_counts(), dtype=pd.Series)
family_array = np.array([survived_family_array,died_family_array])

survived_family_dict = dict(train_df.loc[ (train_df.Survived == 1),'family' ].value_counts())
died_family_dict = dict(train_df.loc[ (train_df.Survived == 0),'family' ].value_counts())

#colors = ['red', 'tan', 'lime']

print(survived_family_array)
plt.hist(survived_family_array, histtype='bar', usevlines=True)
plt.legend(prop={'size': 10})
plt.show()



'''
plt.bar(range(len(survived_family_count)), survived_family_count.values(), align='center')
plt.xticks(range(len(survived_family_count)), survived_family_count.keys())
plt.bar(range(len(died_family_count)), died_family_count.values(), align='center')
plt.xticks(range(len(died_family_count)), died_family_count.keys())
'''


'''
#http://stackoverflow.com/questions/10369681/how-to-plot-bar-graphs-with-same-x-coordinates-side-by-side
import numpy as np
import matplotlib.pyplot as plt

N = 5
menMeans = (20, 35, 30, 35, 27)
menStd =   (2, 3, 4, 1, 2)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)
rects1 = ax.bar(ind, menMeans, width, color='r', yerr=menStd)

womenMeans = (25, 32, 34, 20, 25)
womenStd =   (3, 5, 2, 3, 3)
rects2 = ax.bar(ind+width, womenMeans, width, color='y', yerr=womenStd)

# add some
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('G1', 'G2', 'G3', 'G4', 'G5') )

ax.legend( (rects1[0], rects2[0]), ('Men', 'Women') )

plt.show()
'''