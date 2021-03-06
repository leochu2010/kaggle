""" Titanic neural network
Author : Leo Chu
Date : 18th May 2016
Revised: 18th May 2016

""" 
import pandas as pd
import numpy as np
import csv as csv
from pybrain.structure import FeedForwardNetwork
from sklearn.ensemble import RandomForestClassifier
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import LinearLayer, SigmoidLayer, SoftmaxLayer


print 'preprocessing...'

# Data cleanup
# TRAIN DATA
train_df = pd.read_csv('../../train.csv', header=0)        # Load the train file into a dataframe

# I need to convert all strings to integer classifiers.
# I need to fill in the missing values of the data and make it complete.

# female = 0, Male = 1
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.

# All missing Embarked -> just make them embark from most common place
if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

# All the ages with no data -> make the median of all Ages
median_age = train_df['Age'].dropna().median()
if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
    train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age

# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 


# TEST DATA
test_df = pd.read_csv('../../test.csv', header=0)        # Load the test file into a dataframe

# I need to do the same with the test data now, so that the columns are the same as the training data
# I need to convert all strings to integer classifiers:
# female = 0, Male = 1
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# All missing Embarked -> just make them embark from most common place
if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values
# Again convert all Embarked strings to int
test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)


# All the ages with no data -> make the median of all Ages
median_age = test_df['Age'].dropna().median()
if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age

# All the missing Fares -> assume median of their respective class
if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]

# Collect the test data's PassengerIds before dropping it
ids = test_df['PassengerId'].values
# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 


# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train_df.values
test_data = test_df.values

# write preprocessing function later

print 'Normalize...'
# normalize data


# train neural network

# can I store a neural network?? yes
#http://stackoverflow.com/questions/6006187/how-to-save-and-recover-pybrain-training

# predict data afterward

train_df = (train_df)/(train_df.max())
test_df = (test_df)/(test_df.max())

train_ds = SupervisedDataSet(7,1)
for index, row in train_df.iterrows():
	#print row
	train_ds.addSample((row['Pclass'],row['Age'],row['SibSp'],row['Parch'],row['Fare'],row['Embarked'],row['Gender']),(row['Survived'],))
	#print index

print 'training data length',len(train_ds)

print 'training data shape:',train_df.shape
#print(train_df.describe())

print 'test data shape:',test_df.shape
#print(test_df.describe())

print 'Training...'
net = buildNetwork(7,4,1)

trainer = BackpropTrainer(net, train_ds)

error = 1
epoch = 0
while epoch < 30000:
	 error = trainer.train()
	 epoch += 1
	 print 'epoch:',epoch,' error:',error
#trainer.trainUntilConvergence(verbose=True)



#forest = RandomForestClassifier(n_estimators=100)
#forest = forest.fit( train_data[0::,1::], train_data[0::,0] )

print 'Predicting...'
#output = forest.predict(test_data).astype(int)


predictions_file = open("neural_network.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])


for index, row in test_df.iterrows():
	predict=net.activate([row['Pclass'],row['Age'],row['SibSp'],row['Parch'],row['Fare'],row['Embarked'],row['Gender']])
	if predict > 0.5:
		predict = 1
	else:
		predict = 0
	#print ids[index],':',predict
	open_file_object.writerow([ids[index],predict])
	
predictions_file.close()
print 'Done.'