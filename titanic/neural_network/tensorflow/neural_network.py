import tensorflow as tf
import pandas as pd
import numpy as np
import csv as csv
import math

def calculate_mean_age(df,str):
	count = 0
	has_age_count = 0	
	total_age = 0.0
	for index, row in df.iterrows():
		#print(row['Name'])
		if row['Name'].__contains__(str):
			count += 1
			if pd.notnull(row['Age']):
				total_age += float(row['Age'])
				has_age_count += 1	
	return total_age/has_age_count

def calculate_all_mean_age(df):
	mean_age = {}
	mean_age['Mr.']=calculate_mean_age(df,'Mr.')
	mean_age['Mrs.']=calculate_mean_age(df,'Mrs.')
	mean_age['Master.']=calculate_mean_age(df,'Master.')
	mean_age['Miss.']=calculate_mean_age(df,'Miss.')
	mean_age['Dr.']=calculate_mean_age(df,'Dr.')
	return mean_age


def preprocess_df(df, source_df):
	# I need to convert all strings to integer classifiers.
	# I need to fill in the missing values of the data and make it complete.

	# female = 0, Male = 1
	df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

	# All the ages with no data -> make the median of all Ages
	#median_age = train_df['Age'].dropna().median()
	
	'''
	if len(df.Age[ df.Age.isnull() ]) > 0:
		all_mean_age = calculate_all_mean_age(source_df)	
		df.loc[ (df.Age.isnull()) & (df.Name.str.contains('Miss.')), 'Age'] = all_mean_age['Miss.']
		df.loc[ (df.Age.isnull()) & (df.Name.str.contains('Mrs.')), 'Age'] = all_mean_age['Mrs.']
		df.loc[ (df.Age.isnull()) & (df.Name.str.contains('Master.')), 'Age'] = all_mean_age['Master.']
		df.loc[ (df.Age.isnull()) & (df.Name.str.contains('Mr.')), 'Age'] = all_mean_age['Mr.']
		df.loc[ (df.Age.isnull()) & (df.Name.str.contains('Dr.')), 'Age'] = all_mean_age['Dr.']
	    #train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age
	'''

	#Title
	Title_list = pd.DataFrame(index = source_df.index, columns = ["Title"])
	Surname_list = pd.DataFrame(index = source_df.index, columns = ["Surname"])
	Name_list = list(source_df.Name)
	NL_1 = [elem.split("\n") for elem in Name_list]
	ctr = 0
	for j in NL_1:
	    FullName = j[0]
	    FullName = FullName.split(",")
	    Surname_list.loc[ctr,"Surname"] = FullName[0]
	    FullName = FullName.pop(1)
	    FullName = FullName.split(".")
	    FullName = FullName.pop(0)
	    FullName = FullName.replace(" ", "")
	    Title_list.loc[ctr, "Title"] = str(FullName)
	    ctr = ctr + 1


	#Title and Surname Extraction

	Title_Dictionary = {
	"Capt": "Officer",
	"Col": "Officer",
	"Major": "Officer",
	"Jonkheer": "Sir",
	"Don": "Sir",
	"Sir" : "Sir",
	"Dr": "Dr",
	"Rev": "Rev",
	"theCountess": "Lady",
	"Dona": "Lady",
	"Mme": "Mrs",
	"Mlle": "Miss",
	"Ms": "Mrs",
	"Mr" : "Mr",
	"Mrs" : "Mrs",
	"Miss" : "Miss",
	"Master" : "Master",
	"Lady" : "Lady"
	}    
	    
	def Title_Label(s):
	    return Title_Dictionary[s]

	source_df["Title"] = Title_list["Title"].apply(Title_Label)
	df["Title"] = Title_list["Title"].apply(Title_Label)


	## Filling missing Age data

	mask_Age = source_df.Age.notnull()
	Age_Sex_Title_Pclass = source_df.loc[mask_Age, ["Age", "Title", "Sex", "Pclass"]]
	Filler_Ages = Age_Sex_Title_Pclass.groupby(by = ["Title", "Pclass", "Sex"]).median()
	Filler_Ages = Filler_Ages.Age.unstack(level = -1).unstack(level = -1)
		
	def Age_filler(row):
		if pd.isnull(row['Age']):
		    if row.Sex == "female":
		        age = Filler_Ages.female.loc[row["Title"], row["Pclass"]]
		        return age
		    
		    elif row.Sex == "male":
		        age = Filler_Ages.male.loc[row["Title"], row["Pclass"]]
		        return age
		else:
			return row['Age']

	df["Age"]  = df.apply(Age_filler, axis = 1)   

	# Embarked from 'C', 'Q', 'S'
	# Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.

	# All missing Embarked -> just make them embark from most common place
	#if len(df.Embarked[ df.Embarked.isnull() ]) > 0:
	    #df.Embarked[ df.Embarked.isnull() ] = df.Embarked.dropna().mode().values
	#    df.loc[ (df.Embarked.isnull()),'Embarked' ] = df.Embarked.dropna().mode().values

	#Ports = list(enumerate(np.unique(df['Embarked'])))    # determine all values of Embarked,
	#Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
	#df.Embarked = df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

	# Embarked
	df['C'] = 0
	df['Q'] = 0
	df['S'] = 0
	df.loc[ (df.Embarked == 'C'),'C' ] = 1
	df.loc[ (df.Embarked == 'Q'),'Q' ] = 1
	df.loc[ (df.Embarked == 'S'),'S' ] = 1

	# Title
	df['Master'] = 0
	df['Miss'] = 0
	df['Mr'] = 0
	df['Mrs'] = 0
	df.loc[ (df.Name.str.contains('Master.')),'Master' ] = 1
	df.loc[ (df.Name.str.contains('Miss.')),'Miss' ] = 1	
	df.loc[ (df.Name.str.contains('Mr.')),'Mr' ] = 1
	df.loc[ (df.Name.str.contains('Mrs.')),'Mrs' ] = 1

	# Royalty
	df['Royalty'] = 0
	df.loc[ (df.Name.str.contains('Jonkheer.')),'Royalty' ] = 1
	df.loc[ (df.Name.str.contains('Don.')),'Royalty' ] = 1
	df.loc[ (df.Name.str.contains('Sir.')),'Royalty' ] = 1
	df.loc[ (df.Name.str.contains('theCountess')),'Royalty' ] = 1
	df.loc[ (df.Name.str.contains('Lady.')),'Royalty' ] = 1
	df.loc[ (df.Name.str.contains('Dona.')),'Royalty' ] = 1	

	# Officer
	df['Officer'] = 0
	df.loc[ (df.Name.str.contains('Dr.')),'Officer' ] = 1	
	df.loc[ (df.Name.str.contains('Rev.')),'Officer' ] = 1	
	df.loc[ (df.Name.str.contains('Capt.')),'Officer' ] = 1	
	df.loc[ (df.Name.str.contains('Officer.')),'Officer' ] = 1
	df.loc[ (df.Name.str.contains('Major.')),'Officer' ] = 1	

	#Pclass
	df['Pclass1'] = 0
	df['Pclass2'] = 0
	df['Pclass3'] = 0
	df.loc[ (df.Pclass == 1),'Pclass1' ] = 1
	df.loc[ (df.Pclass == 2),'Pclass2' ] = 2
	df.loc[ (df.Pclass == 3),'Pclass3' ] = 3

	#family size
	df['Singleton'] = 0
	df['SmallFamily'] = 0
	df['LargeFamily'] = 0
	df.loc[ (df.SibSp + df.SibSp +1 == 1), 'Singleton'] = 1
	df.loc[ ((df.SibSp + df.SibSp +1 <5) & (df.SibSp + df.SibSp +1 >1)), 'SmallFamily'] = 1
	df.loc[ (df.SibSp + df.SibSp +1 >4 ), 'LargeFamily'] = 1

	#Adult / Child
	df['Adult'] = 0
	df['Child'] = 0
	df.loc[ (df.Age >= 18),'Adult'] = 1
	df.loc[ (df.Age < 18),'Child'] = 1

	#Mother
	df['Mother'] = 0
	df.loc[((df.Gender==0) & (df.Parch >0) & (df.Age > 18) & (df.Miss==0)),'Mother']=1


	#Ticket
	def tix_clean(j):
	    j = j.replace(".", "")
	    j = j.replace("/", "")
	    j = j.replace(" ", "")
	    return j
    
	source_df[["Ticket"]] = source_df.loc[:,"Ticket"].apply(tix_clean)
	df[["Ticket"]] = df.loc[:,"Ticket"].apply(tix_clean)

	Ticket_count = dict(source_df.Ticket.value_counts())

	def Tix_ct(y):
	    return Ticket_count[y]

	df["TicketGrp"] = df.Ticket.apply(Tix_ct)

	def Tix_label(s):
	    if (s >= 2) & (s <= 4):
	        return 2
	    elif ((s > 4) & (s <= 8)) | (s == 1):
	        return 1
	    elif (s > 8):
	        return 0
	
	df["TicketGrp"] = df.loc[:,"TicketGrp"].apply(Tix_label)   	
	df["TicketGrp1"] = 0
	df["TicketGrp2"] = 0
	df["TicketGrp3"] = 0
	df.loc[ (df.TicketGrp == 0),'TicketGrp1' ] = 1
	df.loc[ (df.TicketGrp == 1),'TicketGrp2' ] = 1
	df.loc[ (df.TicketGrp == 2),'TicketGrp3' ] = 1

	# All the missing Fares -> assume median of their respective class
	if len(df.Fare[ df.Fare.isnull() ]) > 0:
	    median_fare = np.zeros(3)
	    for f in range(0,3):                                              # loop 0 to 2
	        median_fare[f] = df[ df.Pclass == f+1 ]['Fare'].dropna().median()
	    for f in range(0,3):                                              # loop 0 to 2
	        df.loc[ (df.Fare.isnull()) & (df.Pclass == f+1 ), 'Fare'] = median_fare[f]

	#logFare
	df['LogFare']=0
	df.LogFare = df.Fare.map( lambda x: math.log1p(x))


	# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
	df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId','Embarked','Title'], axis=1) 
	return df

def split_dataframe(df):
	df.sort_values(by='Name')
	df['Key']=df.index%10

	validation_df = pd.DataFrame(df.loc[ (df.Key >= 8)])
	train_df = pd.DataFrame(df.loc[ (df.Key < 8)])	
	validation_df = validation_df.drop(['Key'], axis=1)
	train_df = train_df.drop(['Key'], axis=1)
	return {'training_set':train_df, 'validation_set':validation_df}

def print_accuracy(test_df, test_ys):
	total_samples = len(test_df)
	correct_count = 0.0
	i = 0
	for index, row in test_df.iterrows():		
		if test_ys[i] == row['Survived']:
			correct_count += 1
		i+=1
	print('Accuracy=',round(100*correct_count/total_samples,2),'%')


print("preprocessing...")

# Data cleanup
# TRAIN DATA
train_df = pd.read_csv('../../train.csv', header=0)        # Load the train file into a dataframe
# TEST DATA
test_df = pd.read_csv('../../test.csv', header=0)        # Load the test file into a dataframe

train_set = train_df.drop("Survived", axis = 1)
df_combo = pd.concat((train_set, test_df), axis = 0, ignore_index = True)


test=False

# Parameters
learning_rate = 0.0001
training_epochs = 10000
min_cost = 20
display_step = 100
regularization_strength = 0.5
train_keep_prob = 0.5
train_keep_prob_input = 0.8

#features = ['Pclass1','Pclass2','Pclass3','Age','SibSp','Parch','LogFare','Gender','C','Q','S','Master','Miss','Mr','Mrs','Singleton','SmallFamily','LargeFamily','Child','Mother','Royalty','Officer','TicketGrp1','TicketGrp2','TicketGrp3']
features = ['Pclass1','Pclass2','Age','SibSp','Parch','LogFare','Gender','C','Q','Master','Miss','Mr','Mrs','Singleton','SmallFamily','Child','Mother','Royalty','Officer','TicketGrp1','TicketGrp2']

if test:
	print('Testing...')
	split_df = split_dataframe(train_df)
	train_df = split_df['training_set']
	validation_df = split_df['validation_set']	
	test_df = validation_df

train_df = preprocess_df(train_df, df_combo)

# Collect the test data's PassengerIds before dropping it
ids = test_df['PassengerId'].values
test_df = preprocess_df(test_df, df_combo)

# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
#train_data = train_df.values
#test_data = test_df.values

# write preprocessing function later

print('Normalize...')
# normalize data

train_df = (train_df)/(train_df.max())
test_df = (test_df)/(test_df.max())

print(train_df.describe())
print(train_df.shape)

# Reference
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3%20-%20Neural%20Networks/multilayer_perceptron.py

# Network Parameters
n_classes = 2 # MNIST total classes (0-9 digits)
input_add_output=len(features)+n_classes
n_hidden_1 = int(input_add_output*2/3) # 1st layer num features
n_hidden_2 = int(input_add_output*4/9) # 2nd layer num features
n_hidden_3 = int(input_add_output*8/27) # 3nd layer num features
n_input = len(features) # MNIST data input (img shape: 28*28)



# tf Graph input
x = tf.placeholder("float32", [None, n_input])
y = tf.placeholder("float32", [None, n_classes])
keep_prob1 = tf.placeholder(tf.float32)
keep_prob2 = tf.placeholder(tf.float32)
keep_prob3 = tf.placeholder(tf.float32)

# Create model
def multilayer_perceptron(_X, _weights, _biases):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) #Hidden layer with RELU activation
    layer_1_drop = tf.nn.dropout(layer_1, keep_prob1)
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1_drop, _weights['h2']), _biases['b2'])) #Hidden layer with RELU activation
    layer_2_drop = tf.nn.dropout(layer_2, keep_prob2)
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2_drop, _weights['h3']), _biases['b3'])) #Hidden layer with RELU activation        
    layer_3_drop = tf.nn.dropout(layer_3, keep_prob3)
    return tf.matmul(layer_3_drop, _weights['out']) + _biases['out']

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),    
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),    
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer
#optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate, l1_regularization_strength=regularization_strength).minimize(cost) 


# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph

batch_xs = train_df.as_matrix(columns=features)
batch_ys = train_df.as_matrix(columns=['Survived','Survived'])
for ys in batch_ys:
	ys[0] = 1-ys[0]

batch_test_xs = test_df.as_matrix(columns=features)
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    #for epoch in range(training_epochs):
    avg_cost = min_cost * 2
    epoch = 0
    while avg_cost > min_cost or epoch < training_epochs:
        avg_cost = 0.        
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob1: 1.0, keep_prob2: 1.0, keep_prob3: 1.0})
        # Compute average loss
        avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob1: train_keep_prob_input, keep_prob2: train_keep_prob, keep_prob3: train_keep_prob})
        # Display logs per epoch step
        if epoch % display_step == 0 or epoch == training_epochs-1:            
            if test:
            	print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            	prediction = tf.argmax(pred, 1)
            	test_ys = prediction.eval({x: batch_test_xs, keep_prob1: 1.0, keep_prob2: 1.0, keep_prob3: 1.0})
            	print_accuracy(test_df, test_ys)
            else:
            	print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
        epoch += 1

    print("Optimization Finished!")

    # Test model
    prediction = tf.argmax(pred, 1)        
    test_ys = prediction.eval({x: batch_test_xs, keep_prob1: 1.0, keep_prob2: 1.0, keep_prob3: 1.0})
    sess.close()

print("Result:", test_ys)

if len(test_df) >178 :

	predictions_file = open("neural_network.csv", "w")
	open_file_object = csv.writer(predictions_file)
	open_file_object.writerow(['PassengerId','Survived'])

	for index in range(0, len(test_ys)):
		open_file_object.writerow([ids[index],test_ys[index]])
		
	predictions_file.close()
else:
	print_accuracy(test_df, test_ys)

print('Done.')