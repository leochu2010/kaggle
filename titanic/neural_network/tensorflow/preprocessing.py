import tensorflow as tf
import pandas as pd
import numpy as np
import csv as csv
import math

print("preprocessing...")

# Data cleanup
# TRAIN DATA
train_df = pd.read_csv('../../train.csv', header=0)        # Load the train file into a dataframe


print(train_df.describe())

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

	def tix_clean(j):
	    j = j.replace(".", "")
	    j = j.replace("/", "")
	    j = j.replace(" ", "")
	    return j
    
	source_df[["Ticket"]] = source_df.loc[:,"Ticket"].apply(tix_clean)

	Ticket_count = dict(source_df.Ticket.value_counts())

	def Tix_ct(y):
	    return Ticket_count[y]

	df["TicketGrp"]=0
	df[["Ticket"]] = df.loc[:,"Ticket"].apply(tix_clean)	
	df["TicketGrp"] = df.Ticket.apply(Tix_ct)

	def Tix_label(s):
	    if (s >= 2) & (s <= 4):
	        return 2
	    elif ((s > 4) & (s <= 8)) | (s == 1):
	        return 1
	    elif (s > 8):
	        return 0

	df["TicketGrp"] = df.loc[:,"TicketGrp"].apply(Tix_label)   
	print(df["TicketGrp"])


	# All the ages with no data -> make the median of all Ages
	#median_age = train_df['Age'].dropna().median()
	if len(df.Age[ df.Age.isnull() ]) > 0:
		all_mean_age = calculate_all_mean_age(source_df)	
		df.loc[ (df.Age.isnull()) & (df.Name.str.contains('Miss.')), 'Age'] = all_mean_age['Miss.']
		df.loc[ (df.Age.isnull()) & (df.Name.str.contains('Mrs.')), 'Age'] = all_mean_age['Mrs.']
		df.loc[ (df.Age.isnull()) & (df.Name.str.contains('Master.')), 'Age'] = all_mean_age['Master.']
		df.loc[ (df.Age.isnull()) & (df.Name.str.contains('Mr.')), 'Age'] = all_mean_age['Mr.']
		df.loc[ (df.Age.isnull()) & (df.Name.str.contains('Dr.')), 'Age'] = all_mean_age['Dr.']
	    #train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age

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
	df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId','Embarked'], axis=1) 
	return df

# TRAIN DATA
train_df = pd.read_csv('../../train.csv', header=0)        # Load the train file into a dataframe
# TEST DATA
test_df = pd.read_csv('../../test.csv', header=0)        # Load the test file into a dataframe


#print(train_df.loc[ (train_df.Name.str.contains('Miss.')), 'Age'].dropna().mean())
#print(train_df.loc[ (train_df.Name.str.contains('Mrs.')), 'Age'].dropna().mean())
#print(train_df.loc[ (train_df.Name.str.contains('Master.')), 'Age'].dropna().mean())
#print(train_df.loc[ (train_df.Name.str.contains('Mr.')), 'Age'].dropna().mean())
#print(train_df.loc[ (train_df.Name.str.contains('Dr.')), 'Age'].dropna().mean())

print(train_df.isnull().sum())

test_df = pd.read_csv('../../test.csv', header=0)        # Load the train file into a dataframe
print(test_df.isnull().sum())

train_df['Key']=train_df.index%10

#print(train_df.loc[ (train_df.index > 800)])

test1_df = pd.DataFrame(train_df.loc[ (train_df.Key >= 8)])
train_df = pd.DataFrame(train_df.loc[ (train_df.Key < 8)])
print(test1_df.shape)
print(train_df.shape)

def modify_df(df):
	df['New']='NNN'

modify_df(test1_df)

#print(test1_df['New'])

train_df['C'] = 0
train_df['Q'] = 0
train_df['S'] = 0
train_df.loc[ (train_df.Embarked == 'C'),'C' ] = 1
train_df.loc[ (train_df.Embarked == 'Q'),'Q' ] = 1
train_df.loc[ (train_df.Embarked == 'S'),'S' ] = 1

print(train_df.describe())

train_df=train_df.sort(['Ticket'])
for index, row in train_df.iterrows():	
	#print(row['Age'],row['Pclass'],row['Fare'],row['Ticket'],row['SibSp'],row['Parch'])
	if row['Name'].__contains__('dona'):
		print(row['Name'])




## 
##         Capt          Col          Don         Dona           Dr 
##            1            4            1            1            8 
##     Jonkheer         Lady        Major       Master         Miss 
##            1            1            2           61          260 
##         Mlle          Mme           Mr          Mrs           Ms 
##            2            1          757          197            2 
##          Rev          Sir the Countess 
##            8            1            1
## Master, Miss, Mr, Mrs, Others


##group by ticket
##group max age, group min age, group avg age

train_df['logFare']=0
train_df.logFare = train_df.Fare.map( lambda x: math.log1p(x))     # Convert all Embark strings to int
train_df=train_df.sort(['Ticket'])
for index, row in train_df.iterrows():	
	#print(row['logFare'],row['Fare'])
	if row['Name'].__contains__('Col.'):
		print(row['Name'])

		
train_set = train_df.drop("Survived", axis = 1)
df_combo = pd.concat((train_set, test_df), axis = 0, ignore_index = True)

preprocess_df(train_df,df_combo)

