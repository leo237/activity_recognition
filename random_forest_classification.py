import numpy as np 
import pandas as pd 
import csv as csv
from sklearn.ensemble import RandomForestClassifier
import pickle


train = [2,3,4]
validate = 1

data = []
for i in train:
	fileName = '/Users/Leo/Desktop/clean/finalData/crossValidationPart'+str(i) +'.pickle'
	print "filename ", fileName
	pickledFile = open(fileName,'r')
	tempData = pickle.load(pickledFile)
	tempData = tempData.astype(np.float32)
	print tempData.shape
	tempData = tempData.tolist()
	data = data+tempData
	pickledFile.close()

data = np.asarray(data)
print data.shape

validationSetFileName = fileName = '/Users/Leo/Desktop/clean/finalData/crossValidationPart'+str(validate) +'.pickle'
pickledFile = open(fileName,'r')
validationData = pickle.load(pickledFile)
validationData = validationData.astype(np.float32)

input_data = data[:,1:4]
label_data = data[:,4].astype(int)

validate_input_data = validationData[:,1:4]
validate_label_data = validationData[:,4].astype(int)

###########################################################

rf = RandomForestClassifier(n_estimators=10,criterion="gini",n_jobs=2,verbose=4,max_features="auto")
rf.fit(input_data, label_data)

###########################################################
#Pickle rf
saveRF = open('/Users/Leo/Desktop/clean/rf_model_n_estimators_10.pickle','w')
pickle.dump(rf, saveRF)
saveRF.close()

###########################################################

output = rf.predict(validate_input_data)
print output
print validate_label_data

#################################################################
correct = 0
total = len(output)

for i in xrange(total):
	if output[i] == validate_label_data[i]:
		correct+=1

print "Correct : " +  str(correct)
print "Total : " + str(total)

accuracy = float(correct)/float(total)

print "Accuracy : " + str(accuracy)


