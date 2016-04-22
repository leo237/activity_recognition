import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fileName = '/Users/Leo/Desktop/clean/data/concatenatedData.pickle'
pickleFile = open(fileName,'r')
data = pickle.load(pickleFile)

print data[0]
np.random.shuffle(data)
print data[0]

numberOfParts = 5
allParts = []

sizeOfEachPart = int(data.shape[0]/5)
previous = 0

for i in xrange(1,numberOfParts):
	individualPart = data[previous:i*sizeOfEachPart,]
	print individualPart.shape
	allParts.append(individualPart)
	previous = i*sizeOfEachPart
	print previous
allParts.append(data[previous:,])

for i in xrange(1,numberOfParts):
	fileName = '/Users/Leo/Desktop/clean/finalData/crossValidationPart'+str(i)+'.pickle'
	pickledFile = open(fileName,'w')
	pickle.dump(allParts[i-1],pickledFile)
	pickledFile.close()

fileName = '/Users/Leo/Desktop/clean/finalData/testData.pickle'
pickledFile = open(fileName,'w')
pickle.dump(allParts[-1], pickledFile)
pickledFile.close()


fileName = '/Users/Leo/Desktop/clean/finalData/smallDataForTestingHyperparameters.pickle'
pickledFile=open(fileName,'w')
pickle.dump(allParts[0][:100,], pickledFile)
pickledFile.close()

