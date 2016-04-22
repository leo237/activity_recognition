import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fileName = 'data/concatenatedData1.pickle'
pickleFile = open(fileName,'r')
data = pickle.load(pickleFile)

print "pickled"

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x1 = []
y1 = []
z1 = []
x2 = []
y2 = []
z2 = []
x3 = []
y3 = []
z3 = []
x4 = []
y4 = []
z4 = []
x5 = []
y5 = []
z5 = []
x6 = []
y6 = []
z6 = []
x7 = []
y7 = []
z7 = []

for each in data:
	if each[4] == 1:
		x1.append(each[1])
		y1.append(each[2])
		z1.append(each[3])
	elif each[4] == 2:
		x2.append(each[1])
		y2.append(each[2])
		z2.append(each[3])
	elif each[4] == 3:
		x3.append(each[1])
		y3.append(each[2])
		z3.append(each[3])
	elif each[4] == 4:
		x4.append(each[1])
		y4.append(each[2])
		z4.append(each[3])
	elif each[4] == 5:
		x5.append(each[1])
		y5.append(each[2])
		z5.append(each[3])
	elif each[4] == 6:
		x6.append(each[1])
		y6.append(each[2])
		z6.append(each[3])
	elif each[4] == 7:
		x7.append(each[1])
		y7.append(each[2])
		z7.append(each[3])

x_acc = data[:,1]
y_acc = data[:,2]
z_acc = data[:,3]
output = data[:,4]

outputList = output.tolist()

print set(output)
print outputList.count(1)
print outputList.count(2)
print outputList.count(3)
print outputList.count(4)
print outputList.count(5)
print outputList.count(6)
print outputList.count(7)

mar = []
col = []
for each in output:
	if each == 1:
		mar.append('o')
		col.append('white')
	elif each == 2:
		mar.append('D')
		col.append('green')
	elif each == 3:
		mar.append('s')
		col.append('black')
	elif each == 4:
		mar.append('^')
		col.append('cyan')
	elif each == 5:
		mar.append('h')
		col.append('magenta')
	elif each == 6:
		mar.append('4')
		col.append('yellow')
	elif each == 7:
		mar.append('v')
		col.append('blue')

# ax.scatter(x1, y1, z1, c='red')
# ax.scatter(x2, y2, z2, c='green')
# ax.scatter(x3, y3, z3, c='blue')
# ax.scatter(x4, y4, z4, c='cyan')
# ax.scatter(x5, y5, z5, c='magenta')
# ax.scatter(x6, y6, z6, c='yellow')
# ax.scatter(x7, y7, z7, c='black')

ax.set_xlabel('X Acceleration')
ax.set_ylabel('Y Acceleration')
ax.set_zlabel('Z Acceleration')

plt.show()
