[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-github-username>/<your-repo-name>/blob/master/<notebook-name>.ipynb)

def loaddata():
	"""
	Loads the csv file into a list of lists
	:param filename: Name of the .csv file with quotations

	:returns: a list of lists
	"""
	import csv
	import numpy

	reader=csv.reader(open('dermatology.csv','r'))
	patients=[]

	for r in reader:
		patients.append(r)

	return patients

patients=loaddata()
import numpy
patients=numpy.asarray(patients)

def viz1(dataset,col):
	"""
	Six boxplots comparing column data for each type of skin lesion

	:param arr: A 2D numpy array 
	:param col: An integer denoting the column of interest in the array

	:returns: Six vertical boxplots 
	"""
	import numpy as np 
	import matplotlib.pyplot as plt
	data=dataset.astype(np.int)
	psoriasis=[] # list containing coloumn data for psoriasis
	seboreic=[] # list containing data for seboreic dermatitis
	lichen=[] 
	pityriasis_rose=[] 
	cronic=[]
	pityriasis_rubra=[]
	for row in data:  # for the indicated coloumn place the values into their respective list based on skin lesion type
		if row[34]==1:
			psoriasis.append(row[col])
		if row[34]==2:
			seboreic.append(row[col])
		if row[34]==3:
			lichen.append(row[col])
		if row[34]==4:
			pityriasis_rose.append(row[col])
		if row[34]==5:
			cronic.append(row[col])
		if row[34]==6:
			pityriasis_rubra.append(row[col])

	X=[psoriasis,seboreic,lichen, pityriasis_rose,
	cronic, pityriasis_rubra] # sequence of all six lists

	plt.boxplot(X)
	plt.show()


def viz2(dataset,col):
	"""
	Creates a histogram from column data from a 2D numpy array

	:param dataset: A 2D numpy array
	:param col: An integers denoting the coloumn of interest

	:returns: A histogram of coloumn data
	"""
	import numpy as np
	import matplotlib.pyplot as plt
	data=dataset.astype(np.int) # convert from an array of strings to an array of integers
	plt.hist(data[:,col]) 
	plt.show()


def learn1(data):
	""" 
	Implements a supervised algorithm of K-nearest neighbors to classify
	a 2D numpy array on the basis of labels from the last column in the array (6 different disease
	types). Divides the whole dataset into a test and train segments. Data is fit to the train data
	and scored on the test data

	:param data: A 2D numpy array with 35 coloumns

	:returns: A score of how well the algorithm classifies the given 2D numpy array on the basis of
	the 35th coloumn
	"""

	from sklearn.neighbors import KNeighborsClassifier
	import numpy as np
	lesion_data=data.astype(np.int) # convert from an array of strings to an array of integers
	knn = KNeighborsClassifier()
	from sklearn.model_selection import train_test_split # splits the data in a test and train sample 
	X_train, X_test, y_train, y_test = train_test_split(lesion_data, lesion_data[:,34], test_size=0.33, random_state=42)
	knn.fit(X_train,y_train) # fit the calssifer to the train dataset
	print (knn.predict(X_test)) # predict the test sample
	return knn.score(X_test, y_test)
	

def learn2(data):
	"""
	Implements an unsupervised k-means clustering algorithm to classify data 
	:param data: A 2D numpy array of data

	:returns: A 3D scatter plot of data
	"""
	import numpy as np
	learning_data=data.astype(np.int)

	data_nolabels=learning_data[:,:34]

	from sklearn import cluster
	k_means = cluster.KMeans(6)
	k_means.fit(data_nolabels)
	print (k_means.labels_)


	import numpy as np
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D

	from sklearn.cluster import KMeans
	

	X = data_nolabels
	y = learning_data[:,34]

	estimators = [('k_means_diseases_6', KMeans(n_clusters=6))]

	fignum = 1
	titles = ['6 clusters']
	for name, est in estimators:
	    fig = plt.figure(fignum, figsize=(4, 3))
	    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
	    est.fit(X)
	    labels = est.labels_

	    ax.scatter(X[:,1],X[:, 0], X[:, 2],
	               c=labels.astype(np.int), edgecolor='k')

	    ax.w_xaxis.set_ticklabels([])
	    ax.w_yaxis.set_ticklabels([])
	    ax.w_zaxis.set_ticklabels([])
	    ax.set_title(titles[fignum - 1])
	    ax.dist = 12
	    fignum = fignum + 1

	fig.show()


def part2():
	"""
	Loads all the functions from part2 of the assignment by loading the csv data, visualizations
	and machine learning systems

	:returns: Outputs from 5 functions

	"""
	loaddata()
	viz1(patients,12) # test for the attribute of eosinophils in the infiltrate
	viz2(patients,34) # produce a histogram of the incidences of each skin lesion classification
	print (learn1(patients))
	return learn2(patients)

