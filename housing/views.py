from django.shortcuts import render
from django.views import generic
from .models import Housing, Land
from django.views.generic.edit import CreateView, UpdateView, DeleteView
from django.views.generic import View
from django.http import HttpResponse
import os
from django.conf import settings
from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp
import pandas as pd
import numpy as np
import itertools
from . import REAL_ESTATE
from bokeh.plotting import figure, output_file, show
from bokeh.models import CustomJS, ColumnDataSource
from bokeh.embed import components
from bokeh.models import HoverTool
from bokeh.layouts import column
from bokeh.models.widgets import Panel, Tabs, Slider
from bokeh.io import curdoc
from bokeh.charts import TimeSeries
# Create your views here.

last=11
tvals=[["2012","2013","2014","2015","2016","2017"],["Kathmandu","Lalitpur","Bhaktapur","Nuwakot"],["Residental","Commercial","Agriculture"],["Pitch","Earthern","Goreto","Gravel"],["TEW","TEWS"],["High Tension","No High Tension"],["Near Stream","No Stream"],["Irregular","Rectangular","Triangular","Trapezoidal","L-Shaped"],["Plain","Slope","Higher","Lower"]]
vals=[["TWELVE","THIRTEEN","FOURTEEN","FIFTEEN","SIXTEEN","SEVENTEEN"],["Kathmandu","Lalitpur","Bhaktapur","Nuwakot"],["RE","CM","AG"],["P","E","GO","GR"],["TEW","TEWS"],["YES","NO"] ,["YES","NO"],["I","R","TR","TZ","L"],["P","S","H","L"]];
avals=[[100,5000,1000],[100,30,10,70],[70,100],[50,1000],[1,2],[50,200,20,150,60],[200,120,140,130]]
outputCats=[[100000,1],[1000000,100],[2000000,200],[3000000,300],[4000000,400],[5000000,500]]
Cats=[["A",500],["B",400],["C",300],["D",200],["E",100],["F",1]]
network=[]
minmax=[]

def IndexView(request):
	# input("easdnter")
	template_name = 'housing/index.html'
	context_object_name = 'all_housings'
	a={"y":0,"z":0,"a":0,"b":0,"c":0,"d":0,"e":0,"f":0,"g":0,"h":0,"i":0,"j":0,"k":0}
	return render(request,template_name,a)

def Predict(request):
	# post=PostForm()

	print(request.POST.get("a"),request.POST.get("b"),request.POST.get("c"),request.POST.get("d"),request.POST.get("e"),request.POST.get("f"),request.POST.get("g"),request.POST.get("h"),request.POST.get("i"),request.POST.get("j"),request.POST.get("k"))
	# print(request.P)
	# input("enter")
	# k=realstate()
	outp=PredictPrice(request.POST.get("a"),request.POST.get("b"),request.POST.get("c"),request.POST.get("d"),request.POST.get("e"),request.POST.get("f"),request.POST.get("g"),request.POST.get("h"),request.POST.get("i"),request.POST.get("j"),request.POST.get("k"))
	template_name = 'housing/index.html'

	a={"y":outp[0],"z":outp[1],"a":request.POST.get("a"),"b":request.POST.get("b"),"c":request.POST.get("c"),"d":request.POST.get("d"),"e":request.POST.get("e"),"f":request.POST.get("f"),"g":request.POST.get("g"),"h":request.POST.get("h"),"i":request.POST.get("i"),"j":request.POST.get("j"),"k":request.POST.get("k")}
	return render(request,template_name,a)

def PredictReg(request):
	# post=PostForm()
	# print(request.POST.get("Year"),request.POST.get("Road_width"));

	# input("enter")
	# outp=PredictPrice(request.POST.get("a"),request.POST.get("b"),request.POST.get("c"),request.POST.get("d"),request.POST.get("e"),request.POST.get("f"),request.POST.get("g"),request.POST.get("h"),request.POST.get("i"),request.POST.get("j"),request.POST.get("k"))
	template_name = 'housing/index.html'
	k = REAL_ESTATE.realstate(request.POST.get("Year"), request.POST.get("Road_width"),
							  request.POST.get("Location_Access"), request.POST.get("Government_rate"),
							  request.POST.get("Road_type"), request.POST.get("Land_type"))
	# print(k)
	res_test_list = k[3]
	res_train_list=k[4]
	predicted_test_list=k[5]
	predicted_train_list=k[6]
	y_actual_test_list=k[7]
	y_actual_train_list=k[8]




	b = {"aa": request.POST.get("Year"), "bb": request.POST.get("Road_width"),
		 "cc": request.POST.get("Location_Access"), "dd": request.POST.get("Government_rate"),
		 "ee": request.POST.get("Road_type"), "ff": request.POST.get("Land_type")}


	title = 'Residual vs Predicted price plot'
	hover1 = HoverTool(tooltips=[
		("(x,y)", "($x, $y)"),
	])

	hover2 = HoverTool(tooltips=[
		("(x,y)", "($x, $y)"),
	])
	plot1 = figure(title=title,
				  x_axis_label="Predicted value",
				  y_axis_label="residual",
				  tools=[hover1, "pan,wheel_zoom,box_zoom,reset,save"],
				  plot_width=500,
				  plot_height=350,
				  responsive=False,
				  toolbar_location='below',
				  logo=None)

	plot1.circle(predicted_test_list[1], res_test_list[0], line_width=1.5)
	plot2=figure(title="Predicted vs Actual price",
				  x_axis_label="Actual Price",
				  y_axis_label="Predicted value",
				  tools=[hover2, "pan,wheel_zoom,box_zoom,reset,save"],
				  plot_width=500,
				  plot_height=350,
				  responsive=False,
				  toolbar_location='below',
				  logo=None)
	#show(plot)
	plot2.circle(y_actual_test_list[1],predicted_test_list[1],line_width=1.5)
	script, div = components(plot1)
	script1,div1=components(plot2)

	a = {"zz": k[1][0], "ba": k[1][1], "bb": k[1][2], "bc": k[1][3], "bd": k[1][4], "be": k[1][5], "bf": k[2][0],
		 "bg": k[2][1],
		 "bh": k[2][2], "bi": k[2][3], "bj": k[2][4], "bk": k[2][5], "by": k[0],"script": script, "div": div,"script1":script1,"div1":div1 ,"aa": request.POST.get("Year"), "bb": request.POST.get("Road_width"),
		 "cc": request.POST.get("Location_Access"), "dd": request.POST.get("Government_rate"),
		 "ee": request.POST.get("Road_type"), "ff": request.POST.get("Land_type") }
	return render(request, template_name, a)







# outputCats=[[1,1],[10,10],[50,50],[250,250],[500,500],[1000,1000],[2500,2500],[5000,5000],[9000,9000]]

# Load a CSV file


def convertDate(dataset):
	count=0
	for i in range(0,len(dataset)):
		if(dataset[i][0]=="TWELVE"):
			dataset[i][11]=dataset[i][11]*1
		elif(dataset[i][10]=="THIRTEEN"):
			dataset[i][11]=dataset[i][11]*0.25
		elif(dataset[i][10]=="FOURTEEN"):
			dataset[i][11]=dataset[i][11]*0.125
		elif(dataset[i][10]=="FIFTEEN"):
			dataset[i][11]=dataset[i][11]*0.0625
		elif(dataset[i][10]=="SIXTEEN"):
			dataset[i][11]=dataset[i][11]*0.05
		else:
			dataset[i][11]=dataset[i][11]*0.0325
	for i in range(0,len(dataset)):
		if(dataset[i][0]=="TWELVE"):
			dataset[i][11]=dataset[i][12]*1
		elif(dataset[i][10]=="THIRTEEN"):
			dataset[i][11]=dataset[i][12]*0.25
		elif(dataset[i][10]=="FOURTEEN"):
			dataset[i][11]=dataset[i][12]*0.125
		elif(dataset[i][10]=="FIFTEEN"):
			dataset[i][11]=dataset[i][12]*0.0625
		elif(dataset[i][10]=="SIXTEEN"):
			dataset[i][11]=dataset[i][12]*0.05
		else:
			dataset[i][11]=dataset[i][12]*0.0325
		# print(count)
		# input("asd")

def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset




# Convert string column to float
def ff(a):
	for b in range(0,len(a)):
		g=0

		# print(b,a[b])
		g=a[b]-1

def fuzzOutput(dataset):
	for i in range(0,len(dataset)):
		c=1
		for k in range(0,len(outputCats)):
			if(dataset[i][12]>outputCats[k][0]):
				c=outputCats[k][1]
		dataset[i][12]=c

def fuzzonlyoutput(a):
	c=1
	for k in range(0,len(outputCats)):
		if(a>outputCats[k][0]):
			c=outputCats[k][1]
	return c

def div10000(dataset):
	for i in range(0,len(dataset)):
		dataset[i][12]=dataset[i][12]/1000

def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
 
# Convert string column to integer
def str_column_to_int(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
 
# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	d=0
	for column in zip(*dataset):
		d+=1
		ff(column)
		# print(d)
		# input('asd')
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats
 
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	count=0
	for row in dataset:
		count+=1
		# print(count)
		for i in range(0,len(row)):
			# print(len(row))
			# print(row[i],i,len(row),count)
			# input("row")
		
			# if(i==11)	:
			# print(row[i],minmax[i][0],minmax[i][1])
				# input('11')
			# print(row[i])
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
 			
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
 

def getNearestActual(p):
	for k in range(0,len(outputCats)-1):
		if(p >outputCats[k][1]*0.5 and p<outputCats[k+1][1]*0.75):
			# if(p< outputCats[k][1]*1.5 ):
			# 	return outputCats[k][1]
			return outputCats[k][1]
	return 500

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	error=0
	count=0
	correct=0
	for i in range(len(actual)):
		# print(i)
		# print((abs(actual[i] - predicted[i][0])/actual[i]))
		# print(actual[i],predicted[i][0])
		global minmax
		# print(minmax)
		# input("enter")

		a1=actual[i]*(minmax[12][1]-minmax[12][0])+minmax[12][0]
		p1=predicted[i][0]*(minmax[12][1]-minmax[12][0])+minmax[12][0]
		a=fuzzonlyoutput(a1)
		p=fuzzonlyoutput(p1)


		pp = getNearestActual(p);
		# print(pp,p,a)
		# input("asd")
		# print(a,p)
		
		# # input("asd")

		# if(abs(a-pp)>=150):
		# 	if(pp==500 or a==500):
		# 		oo=0
		# 	else:
		# 		count+=1

		if(a==pp):
			correct+=1
		elif(a+1>pp and a-1<pp):
			correct+=1
		elif(abs(a1-p1)/a1<0.5):
			correct+=1
		# else:
		# 	print(actual[i]*(minmax[12][1]-minmax[12][0])+minmax[12][0],predicted[i][0]*(minmax[12][1]-minmax[12][0])+minmax[12][0])
		# 	print(a,pp)
		# input("ASdasd")
	# print(count)
	# input("asd")
		# error+=abs(a1-p1)/a1
		# print(a,p)
		# input("enter")
		# print((abs(p-a)/a)*100)
	# print(len(actual))
	# input("accuracy")
	# print(actual[i],predicted[i])
	# print(error/ float(len(actual))*100)
	# input("e")
	# er= error /len(actual)*100
	return correct/len(actual)*100
	# res=list()
	# res.append(er)
	# res.append(cor)
	# return res
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	# print(len(folds))
	# input("asd")
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)

		actual = [row[-1] for row in fold]
		# print(actual)
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
		# print(actual)
		# print(predicted)
		# print(accuracy)
		# input("asdasd")
		
	return scores
 
# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation
 
# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))
 
# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs
 
# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)
 
# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	# print(len(network))
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				# print(len(layer),j)
				# input("Asd")
				
				for neuron in network[i + 1]:
					# print(j,neuron['weights'][0],len(layer))
					
					# print("ppp")
					# input("ppp")
					error += (neuron['weights'][j] * neuron['delta'])
				# input("lll")
				errors.append(error)
		else:
			
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
 
# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']
 
# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		count=0
		outliers=[]
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			# print("asdasdasd")
			# print(row[11],minmax[11][0],minmax[11])
			expected[0] = row[11] 
			# print(expected[0]-outputs[0])
			# input("Asd")
			# input("enter")
			count+=1
			# if(epoch==498):
			# 	print(count)
			# 	print(outputs,expected)
			# 	if(expected[0]!=0):
			# 		# print((abs(outputs[0]-expected[0])/expected[0])*100)
			# 		# input("asd")
			# 		if((abs(outputs[0]-expected[0])/expected[0])*100>100):
			# 			print("outlier")
			# 			outliers.append(count)
			
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		# if(epoch==498):
		# 	print(len(outliers))
		# 	input("Asd")
		# 	print(outliers)
		# 	input("asdasd")
			
			
# Initialize a network
def initialize_network(n_inputs, n_hidden,n_outputs):
	network = list()
	hidden_layer = [{'weights':[0.01 for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[0.01 for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network
 
# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	
	# print(outputs)
	# input("predict")
	return outputs
 

# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
	n_inputs = len(train[0]) - 1
	n_outputs = len(set([row[-1] for row in train]))
	# network = initialize_network(11, n_hidden, 1)

	# print(network)
	# input("asdasd")
	train_network(network, train, l_rate, n_epoch, 1)
	predictions = list()
	for row in test:
		prediction = predict(network, row)
		# print(prediction)
		# input("Asd")
		predictions.append(prediction)
	return(predictions)

def minn(dataset):
	last=len(dataset[0])-1
	k=float(dataset[0][last].strip())
	for i in range(0,len(dataset)):
		if(k>float(dataset[i][last].strip())):
			k=float(dataset[i][last].strip())
	return k

def maxx(dataset):
	last=len(dataset[0])-1
	k=float(dataset[0][last].strip())
	for i in range(0,len(dataset)):
		if(k<float(dataset[i][last].strip())):
			k=float(dataset[i][last].strip())
	return k



def getVal(key,val,mn,mx,dataset):
	cms=[]
	cms= [x for x in dataset if x[key] == val]
	# print(cms,key,val)
	# input("cms")

	total=0.0
	for n in range(0,len(cms)):
		total+=cms[n][12]
	print(total,mx,mn,key,val)
	# input("enter")
	return total/len(cms);
 

def categorialToNumerical(dataset,mn,mx,vals):
	for i in range(0,9):
		for j in range(0,len(vals[i])):
			val=getVal(i,vals[i][j],mn,mx,dataset)
			# v=getVal(0,"RE",mn,mx,dataset)
			for k in range(0,len(dataset)):
				if(dataset[k][i]==vals[i][j]):
					dataset[k][i]=val
		# print(vals[i])

def preConvert(dataset):
	for i in range(0,len(dataset)):
		# print(i)
		dataset[i][9]=float(dataset[i][9].strip())
		dataset[i][10]=float(dataset[i][10].strip())
		dataset[i][11]=float(dataset[i][11].strip())
		dataset[i][12]=float(dataset[i][12].strip())

def convertDate1(dataset):
	count=0
	for i in range(0,len(dataset)):
		print(dataset[i][11])
		if(dataset[i][0]=="TWELVE"):
			dataset[i][11]=dataset[i][11]*1
		elif(dataset[i][10]=="THIRTEEN"):
			dataset[i][11]=dataset[i][11]*0.25
		elif(dataset[i][10]=="FOURTEEN"):
			dataset[i][11]=dataset[i][11]*0.125
		elif(dataset[i][10]=="FIFTEEN"):
			dataset[i][11]=dataset[i][11]*0.0625
		elif(dataset[i][10]=="SIXTEEN"):
			dataset[i][11]=dataset[i][11]*0.05
		else:
			dataset[i][11]=float(dataset[i][11])*0.0325

# Test Backprop on Seeds dataset

def PredictPrice(d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11):
	seed(1)
	# load and prepare data
	#realstate(5.7,10,200,100000,'Pitch','Commercial')
	filename ="static/assets/3_date_govrate_district.csv" 
	file_path = os.path.join(settings.STATIC_ROOT, '3_date_govrate_district.csv')
	# file_ = open(os.path.join(PROJECT_ROOT, filename))
	dataset = load_csv(file_path)

	mn=minn(dataset)
	mx=maxx(dataset)

	preConvert(dataset)


	categorialToNumerical(dataset,mn,mx,vals)

	# for d in dataset:
	# 	print(d)
	# 	input("enter")

	# print(dataset)
	# input("enter")

	# for i in range(len(dataset[0])-1):
		# str_column_to_float(dataset, i)

	# # convert class column to integers
	# str_column_to_int(dataset, len(dataset[0])-1)
	# normalize input variables
	# for i in range(0,len(dataset)):
	# 	if(dataset[i][1] in vals[j]):
	# 		print(dataset[i][1],i)
	# div10000(dataset)

	# getVal(0,"a",mn,mx,dataset)

	# print(minmax)
	# input("asd")
	convertDate(dataset)

	# fuzzOutput(dataset)
	# print(dataset)

	global minmax 
	minmax = dataset_minmax(dataset)
	# print(minmax)

	normalize_dataset(dataset, minmax)
	# evaluate algorithm
	# print(dataset)
	# print("data")
	# input("asdasd ")
	n_folds = 2
	l_rate = 0.4
	n_epoch = 10
	n_hidden = 22
	n_hidden1= 20
	n_hidden2=20
	n_hidden3=20
	n_hidden4=20


	global network
	# for i in range(1,3):
	network = initialize_network(11,22,1)
		# print(i)

			
		# print(i)
	scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
	# o=forward_propagate(network,data[0])
	# print(o)
	# print(o[0]*(minmax[11][1]-minmax[11][0])+minmax[11][0])
	# input("eter")
	print('\n')
	print('Training Metrics')
	print('\n')
	print('K fold Scores: %s' % scores)
	print('Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
	print('Confidence: 50%' )
	print('\n')
	
	
	dataset = load_csv(file_path)
	preConvert(dataset)



	data=[["SEVENTEEN",d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11]]
	tdata=list()
	for i in range(0,9):
		for v in range(0,len(vals[i])):
			if(data[0][i]==vals[i][v]):
				tdata.append(tvals[i][v])


	for i in range(0,9):
		data[0][i]=getVal(i,data[0][i],mn,mx,dataset)
	# print(data[0][0])
	# input("asd")
	data[0][9]=float(data[0][9])
	data[0][10]=float(data[0][10])
	convertDate1(data)



	# print(dataset)


	categorialToNumerical(dataset,mn,mx,vals)
	convertDate(dataset)

	fuzzOutput(dataset)
	# print(dataset)
	minmax = dataset_minmax(dataset)
	print(data)
	# input("enter")
	normalize_dataset(data, minmax)
	output=forward_propagate(network,data[0])
	# print(output)
	d=output[0]*(mx-mn)+mn
	print("\n")
	print("predicted price: ",d)
	fo=fuzzonlyoutput(d)

	for o in Cats:
		if(o[1]==fo):
			fo=o[0]
	print("Predicted Category: ",fo)
	outp=[]
	outp.append(fo)
	outp.append(d)
	return outp