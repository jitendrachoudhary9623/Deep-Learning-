from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
'''
Basic Implementation of Perceptron

Utilizes  sum(w*x) >=threshold then y=1
'''

class Perceptron:
	def __init__(self):
		'''
		b is threshold
		w is weights
		'''
		self.b=None
		self.w=None

	def model(self,x):
		'''
		returns model for a given single input
		uses w*x 
		'''
		return 1 if( np.dot(self.w,x) >= self.b )else 0

	def predict(self,X):
		'''
		Returns Predicted values for provided dataset
		'''
		Y=[]
		for x in X:
			result=self.model(x)
			Y.append(result)
		return np.array(Y)
		

	def fit(self,X,Y,epochs=1):
		'''
		fits the model and find optimal weights and bias
		Utiliz	es learning algorithm
		Perceptron learning algorithm implementation
		'''

		#Initializing the weights to ones with the size of columns
		self.w=np.ones(X.shape[0])
		self.b=0
		accuracy={}
		for i in range(epochs):
			#for each col in dataset
			for x,y in zip(X,Y):
				y_pred=self.model(x)
				#for each misclassified point
				#if prediction is 1 then add else subract
				if y == 1 and y_pred == 0:
					self.w=self.w+x
					self.b=self.b+1
				elif y == 0 and y_pred == 1:
					self.w=self.w-x
					self.b=self.b-1
			accuracy[i]=accuracy_score(self.predict(X),Y)
			print("Accuracy at epoch {} is {}".format(i,accuracy[i]))
		plt.plot(accuracy.values())
		plt.show()
