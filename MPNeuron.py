from sklearn.metrics import accuracy_score

class MPNeuron:
	'''
	Basic Implementation of MPNeuron model
	'''
	def __init__(self):
		'''
		Constructor
		b is threshold value for prediction
		'''
		self.b=None

	def model(self,x):
		'''
		x=input data
		returns output for each column
		'''
		return (sum(x)>=self.b)

	def predict(self,X):
		'''
		X is dataset
		returns predicted values
		'''
		Y=[]
		for x in X:
			result=self.model(x)
			Y.append(result)
		return np.array(Y)

	def fit(self,X,Y):
		'''
		X training columsn
		Y labels for training colums
		fit the model based on the dataset provided
		'''
		accuracy={}
		for b in range(X.shape[1]):
			self.b=b
			Y_pred=sel.predict(X)
			accuracy[b]=accuracy_score(Y_pred,Y)
		best_b=max(accuracy,key=accuracy.get)
		self.b=best_b
		print("Max Accuracy obtained is {} with b={}".format(accuracy[best_b],best_b))
