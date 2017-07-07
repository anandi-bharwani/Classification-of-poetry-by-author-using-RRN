import numpy as np
from util import get_classifier_data, init_weights, classification_rate
import theano
import theano.tensor as T
from sklearn.utils import shuffle 

class RNN(object):
	def __init__(self, M, V):
		self.M = M
		self.V = V

	def fit(self, X, Y, lr=0.001, mu=0.99):
		M = self.M
		V = self.V
		K = len(set(Y))		#K = 2
		lr = np.float32(lr)
		mu = np.float32(mu)

		#Form train and test data set
		XTrain = X[:-50]
		YTrain = Y[:-50]
		XTest = X[-50:]
		YTest = Y[-50:]
		N = len(XTrain)
		print(Y)
		#Initial weights
		Wx = init_weights(V, M)
		Wh = init_weights(M, M)
		bh = np.zeros(M).astype(np.float32)
		h0 = np.zeros(M).astype(np.float32)
		Wo = init_weights(M, K)
		bo = np.zeros(K).astype(np.float32)

		#Theano Variables
		self.Wx = theano.shared(Wx)
		self.Wh = theano.shared(Wh)
		self.bh = theano.shared(bh)
		self.h0 = theano.shared(h0)
		self.Wo = theano.shared(Wo)
		self.bo = theano.shared(bo)

		self.params = [self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]	
#		self.dparams = [theano.shared(np.zeros(p.get_value().shape).astype(np.float32)) for p in self.params]

		thX = T.ivector('X')			#T size vector
		thY = T.iscalar('Y')			#Output, i.e, 0 for robert frost, 1 for edgar allan

		#Recurrence to loop through the input sequence
		def recurrence(x_t, h_t_prev):
			h_t = T.tanh(self.Wx[x_t] + h_t_prev.dot(self.Wh) + self.bh)
			y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
			return h_t, y_t

		[h,y],_ =  theano.scan(
				fn=recurrence,
				sequences=thX,
				n_steps=thX.shape[0],
				outputs_info=[self.h0,None],
				)

		#Prediction and cost calculation
		pY = y[-1,0,:]		#y is  T x 1 x K
		pred = T.argmax(pY)

		cost = -T.mean(T.log(pY[thY]))
		
		updates = [
			(p, p - lr*T.grad(cost,p)) for p in self.params
			]
# + [
#			(d, mu*d - lr*T.grad(cost,p)) for p,d in zip(self.params, self.dparams)
#			]

		#Training and prediction function
		train = theano.function(
			inputs=[thX, thY],
			updates=updates,
			outputs=pY                                
			)
		get_pred_cost = theano.function(
			inputs=[thX, thY],
			outputs=[pred, cost]
			)

		#Stochastic gradient descent
		for i in range(500):
			XTrain, YTrain = shuffle(XTrain,YTrain)
			lr = lr*0.9
			for n in range(N):
				x = XTrain[n]
				y = YTrain[n]

				p = train(x, y)
			#Test set
			n_correct = 0
			tot_c = 0
			for j in range(len(XTest)):
				p, c = get_pred_cost(XTest[j], YTest[j])
				if p==YTest[j]:
					n_correct+=1
				tot_c += c
			print("Iteration: ", i, "Cost: ", tot_c, "Classification rate: ", float(n_correct)/len(XTest))


def main():
	X,Y,V = get_classifier_data()
	print(len(X), len(Y), V)

	V+=2
	model = RNN(100, V)
	model.fit(X,Y)

if __name__=='__main__':
	main()
