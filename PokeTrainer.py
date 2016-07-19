from scipy import optimize
import numpy as np

class Treinador(object):
	def __init__(self, N):
		self.N = N
	
	def treinar(self, x, y, alpha, maxIter, epsilon):
		self.x = np.matrix(list(x.values()))
		self.y = np.matrix(list(y.values()))
		self.alpha = alpha
		self.maxIter = maxIter
		self.epsilon = epsilon
		
		#lista que armazena os custos
		self.J = []
		paramsInicial = self.N.getParams()
		W1 = paramsInicial[0]
		W2 = paramsInicial[1]
		bias = self.N.getBias()
		self.Et = self.N.funcaoCusto(x,y) / len(x)
		self.J.append(self.Et)
		numEpocas = 1
		
		print("ERRO TOTAL inicial", self.Et)
		
		while self.Et >= self.epsilon and numEpocas < self.maxIter:
			dJdW1, dJdW2 = self.N.derivadaCusto(x, y) # calcula os vetores gradiente 
			delta1 = [a_i * self.alpha for a_i in dJdW1]
			W1 = [w_i - delta_i for w_i, delta_i in zip(W1, delta1)] 
			delta2 = [b_i * self.alpha for b_i in dJdW2]
			W2 = [w_i - delta_i for w_i, delta_i in zip(W2, delta2)]
			bias = bias + self.alpha * self.Et 
			self.Et = self.N.funcaoCusto(x,y) / len(x)
			vect = np.concatenate((np.ravel(W1), np.ravel(W2)))
			self.N.setParams(vect)
			self.N.setBias(bias)
			self.J.append(self.Et)
			numEpocas = numEpocas + 1
			print("-------------------------------")
			print("Época: ", numEpocas)
			#print("Saida especulada:")
			#print(self.N.yEstimado)
			print("ERRO: ", self.Et)
			print("-------------------------------")
			
		print("ERRO total Final", self.Et)
		return self.N