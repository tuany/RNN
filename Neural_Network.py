import numpy as np
class Neural_Network(object):
	def __init__(self, tamInput, tamOutput, tamCamadaEsc):
		#Definindo os metaparâmetros da rede
		self.tamInput = tamInput 
		self.tamOutput = tamOutput
		self.tamCamadaEsc = tamCamadaEsc
		
		#Inicializando os pesos da rede: W1 - Pesos da primeira camada (escondida) / W2 - Pesos da segunda camada (Saída)
		self.W1 = np.random.randn(self.tamInput,self.tamCamadaEsc)
		self.W2 = np.random.randn(self.tamCamadaEsc,self.tamOutput)
		
	def sigmoide(self, z):
		#Função de ativação - Sigmóide -
		return 1/(1+np.exp(-z))
		
	def derivadaSigmoide(self, z):
		return np.exp(-z)/((1+np.exp(-z))**2)
		
	def funcaoCusto(self, x, y):
		#Calcula o erro para uma entrada X e o y real
		self.yEstimado = self.propaga(x)
		dif = y-self.yEstimado
		J = (np.sum(np.power(dif, 2))) * 0.5
		return J
		
	def derivadaCusto(self, x, y):
		#Calcula a derivada em função de W1 e W2
		self.yEstimado = self.propaga(x)
		#y = np.reshape(y, (len(y),self.tamOutput))
		#erro a ser retropropagado
		# rever essa parte do sinal negativo
		#print(y)
		#print(self.yEstimado)
		ek = -(y-self.yEstimado)
		delta3 = np.multiply(ek, self.derivadaSigmoide(self.yin))
		#Obtém o erro a ser retropropagado de cada camada, multiplicando pela derivada da função de ativação
		dJdW2 = np.dot(self.zin.T, delta3)

		delta2 = np.dot(delta3, self.W2.T)*self.derivadaSigmoide(self.z)
		dJdW1 = np.dot(x.T, delta2)
		return dJdW1, dJdW2
	
	def propaga(self, *X):
		#propaga as entradas através da estrutura da rede
		#multiplica a matriz de entradas "x" pela de pesos "w1"
		# for i in X:
		# 	self.z[i] = np.dot(X[i], self.W1)
		# 	self.zin[i] = self.sigmoide(self.z[i])
		#x = np.reshape(x, (len(x),self.tamInput))
		#print(" Esse é o formato certo do x", x, np.shape(x))
		self.z = np.dot(x, self.W1)
		#print(self.z) 
		#Aplica a função de ativação para o vetor z2, a saída é da mesma dimensão que a entrada
		self.zin = self.sigmoide(self.z)
		
		#Multiplica a saída da camada do meio pelos pesos da ultima camada
		self.yin = np.dot(self.zin, self.W2)
		
		#aplica a função de ativação do neuronio de saida
		yEstimado = self.sigmoide(self.yin)
		return yEstimado
	
	#funções auxiliares
	
	def getParams(self):
		#concatena os pesos das camadas em um vetor
		#params = np.concatenate((self.W1.ravel(),self.W2.ravel()))
		return self.W1.ravel(), self.W2.ravel()

	def setParams(self, params):
		#ajusta os pesos de acordo com o vetor de pesos passado
		W1_inicio = 0
		W1_fim = self.tamCamadaEsc * self.tamInput
		self.W1 = np.reshape(params[W1_inicio:W1_fim], (self.tamInput, self.tamCamadaEsc))
		W2_fim = W1_fim + self.tamCamadaEsc * self.tamOutput
		self.W2 = np.reshape(params[W1_fim:W2_fim], (self.tamCamadaEsc, self.tamOutput))

	def computaGradientes(self, x, y):
		dJdW1, dJdW2 = self.derivadaCusto(x, y)
		return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))
		
		