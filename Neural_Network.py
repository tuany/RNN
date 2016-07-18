import numpy as np
class Neural_Network(object):
	def __init__(self, tamInput, tamOutput, tamCamadaEsc, lambdaVal):
		# Definindo os metaparâmetros da rede
		self.tamInput = tamInput 
		self.tamOutput = tamOutput
		self.tamCamadaEsc = tamCamadaEsc
		self.lambdaVal = lambdaVal
		
		# Inicializando os pesos da rede: W1 - Pesos da primeira camada (entrada) / W2 - Pesos da segunda camada (Saída)
		self.W1 = np.random.randn(self.tamInput, self.tamCamadaEsc)

		self.W2 = np.random.randn(self.tamCamadaEsc, self.tamOutput)
		# reshape
		#self.W1 = np.reshape(self.W1, (len(self.W1), self.tamCamadaEsc))
		#self.W2 = np.reshape(self.W2, (len(self.W2), self.tamOutput))

	def sigmoide(self, z):
		#Função de ativação - Sigmóide -
		return 1/(1+np.exp(-z))
		
	def derivadaSigmoide(self, z):
		return np.exp(-z)/(np.power(1+np.exp(-z), 2))
		
	def funcaoCusto(self, x, y):
		# Calcula o erro para uma entrada X e o y real
		self.yEstimado = self.propaga(x)
		dif = y-self.yEstimado
		J = ((np.sum(np.power(dif, 2))) * 0.5)/x.shape[0] + (self.lambdaVal/len(x)) * (np.sum(np.power(self.W1,2)) + np.sum(np.power(self.W2,2))) 
		return J
		
	def derivadaCusto(self, x, y):
		# Calcula a derivada em função de W1 e W2
		self.yEstimado = self.propaga(x)
		# y = np.reshape(y, (len(y),self.tamOutput))
		# erro a ser retropropagado
		# rever essa parte do sinal negativo
		ek = -np.subtract(y.T, self.yEstimado)
		print("Erro k:")
		print(ek.shape)
		print(ek)

		print("Derivada sigmoid YIN : ", self.derivadaSigmoide(self.yin).shape)
		print(self.derivadaSigmoide(self.yin))

		delta3 = np.multiply(ek, self.derivadaSigmoide(self.yin))
		# Obtém o erro a ser retropropagado de cada camada, multiplicando pela derivada da função de ativação
		#adicionando o termo de regularização no gradiente (+lambda * pesos)
		dJdW2 = np.dot(self.zin.T, delta3) + self.lambdaVal*self.W2
		print("dJdW2 ------------ ", dJdW2.shape)
		print(dJdW2)
		print("Z shape:", self.z.shape)
		print(self.z)
		print("Derivada Z shape:", self.derivadaSigmoide(self.z).shape)
		print(self.derivadaSigmoide(self.z))
		print("W2 shape", self.W2.shape)
		print(self.W2)
		delta2 = np.multiply(np.dot(delta3, self.W2.T), self.derivadaSigmoide(self.z))
		#adicionando o termo de regularização no gradiente (+lambda * pesos)
		dJdW1 = np.dot(x, delta2) + self.lambdaVal*self.W1
		return dJdW1, dJdW2
	
	# propaga as entradas através da estrutura da rede
	def propaga(self, X):
		# multiplica a matriz de entradas "x" pela de pesos "w1"
		self.z = np.dot(X.T, self.W1)
		# função de ativação
		self.zin = self.sigmoide(self.z)

		# multiplica a saída da camada do meio pelos pesos da ultima camada
		self.yin = np.dot(self.zin, self.W2)
		
		# aplica a função de ativação do neuronio de saida
		yEstimado = self.sigmoide(self.yin)

		return yEstimado
	
	# funções auxiliares
	
	def getParams(self):
		# concatena os pesos das camadas em um vetor
		# params = np.concatenate((self.W1.ravel(),self.W2.ravel()))
		return self.W1.ravel(), self.W2.ravel()

	def setParams(self, params):
		# ajusta os pesos de acordo com o vetor de pesos passado
		W1_inicio = 0
		W1_fim = self.tamCamadaEsc * self.tamInput
		self.W1 = np.reshape(params[W1_inicio:W1_fim], (self.tamInput, self.tamCamadaEsc))
		W2_fim = W1_fim + self.tamCamadaEsc * self.tamOutput
		self.W2 = np.reshape(params[W1_fim:W2_fim], (self.tamCamadaEsc, self.tamOutput))

	def computaGradientes(self, x, y):
		dJdW1, dJdW2 = self.derivadaCusto(x, y)
		return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))
		
		