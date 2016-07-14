import numpy as np

def validaGradientes(NN, x, y):
		paramsInicial = NN.getParams()
		print(paramsInicial)
		numgrad = np.zeros((len(paramsInicial),1))
		perturb = np.zeros((len(paramsInicial),1))
		epsilon = 0.00001
		
		for p in range(len(paramsInicial)):
			#Iniciando o vetor de perturbação
			perturb[p] = epsilon
			NN.setParams(paramsInicial+perturb)
			perda2 = NN.funcaoCusto(x,y)
			
			NN.setParams(paramsInicial-perturb)
			perda1 = NN.funcaoCusto(x,y)
			
			#computando o gradiente numérico
			numgrad[p] = (perda2-perda1) / (2*epsilon)
			perturb[p] = 0
		
		#Reajustando os params para o valor inicial
		NN.setParams(paramsInicial)
		return numgrad