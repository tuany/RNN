import csv
import os

f = csv.reader(open(os.getcwd()+'\IBOVESPA.csv'), delimiter=',')
valorFechamento = []
for linha in f:
	valorFechamento.append(float(linha[4]))
	
print(valorFechamento)
Y = valorFechamento[:]

print(valorFechamento.pop(-1))
print(Y.pop(0))
	

for i in zip(valorFechamento, Y):
	print(i)