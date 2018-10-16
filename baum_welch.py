from nltk.corpus import brown
import numpy as np
import math

O = brown.words()[: 500]
N = len(O)
vocab = list(set(O))
index = {}
for i, word in enumerate(vocab):
	index[word] = i
V = len(vocab)
M = 10

A = [np.random.dirichlet(np.ones(M)) for x in xrange(M)]
B = [np.random.dirichlet(np.ones(V)) for x in xrange(M)]
Pi = np.random.dirichlet(np.ones(M))
alphas = [[0.0 for i in xrange(N)] for j in xrange(M)]
betas = [[0.0 for i in xrange(N)] for j in xrange(M)]
gammas = [[0.0 for i in xrange(N)] for j in xrange(M)]
xis = [[[0.0 for k in xrange(N)] for i in xrange(M)] for j in xrange(M)]
coefs = [0.0 for i in xrange(N)]

def forward_algorithm():
	for t in xrange(N):
		coef = 0.0
		for i in xrange(M):
			if t == 0:
				alphas[i][t] = Pi[i] * B[i][index[O[0]]]

			else:
				sum_prev_alphas = 0.0
				for j in xrange(M):
					sum_prev_alphas += alphas[j][t - 1] * A[j][i]

				alphas[i][t] = B[i][index[O[t]]] * sum_prev_alphas
			coef += alphas[i][t]

		coefs[t] = coef
		for i in xrange(M):
			alphas[i][t] /= coef

def backward_algorithm():
	for t in xrange(N - 1, -1, -1):
		for i in xrange(M):
			if t == N - 1:
				betas[i][t] = 1

			else:
				sum_next_betas = 0.0
				for j in xrange(M):
					sum_next_betas += betas[j][t + 1] * A[i][j] * B[j][index[O[t + 1]]]

				betas[i][t] = sum_next_betas
		for i in xrange(M):
			betas[i][t] /= coefs[t]

def calc_gammas():
	for i in xrange(M):
		for t in xrange(N):
			numer = alphas[i][t] * betas[i][t]
			denom = 0.0
			for j in xrange(M):
				denom += alphas[j][t] * betas[j][t]

			gammas[i][t] = numer/denom

def calc_xis():
	for i in xrange(M):
		for j in xrange(M):
			for t in xrange(N - 1):
				numer = alphas[i][t] * A[i][j] * betas[j][t + 1] * B[j][index[O[t + 1]]]
				denom = 0.0
				for k in xrange(M):
					for l in xrange(M):
						denom += alphas[k][t] * A[k][l] * betas[l][t + 1] * B[l][index[O[t + 1]]]

				xis[i][j][t] = numer/denom

def update_Pi():
	for i in xrange(M):
		Pi[i] = gammas[i][1]

def update_A():
	for i in xrange(M):
		for j in xrange(M):
			numer = 0.0
			denom = 0.0
			for t in xrange(N - 1):
				numer += xis[i][j][t]
				denom += gammas[i][t]

			A[i][j] = numer/denom

def update_B():
	for i in xrange(M):
		for v in xrange(V):
			numer = 0.0
			denom = 0.0
			for t in xrange(N):
				if index[O[t]] == v:
					numer += gammas[i][t]
				denom += gammas[i][t]

			B[i][v] = numer/denom

def calc_likelihood():
	likelihood = 0.0
	#for i in xrange(M):
	#	likelihood += alphas[i][N - 1]
	for i in xrange(N):
		likelihood += math.log(1.0/coefs[i])

	return -1.0 * likelihood

converged = False
best_likelihood = 0.0
iteration = 0
while not converged and iteration < 7:
	forward_algorithm()
	backward_algorithm()
	calc_gammas()
	calc_xis()
	update_Pi()
	update_A()
	update_B()
	likelihood = calc_likelihood()
	if iteration == 0:
		best_likelihood = likelihood
	else:
		if likelihood > best_likelihood:
			best_likelihood = likelihood
		else:
			converged = True
	iteration += 1
	print likelihood

for i in xrange(M):
	top10 = sorted(range(len(B[i])), key=B[i].__getitem__, reverse=True)
	print "tag" + str(i + 1)
	for j in xrange(10):
		print vocab[top10[j]]

