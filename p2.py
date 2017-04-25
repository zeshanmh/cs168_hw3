import numpy as np
import matplotlib.pyplot as plt
import random

TRAIN_N = 100
TEST_N = 1000
D = 100
N_TRIALS = 10
N_ITERS = 1000000


def gen_data():
	X_train = np.random.normal(0,1, size=(TRAIN_N,D))
	a_true = np.random.normal(0,1, size=(D,1))
	y_train = X_train.dot(a_true) + np.random.normal(0,0.5,size=(TRAIN_N,1))
	X_test = np.random.normal(0,1, size=(TEST_N,D))
	y_test = X_test.dot(a_true) + np.random.normal(0,0.5,size=(TEST_N,1))

	return X_train, a_true, y_train, X_test, y_test

def compute_normalized_error(X, y, a):
	return np.linalg.norm(X.dot(a) - y) / np.linalg.norm(y)

def parta():
	X_train_errors = []
	X_test_errors = []
	for _ in range(N_TRIALS):
		X_train, a_true, y_train, X_test, y_test = gen_data()

		a = np.linalg.inv(X_train).dot(y_train)
		compute_normalized_error(X_train, y_train, a)
		normalized_train_error = compute_normalized_error(X_train, y_train, a)
		X_train_errors.append(normalized_train_error)
		normalized_test_error = compute_normalized_error(X_test, y_test, a)
		X_test_errors.append(normalized_test_error)

	print "Average Normalized Train Error: ", np.mean(X_train_errors)
	print "Average Normalized Test Error: ", np.mean(X_test_errors)

def l2_reg_closed_form(X, y, l):
	return np.linalg.inv(X.T.dot(X) + l * np.identity(D)).dot(X.T).dot(y)

def partb():
	lambdas = [0.0005, 0.005, 0.05, 0.5, 5, 50, 500]
	train_errors = []
	test_errors = []

	for l in lambdas:
		X_train_errors = []
		X_test_errors = []
		for _ in range(N_TRIALS):
			X_train, a_true, y_train, X_test, y_test = gen_data()

			a = l2_reg_closed_form(X_train, y_train, l)
			normalized_train_error = compute_normalized_error(X_train, y_train, a)
			X_train_errors.append(normalized_train_error)
			normalized_test_error = compute_normalized_error(X_test, y_test, a)
			X_test_errors.append(normalized_test_error)

		train_errors.append(np.mean(X_train_errors))
		test_errors.append(np.mean(X_test_errors))


	fig, ax = plt.subplots()
	plt.plot(xrange(len(lambdas)), train_errors, marker='x', color='b', label='Training Error')
	plt.plot(xrange(len(lambdas)), test_errors, marker='o', linestyle='-', color='r', label='Test Error')
	ax.set_xticklabels(lambdas)
	plt.xlabel('Lambda')
	plt.ylabel('Normalized Error')
	plt.title('Normalized Error vs. Lambda')
	plt.legend(loc=9)
	plt.show()

def compute_loss(X, y, a): 
	return np.sum((X.dot(a) - y)**2)

def sgd(X, y, alpha=0.0005): 
	a = np.zeros((D,1))
	rows = X.shape[0]
	#losses = []

	for i in xrange(N_ITERS): 
		#loss = compute_loss(X,y,a)
		#losses.append(loss)
		#if i % 100000 == 0: print 'loss at iteration %d: %f' % (i+1,loss)
		index = random.randint(0, rows - 1)
		x = X[index]
		grad = 2*(x.dot(a) - y[index])*x.T
		a = a - alpha * grad.reshape((-1,1))

	return a
	#return losses 

def partc():
	step_sizes = [0.00005, 0.0005, 0.005]
	train_errors = {}
	test_errors = {}
	for alpha in step_sizes:
		train_errors[alpha] = []
		test_errors[alpha] = []

	true_train_errors = []
	true_test_errors = []


	for i in range(N_TRIALS):
		print "Starting trial", i
		X_train, a_true, y_train, X_test, y_test = gen_data()

		for alpha in step_sizes:
			a = sgd(X_train, y_train, alpha)
			normalized_train_error = compute_normalized_error(X_train, y_train, a)
			train_errors[alpha].append(normalized_train_error)
			normalized_test_error = compute_normalized_error(X_test, y_test, a)
			test_errors[alpha].append(normalized_test_error)

		normalized_train_error = compute_normalized_error(X_train, y_train, a_true)
		true_train_errors.append(normalized_train_error)
		normalized_test_error = compute_normalized_error(X_test, y_test, a_true)
		true_test_errors.append(normalized_test_error)

	for alpha in step_sizes:
		print alpha, np.mean(train_errors[alpha]), np.mean(test_errors[alpha])
	print "true", np.mean(true_train_errors), np.mean(true_test_errors)


def sgd2(X, y, X_test, y_test, alpha): 
	a = np.zeros((D,1))
	rows = X.shape[0]

	train_errors = []
	test_errors = []
	norms = []

	for i in xrange(N_ITERS): 
		if i % 100000 == 0:
			print "iteration", i
		index = random.randint(0, rows - 1)
		x = X[index]
		grad = 2*(x.dot(a) - y[index])*x.T
		a = a - alpha * grad.reshape((-1,1))

		#normalized_train_error = compute_normalized_error(X, y, a)
		#train_errors.append(normalized_train_error)
		normalized_test_error = compute_normalized_error(X_test, y_test, a)
		test_errors.append(normalized_test_error)
		#l2_norm = np.linalg.norm(a)
		#norms.append(l2_norm)

	return train_errors, test_errors, norms


def partd():
	step_sizes = [0.00005, 0.005]
	train_errors = {}
	test_errors = {}
	norms = {}

	X_train, a_true, y_train, X_test, y_test = gen_data()
	for alpha in step_sizes:
		train_e, test_e, n = sgd2(X_train, y_train, X_test, y_test, alpha)
		#train_errors[alpha] = train_e
		test_errors[alpha] = test_e
		#norms[alpha] = n

	#true_train_error = compute_normalized_error(X_train, y_train, a_true)


	fig, ax = plt.subplots()
	plt.plot(xrange(1, N_ITERS + 1), test_errors[step_sizes[0]], color='b', label='alpha = 5e-5')
	plt.plot(xrange(1, N_ITERS + 1), test_errors[step_sizes[1]], color='r', label='alpha = 5e-3')
	#plt.axhline(true_train_error, color='g', label='true')
	plt.xlabel('Iteration')
	plt.ylabel('Normalized Test Error')
	plt.title('Normalized Test Error vs. Iteration')
	plt.legend(loc=2)
	plt.show()

def sgd3(X, y, r=0, alpha=0.00005): 
	a = np.random.normal(0, 1, size=(D,1))
	norm = np.linalg.norm(a)
	a = a * r / norm
	# a = np.zeros((D,1))
	rows = X.shape[0]
	#losses = []

	for i in xrange(N_ITERS): 
		#loss = compute_loss(X,y,a)
		#losses.append(loss)
		#if i % 100000 == 0: print 'loss at iteration %d: %f' % (i+1,loss)
		index = random.randint(0, rows - 1)
		x = X[index]
		grad = 2*(x.dot(a) - y[index])*x.T
		a = a - alpha * grad.reshape((-1,1))

	return a
	#return losses 

def parte():
	radii = [0, 0.1, 0.5, 1, 10, 20, 30]
	train_errors = []
	test_errors = []

	for r in radii:
		print "r: ", r
		X_train_errors = []
		X_test_errors = []
		
		for _ in range(N_TRIALS):
			X_train, a_true, y_train, X_test, y_test = gen_data()

			a = sgd3(X_train, y_train, r)
			normalized_train_error = compute_normalized_error(X_train, y_train, a)
			X_train_errors.append(normalized_train_error)
			normalized_test_error = compute_normalized_error(X_test, y_test, a)
			X_test_errors.append(normalized_test_error)

		train_errors.append(np.mean(X_train_errors))
		test_errors.append(np.mean(X_test_errors))


	fig, ax = plt.subplots()
	plt.plot(xrange(len(radii)), train_errors, marker='x', color='b', label='Training Error')
	plt.plot(xrange(len(radii)), test_errors, marker='o', linestyle='-', color='r', label='Test Error')
	ax.set_xticklabels(radii)
	plt.xlabel('Radius')
	plt.ylabel('Normalized Error')
	plt.title('Normalized Error vs. Radius')
	plt.legend(loc=9)
	plt.show()

def main():
	#parta()
	#partb()
	#partc()
	#partd()
	parte()

if __name__ == "__main__":
    main()