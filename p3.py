import numpy as np


TRAIN_N = 100
TEST_N = 10000
D = 200
N_TRIALS = 1000
NUM_ITERS = 100


def gen_data():
	X_train = np.random.normal(0,1, size=(TRAIN_N,D))
	a_true = np.random.normal(0,1, size=(D,1))
	y_train = X_train.dot(a_true) + np.random.normal(0,0.5,size=(TRAIN_N,1))
	X_test = np.random.normal(0,1, size=(TEST_N,D))
	y_test = X_test.dot(a_true) + np.random.normal(0,0.5,size=(TEST_N,1))

	return X_train, a_true, y_train, X_test, y_test

def compute_normalized_error(X, y, a):
	return np.linalg.norm(X.dot(a) - y) / np.linalg.norm(y)

def compute_loss(X, y, a): 
	return np.sum((X.dot(a) - y)**2)

def gradient_descent(X, y, alpha=0.0005, l=0.1, verbose=False): 
	a = np.zeros((D,1))
	losses = []
	for i in xrange(NUM_ITERS): 
		# loss = np.log(compute_loss(X,y,a))
		loss = compute_loss(X,y,a)
		losses.append(loss)
		if verbose:
			print 'loss at iteration %d: %f' % (i+1,loss)
		grad = np.sum((2*(X.dot(a) - y))*X,axis=0)
		grad = grad.reshape((-1, 1)) + 2*l*a
		a = a - alpha * grad
	
	return a


def main():
	train_errors = []
	test_errors = []

	v = True
	for n in range(N_TRIALS):
		X_train, a_true, y_train, X_test, y_test = gen_data()

		if n == 1:
			v = False
		a = gradient_descent(X_train, y_train, verbose=v)

		normalized_train_error = compute_normalized_error(X_train, y_train, a)
		train_errors.append(normalized_train_error)
		normalized_test_error = compute_normalized_error(X_test, y_test, a)
		test_errors.append(normalized_test_error)
		if n < 10:
			print normalized_train_error, normalized_test_error

	print "normalized train error:", np.mean(normalized_train_error)
	print "normalized test error:", np.mean(normalized_test_error)


if __name__ == '__main__':
	main()

