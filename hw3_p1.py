import numpy as np 
import matplotlib.pyplot as plt

D = 100 
N = 1000 
NUM_ITERS = 20

def gen_data(): 
	np.random.seed(1234)
	X = np.random.normal(0,1,size=(N,D))
	a_true = np.random.normal(0,1,size=(D,1))
	y = X.dot(a_true) + np.random.normal(0,0.5,size=(N,1))

	return X, y, a_true

def closed_form(X, y): 
	return (np.linalg.pinv(X.T.dot(X)).dot(X.T)).dot(y)

def compute_loss(X, y, a): 
	return np.sum((X.dot(a) - y)**2)

def gradient_descent(X, y, alpha=0.00005): 
	a = np.zeros((D,1))
	losses = []
	for i in xrange(NUM_ITERS): 
		# loss = np.log(compute_loss(X,y,a))
		loss = compute_loss(X,y,a)
		losses.append(loss)
		print 'loss at iteration %d: %f' % (i+1,loss)
		grad = np.sum((2*(X.dot(a) - y))*X,axis=0)
		a = a - alpha * grad.reshape((-1,1))
	
	return losses

def sgd(X, y, alpha=0.0005): 
	a = np.zeros((D,1))
	losses = []

	for i in xrange(N): 
		loss = compute_loss(X,y,a)
		losses.append(loss)
		if i % 1 == 0: print 'loss at iteration %d: %f' % (i+1,loss)
		x = X[i]
		grad = 2*(x.dot(a) - y[i])*x.T
		a = a - alpha * grad.reshape((-1,1))

	return losses 

def plot(losses, step):
	lw = 1.0 
	plt.plot(range(len(losses)), losses, '-', label='loss_vs_time' + str(step))
	plt.xlabel('num iterations')
	plt.ylabel('loss')
	plt.title('objective function value over time (gradient descent)')

if __name__ == '__main__':
	X, y, a_true = gen_data()

	a = closed_form(X,y)
	print "loss with closed form a:", compute_loss(X, y, a)
	print "loss with all zeros a:", compute_loss(X, y, np.zeros((D,1)))

	opt = sgd
	losses_step1 = opt(X,y,alpha=0.0005)
	losses_step2 = opt(X,y,alpha=0.005)
	losses_step3 = opt(X,y,alpha=0.01)

	plot(losses_step1, step=1)
	plot(losses_step2, step=2)
	plot(losses_step3, step=3)

	plt.yscale('log')
	plt.legend(["step size=0.00005", "step size=0.0005", "step size=0.0007"], loc=2)
	plt.show()