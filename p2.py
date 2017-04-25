import numpy as np
import matplotlib.pyplot as plt

train_n = 100
test_n = 1000
d = 100
n_trials = 10


def parta():
	X_train_errors = []
	X_test_errors = []
	for _ in range(n_trials):
		X_train = np.random.normal(0,1, size=(train_n,d))
		a_true = np.random.normal(0,1, size=(d,1))
		y_train = X_train.dot(a_true) + np.random.normal(0,0.5,size=(train_n,1))
		X_test = np.random.normal(0,1, size=(test_n,d))
		y_test = X_test.dot(a_true) + np.random.normal(0,0.5,size=(test_n,1))

		a = np.linalg.inv(X_train).dot(y_train)
		normalized_train_error = np.linalg.norm(X_train.dot(a) - y_train) / np.linalg.norm(y_train)
		X_train_errors.append(normalized_train_error)
		normalized_test_error = np.linalg.norm(X_test.dot(a) - y_test) / np.linalg.norm(y_test)
		X_test_errors.append(normalized_test_error)


	print "Average Normalized Train Error: ", np.mean(X_train_errors)
	print "Average Normalized Test Error: ", np.mean(X_test_errors)


def partb():
	lambdas = [0.0005, 0.005, 0.05, 0.5, 5, 50, 500]
	train_errors = []
	test_errors = []

	for l in lambdas:
		X_train_errors = []
		X_test_errors = []
		for _ in range(n_trials):
			X_train = np.random.normal(0,1, size=(train_n,d))
			a_true = np.random.normal(0,1, size=(d,1))
			y_train = X_train.dot(a_true) + np.random.normal(0,0.5,size=(train_n,1))
			X_test = np.random.normal(0,1, size=(test_n,d))
			y_test = X_test.dot(a_true) + np.random.normal(0,0.5,size=(test_n,1))

			a = np.linalg.inv(X_train.T.dot(X_train) + l * np.identity(d)).dot(X_train.T).dot(y_train)
			normalized_train_error = np.linalg.norm(X_train.dot(a) - y_train) / np.linalg.norm(y_train)
			X_train_errors.append(normalized_train_error)
			normalized_test_error = np.linalg.norm(X_test.dot(a) - y_test) / np.linalg.norm(y_test)
			X_test_errors.append(normalized_test_error)

		train_errors.append(np.mean(X_train_errors))
		test_errors.append(np.mean(X_test_errors))


	plt.plot(lambdas, test_errors, '-', label='lambda_vs_error')
	plt.xlabel('lambda')
	plt.ylabel('error')
	plt.title('train and test error for different lambdas')

	#plt.yscale('log')
	plt.legend(["step size=0.0005", "step size=0.005", "step size=0.01"], loc=2)
	plt.show()





def main():
	#parta()
	partb()

if __name__ == "__main__":
    main()