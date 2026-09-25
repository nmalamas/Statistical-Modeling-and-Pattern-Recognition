import numpy as np
import matplotlib.pyplot as plt

# Class samples
samples = {
    'ω1': np.array([[0.1, 6.8, -3.5, 2, 4.1, 3.1, -0.8, 0.9, 5, 3.9],
                    [1.1, 7.1, -4.1, 2.7, 2.8, 5, -1.3, 1.2, 6.4, 4]]),
    'ω2': np.array([[7.1, -1.4, 4.5, 6.3, 4.2, 1.4, 2.4, 2.5, 8.4, 4.1],
                    [4.2, -4.3, 0, 1.6, 1.9, -3.2, -4, -6.1, 3.7, -2.2]]),
    'ω3': np.array([[-3, 0.5, 2.9, -0.1, -4, -1.3, -3.4, -4.1, -5.1, 1.9],
                    [-2.9, 8.7, 2.1, 5.2, 2.2, 3.7, 6.2, 3.4, 1.6, 5.1]]),
    'ω4': np.array([[-2, -8.9, -4.2, -8.5, -6.7, -0.5, -5.3, -8.7, -7.1, -8],
                    [-8.4, 0.2, -7.7, -3.2, -4, -9.2, -6.7, -6.4, -9.7, -6.3]])
}

colors = {'ω1': 'r', 'ω2': 'g', 'ω3': 'b', 'ω4': 'm'}


def plot_samples(samples, title):
    for key in samples:
        plt.scatter(samples[key][0, :], samples[key][1, :], c=colors[key], label=key)
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

plot_samples(samples, 'Class Samples')

def batch_perceptron(samples, class1, class2):
    X1 = samples[class1]
    X2 = samples[class2]

    # Combining the data
    X = np.hstack((X1, X2))
    y = np.hstack((np.ones(X1.shape[1]), -np.ones(X2.shape[1])))

    # Weight initialization
    w = np.zeros(X.shape[0] + 1) 
    X = np.vstack((np.ones(X.shape[1]), X)) 


    epochs = 1000 #arithmos epanalipseon
    learning_rate = 1.0 
    num_updates = 0

    for epoch in range(epochs):
       for i in range(X.shape[1]):
            if y[i] * np.dot(w, X[:, i]) <= 0: 
                w += learning_rate * y[i] * X[:, i]
                num_updates += 1

    # Decision boundary
    x1 = np.linspace(-10, 10, 100)
    x2 = -(w[0] + w[1] * x1) / w[2]

    #plot_samples({class1: samples[class1], class2: samples[class2]}, f'{class1} vs {class2}')
    #plt.plot(x1, x2, 'k--', label='Decision Boundary')
    #plt.legend()
    #plt.show()

    plt.figure()
    plt.scatter(X1[0, :], X1[1, :], c=colors[class1], label=class1)
    plt.scatter(X2[0, :], X2[1, :], c=colors[class2], label=class2)
    plt.plot(x1, x2, 'k--', label='Decision Boundary')
    plt.title(f'{class1} vs {class2}')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()
    return num_updates 
    

num_updates_12 = batch_perceptron(samples, 'ω1', 'ω2')
num_updates_23 = batch_perceptron(samples, 'ω2', 'ω3')
num_updates_34 = batch_perceptron(samples, 'ω3', 'ω4')

print(f'Number of updates for ω1 vs ω2: {num_updates_12}')
print(f'Number of updates for ω2 vs ω3: {num_updates_23}')
print(f'Number of updates for ω3 vs ω4: {num_updates_34}')
