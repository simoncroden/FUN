import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Implementation of SAGA algorithm for logistic regression
def SAGA(X, y, lam=0.5, eta=0.01, epochs=100,batch_size=100):
    n, m = X.shape
    theta = np.zeros(m)
    mu = 0.0

    grad_theta_mem = np.zeros((n, m))
    grad_mu_mem = np.zeros(n)
    grad_theta_avg = np.zeros(m)
    grad_mu_avg = 0.0

    obj_values = []

    for epo in range(epochs):
        indexes = np.random.permutation(n)
        for i in indexes[:batch_size]: 
            x_i = X[i]
            y_i = y[i]

            z = y_i * (np.dot(theta, x_i) + mu)
            g = -y_i / (1 + np.exp(z))

            grad_theta = g * x_i
            grad_mu = g

            v_theta = grad_theta - grad_theta_mem[i] + grad_theta_avg
            v_mu    = grad_mu    - grad_mu_mem[i]    + grad_mu_avg

            theta -= eta * (v_theta + lam * theta)
            mu    -= eta * v_mu

            grad_theta_avg += (grad_theta - grad_theta_mem[i]) / n
            grad_mu_avg    += (grad_mu    - grad_mu_mem[i]) / n

            grad_theta_mem[i] = grad_theta
            grad_mu_mem[i] = grad_mu

        loss = np.mean(np.log(1 + np.exp(-y * (X @ theta + mu)))) + lam / 2 * np.linalg.norm(theta)**2
        obj_values.append(loss)

    return obj_values

# Implementation of SAG algorithm for logistic regression
def SAG(X, y, lam=0.5, eta=0.01, epochs=100,batch_size=100):
    n, m = X.shape
    theta = np.zeros(m)
    mu = 0.0

    grad_theta_mem = np.zeros((n, m))
    grad_mu_mem = np.zeros(n)

    grad_theta_avg = np.zeros(m)
    grad_mu_avg = 0.0

    obj_values = []

    for epo in range(epochs):
        indexes = np.random.permutation(n)
        for i in indexes[:batch_size]: 
            x_i = X[i]
            y_i = y[i]

            grad_theta_avg -= grad_theta_mem[i] / n
            grad_mu_avg    -= grad_mu_mem[i] / n

            z = y_i * (np.dot(theta, x_i) + mu)
            g = -y_i / (1 + np.exp(z))

            grad_theta = g * x_i + lam * theta
            grad_mu = g

            grad_theta_mem[i] = grad_theta
            grad_mu_mem[i] = grad_mu

            grad_theta_avg += grad_theta / n
            grad_mu_avg    += grad_mu / n

            theta -= eta * grad_theta_avg
            mu    -= eta * grad_mu_avg

        loss = np.mean(np.log(1 + np.exp(-y * (X @ theta + mu)))) + lam / 2 * np.linalg.norm(theta)**2
        obj_values.append(loss)

    return obj_values

# Implementation of SGD algorithm for logistic regression
def SGD(X, y, lam=0.5, eta=0.01, epochs=100,batch_size=100):
    n, m = X.shape
    theta = np.zeros(m)
    mu = 0.0 

    obj_values = []

    for epo in range(epochs):
        indexes = np.random.permutation(n)
        for i in indexes[:batch_size]: 
            x_i = X[i]
            y_i = y[i]

            z = y_i * (np.dot(theta, x_i) + mu)
            g = -y_i / (1 + np.exp(z))

            theta -= eta * (g * x_i + lam * theta)
            mu -= eta * g

        loss = np.mean(np.log(1 + np.exp(-y * (X @ theta + mu)))) + lam/2 * np.linalg.norm(theta)**2
        obj_values.append(loss)

    return obj_values

# Function to import data from file
def data_import(file):
    X = []
    y = []

    with open(file) as f:
        for line in f:
            parts = line.split()
            
            y.append(int(parts[0]))
            
            x = np.zeros(14)
            for item in parts[1:]:
                idx, val = item.split(":")
                x[int(idx) - 1] = float(val)
            
            X.append(x)

    X = np.array(X)
    y = np.array(y)

    return y, X

# Main code to run the algorithms and plot results
y,X = data_import("australian_scale.txt")

# Run the algorithms
obj_sgd_SGD = SGD(X,y)
obj_sgd_SAG = SAG(X,y)
obj_sgd_SAGA = SAGA(X,y)

# Plot the objective values
plt.rcParams.update({'font.size': 30})
plt.plot(range(1, len(obj_sgd_SGD)+1), obj_sgd_SGD, label='SGD')
plt.plot(range(1, len(obj_sgd_SAG)+1), obj_sgd_SAG, label='SAG')
plt.plot(range(1, len(obj_sgd_SAGA)+1), obj_sgd_SAGA, label='SAGA')

plt.xlabel('Epoch')
plt.ylabel('Objective value')
plt.title('Objective vs Epoch')
plt.grid(True)
plt.legend()
plt.show()

# Analyze the effect of step size on the final objective value
eta = np.linspace(0.001, 1, 20)
obj_sgd_SGD_step = []
obj_sgd_SAG_step = []
obj_sgd_SAGA_step = []

for i in range(len(eta)):
    obj_sgd_SGD = SGD(X,y, eta=eta[i])
    obj_sgd_SAG = SAG(X,y, eta=eta[i])
    obj_sgd_SAGA = SAGA(X,y, eta=eta[i])

    obj_sgd_SGD_step.append(obj_sgd_SGD[-1])
    obj_sgd_SAG_step.append(obj_sgd_SAG[-1])
    obj_sgd_SAGA_step.append(obj_sgd_SAGA[-1])


plt.rcParams.update({'font.size': 30})
plt.plot(eta, obj_sgd_SGD_step, label='SGD')
plt.plot(eta, obj_sgd_SAG_step, label='SAG')
plt.plot(eta, obj_sgd_SAGA_step, label='SAGA')

plt.xlabel('step size')
plt.ylabel('Objective value')
plt.title('Objective vs step')
plt.grid(True)
plt.legend()
plt.show()
