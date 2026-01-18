from lifelines.datasets import load_dd
from lifelines import KaplanMeierFitter
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import lomax
from scipy.stats import gamma
import seaborn as sns
import pandas as pd

def kaplan_meier_plot(T):
    kmf = KaplanMeierFitter()

    ax = plt.subplot(111)

    kmf.fit(T)
    kmf.plot_survival_function(ax=ax)

def exp_plot(T):
    lam = (np.mean(T))
    print(1/lam)

    x = np.linspace(0, 40, 400)

    y =  np.exp(-(1/lam) * x)

    plt.plot(x,y, label="Exp")


def handmade_kaplan_meier(T):
    T_sorted = np.sort(T)
    n = len(T_sorted)

    S = [1]
    d = []
    t = [0]
    ni = []

    unique, counts = np.unique(T_sorted, return_counts=True)

    for idx,i in enumerate(T_sorted):
        if i not in t:
            t.append(i)
            d.append(counts[np.where(unique == i)[0][0]])
            ni.append(n-idx)
            S.append(S[-1]*(1 - d[-1]/ni[-1]))

    N = []

    for idx in range(1,len(S)): 
        N.append(-np.log(S[idx]/S[idx-1]))

    N = N[:-1]
    #plt.plot(t[1:],N, label="Kaplan–Meier")
    #plt.hist(N, bins=20, label="Kaplan–Meier")
    lam = 1/(np.mean(T))
    
    plt.plot(t[1:], lam * np.ones(len(t[1:])), c='green')


    plt.plot(t[1:-1], N, label="Kaplan–Meier", c='orange')
    #values, base = np.histogram(N)
    #evaluate the cumulative
    #cumulative = np.cumsum(values)
    # plot the cumulative function
    #plt.plot(base[:-1], cumulative, c='blue')

    plt.xlabel("Time")
    plt.ylabel("Survival probability")
    plt.title("Kaplan–Meier Survival (No Censoring)")
    plt.grid(True)

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


def lomax_mle(T):
    unique, counts = np.unique(np.sort(T), return_counts=True)
    t = np.concatenate(([0], unique))
    data = T

    def nll(params):
        alpha, lam = params
        if alpha <= 0 or lam <= 0:
            return np.inf
        
        values = np.zeros(len(T))

        for idx in range(1, len(T)):
            if idx == 1:
                values[idx] = np.log((1+1*lam)**(-alpha)-(1+T[idx]*lam)**(-alpha))
            else:
                values[idx] = np.log((1+T[idx-1]*lam)**(-alpha)-(1+T[idx]*lam)**(-alpha))

        return -np.sum(values)

    init = np.array([1.0, 5.0])
    res = minimize(nll, init, method='L-BFGS-B', bounds=[(1e-6, None), (1e-6, None)])
    alpha_hat, lambda_hat = res.x

    print("Estimated α (shape):", alpha_hat)
    print("Estimated λ (scale):", lambda_hat)

    S = lomax.sf(t, c=alpha_hat, scale=lambda_hat)

    plt.plot(t, S, label="Lomax Survival Function")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.title("Lomax Survival Function with MLE Parameters")
    plt.grid(True)

    return alpha_hat, lambda_hat


def gamma_plot(alpha_hat, lambda_hat):
    shape = 1/alpha_hat
    scale = lambda_hat

    x = np.linspace(0, 1, 500)
    pdf = gamma.pdf(x, a=shape, scale=scale)

    plt.plot(x, pdf, label="Gamma(α={}, scale={:.2f})".format(shape, scale))
    plt.xlabel("Rate λ")
    plt.ylabel("Density")
    plt.title("Gamma distribution underlying Lomax")
    plt.legend()


data = load_dd()
T = data["duration"]

#kaplan_meier_plot(T)
#alpha_hat, lambda_hat = lomax_mle(T)
#plt.show()
#handmade_kaplan_meier(T)
#exp_plot(T)
#plt.legend()


#gamma_plot(alpha_hat, lambda_hat)
#plt.show()

def opt_mle(T):
    unique, counts = np.unique(np.sort(T), return_counts=True)
    t = np.concatenate(([0], unique))
    data = T

    T_sorted = np.sort(T)
    n = len(T_sorted)

    S = [np.float64(1)]
    d = []
    t = [0]
    ni = []

    unique, counts = np.unique(T_sorted, return_counts=True)

    for idx,i in enumerate(T_sorted):
        if i not in t:
            t.append(i)
            d.append(counts[np.where(unique == i)[0][0]])
            ni.append(n-idx)
            S.append(S[-1]*(1 - d[-1]/ni[-1]))

    def nll(params):
        alpha, lam, w = params
        N = 100
        
        values = np.zeros(len(T))

        for idx in range(1, len(S)-1):
            values[idx] = np.log(np.exp(-w*lam*(idx-1))*(1+(1-w)*lam*(idx-1)/alpha)**(-alpha) - np.exp(-w*lam*(idx))*(1+(1-w)*lam*(idx)/alpha)**(-alpha))*N*(S[idx-1]-S[idx])
            #values[idx] = np.log((1+(idx-1)/lam)**(-alpha) - (1+idx/lam)**(-alpha))*N*(S[idx-1]-S[idx])
        return -np.sum(values)

    """"
    def nll(params):
        alpha, lam = params
        if alpha <= 0 or lam <= 0:
            return np.inf
        
        values = np.zeros(len(T))

        for idx in range(1, len(T)):
            values[idx] = np.log((1+T[idx-1]*lam)**(-alpha)-(1+T[idx]*lam)**(-alpha))

        return -np.sum(values)
    """
    init = np.array([1.0, 5.0,1])

    res = minimize(nll, init, method='L-BFGS-B', bounds=[(1e-6, None), (1e-6, None),(1e-6, None)])
    alpha_hat, lambda_hat,w = res.x

    print("Estimated α (shape):", alpha_hat)
    print("Estimated λ (scale):", lambda_hat)
    print("Estimated w (scale):", w)


    return alpha_hat, lambda_hat, w



data = load_dd()
T = data["duration"]


#alpha_hat, lambda_hat,w = opt_mle(T)

kaplan_meier_plot(T)

plt.show()