import numpy as np

class HMM():

    def __init__(self, Observations, Transition, Emission, Initial_distribution):
        self.Observations = Observations
        self.Transition = Transition
        self.Emission = Emission
        self.Initial_distribution = Initial_distribution

    
    def forward(self):

        alpha = np.zeros((self.Observations.shape[0], self.Transition.shape[0]))
        alpha[0, :] = self.Initial_distribution * self.Emission[:, self.Observations[0]] #first row

        for i in range(1, self.Observations.shape[0]):
            for j in range(self.Transition.shape[0]):
                a = alpha[i-1] @ self.Transition[:, j]
                alpha[i, j] = a * self.Emission[j, self.Observations[i]]

        return alpha
    
    def backward(self):
    
        beta = np.zeros((self.Observations.shape[0], self.Transition.shape[0]))
        beta[self.Observations.shape[0]-1] = np.ones((self.Transition.shape[0]))

        for i in range(self.Observations.shape[0]-2, -1, -1):
            for j in range(self.Transition.shape[0]):
                a = beta[i+1] * self.Emission[:, self.Observations[i+1]]
                beta[i, j] = a @ self.Transition[j, :]
                
        return beta
    
    def gamma_comp(self, alpha, beta):

        xi = np.moveaxis(hmm.xi_comp(alpha, beta, 0), 0, -1)
        gamma = np.sum(xi, axis=1) #calculating gamma from xi
        gamma = np.hstack((gamma, np.sum(xi[:, :, len(self.Observations) - 2], axis=0).reshape((-1, 1))))
        gamma = np.moveaxis(gamma, -1, 0) #reshaping back

        return gamma
    
    def xi_comp(self, alpha, beta, gamma):
        
        xi = np.zeros((self.Transition.shape[0], self.Transition.shape[0], len(self.Observations)-1))

        for j in range(len(self.Observations) - 1):
            temp = alpha[j, :].T @ self.Transition * self.Emission[:, self.Observations[j + 1]].T
            b = temp @ beta[j + 1, :]
            for i in range(self.Transition.shape[0]):
                a = alpha[j, i] * self.Transition[i, :] * self.Emission[:, self.Observations[j + 1]].T * beta[j + 1, :].T
                xi[i, :, j] = a / b
        xi = np.moveaxis(xi, -1, 0)

        return xi

    def update(self, alpha, beta, gamma, xi):

        new_init_state = np.zeros_like(self.Initial_distribution)
        T_prime = np.zeros_like(self.Transition)
        M_prime = np.zeros_like(self.Emission)
        xi = np.moveaxis(xi, 0, -1)
        gamma = np.moveaxis(gamma, 0, -1)
        
        T_prime = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
        b = np.sum(gamma, axis=1)
        for i in range(M_prime.shape[1]):
            M_prime[:, i] = np.sum(gamma[:, self.Observations == i], axis=1)
        M_prime = np.divide(M_prime, b.reshape((-1, 1)))
        new_init_state = gamma[:,0]

        return T_prime, M_prime, new_init_state
    
    def trajectory_probability(self, alpha, beta, T_prime, M_prime, new_init_state):

        P_original = np.array([0.])
        P_prime = np.array([0.])
        hmm_new = HMM(obs, T_prime, M_prime, new_init_state) #lambda prime
        alpha_new = hmm_new.forward()
        P_original = sum(alpha[-1,:])
        P_prime = sum(alpha_new[-1,:])
        return P_original, P_prime

# obs = np.array([2,0,0,2,1,2,1,1,1,2,1,1,1,1,1,2,2,0,0,1])
# T = np.array([[0.5,0.5], [0.5,0.5]])
# M = np.array([[0.4,0.1,0.5], [0.1,0.5,0.4]])
# pi = [0.5,0.5]
# hmm = HMM(obs, T, M, pi)

# alpha = hmm.forward()
# beta = hmm.backward()
# gamma = hmm.gamma_comp(alpha, beta)
# xi = hmm.xi_comp(alpha, beta, gamma)
# T_prime, M_prime, new_init_state = hmm.update(alpha, beta, gamma, xi)
# P_original, P_prime = hmm.trajectory_probability(alpha, beta, T_prime, M_prime, new_init_state)
# print("alpha: ", alpha)
# print("beta: ", beta)
# print("gamma: ", gamma)
# print("xi: ", xi)
# print("T Prime: ", T_prime)
# print("M_Prime: ", M_prime)
# print("new pi: ", new_init_state)
# print("P Original: ", P_original)
# print("P Prime: ", P_prime)