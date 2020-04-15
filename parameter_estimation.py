import numpy as np
import pickle
import glob
import time
import matplotlib.pyplot as plt

import densities
import estimation


n_iter = 20000

tuning_parameters = {
    'g_tau': 1.1,  # tau prior
    'h_tau': 0.1,  # tau prior
    'a_tau': 10,  # tau proposal
    'b_tau': 0.1,  # tau proposal scale param
    'g_alpha': 20,  # alpha prior
    'h_alpha': 0.1,  # alpha prior
    'a_alpha': 10,  # proposal
    'b_alpha': 0.1,  # proposal
    'g_beta': 30,  # prior
    'h_beta': 2,  # prior
    'a_beta': 6,  # proposal
    'b_beta': 0.2,  # proposal
    'g_sigma': 25,  # prior
    'h_sigma': 0.1,  # prior
    'a_sigma': 10,  # proposal
    'b_sigma': 0.1,  # proposal
    'nu_p': 0.25,  # proposal
    'nu_p_tilde': 0.4  # proposal
}


class InitialiseParameterEstimation:

    def __init__(self, filename, n_iterations):
        self.original_chain = self.get_original_decision_chain(filename)
        self.z = self.original_chain.z
        self.p_data = self.initialise_p(n_iterations)
        self.x_data = self.initialise_x(n_iterations)
        self.y_data = self.initialise_y(n_iterations)
        self.p_tilde_data, self.p_tilde_save_data = self.initialise_p_tilde(n_iterations)

    def get_original_decision_chain(self, filename):
        file = open(filename, 'rb')
        decision_chain = pickle.load(file)
        return decision_chain

    def initialise_p(self, n_iterations):
        self.p_data = np.zeros((n_iterations, len(self.z)))
        p_initial_guess = np.ones(len(self.z)) * 0.51
        self.p_data[0] = p_initial_guess
        return self.p_data

    def initialise_x(self, n_iterations):
        self.x_data = np.zeros(n_iterations)
        self.x_data[0] = np.random.binomial(1, 0.5)
        return self.x_data

    def initialise_y(self, n_iterations):
        self.y_data = np.zeros((n_iterations, len(self.z)), dtype=int)
        self.y_data[0] = np.random.binomial(1, 0.5, len(self.z))
        return self.y_data

    def initialise_p_tilde(self, n_iterations):
        self.p_tilde_data = np.zeros((n_iterations, len(self.z) - 1, len(self.z) - 1))
        self.p_tilde_save_data = self.p_tilde_save_data = np.zeros((n_iterations, 3))
        p_tilde_initial = densities.p_tilde_given(self.p_data[0], sigma=1)
        self.p_tilde_data[0] = p_tilde_initial
        self.p_tilde_save_data[0, 0] = self.p_tilde_data[0, 0, 0]
        self.p_tilde_save_data[0, 1] = self.p_tilde_data[0, 10, 5]
        self.p_tilde_save_data[0, 2] = self.p_tilde_data[0, 30, 20]
        return self.p_tilde_data, self.p_tilde_save_data


def get_decision_data(path, n_iterations):
    object_files = np.sort(glob.glob(path))
    decision_data = np.zeros(len(object_files), dtype=object)
    for i in range(len(object_files)):
        decision_data[i] = InitialiseParameterEstimation(object_files[i], n_iterations)
    return decision_data


def run_mh_sampler(tuning_parameters, n_iter, save_data=True):
    n_chains = 4
    assert n_iter % n_chains == 0, "Number of chains not compatible with number of iterations."

    decision_data = get_decision_data("simulated_data/n_50/*.obj", n_iter // n_chains)
    #  Initialise tau
    tau_data = np.zeros(len(decision_data) * n_iter - 3)
    tau_data[0] = 0.05

    #  Initialise sigma
    sigma_data = np.zeros(len(decision_data) * n_iter - 3)
    sigma_data[0] = 2

    # Initialise alpha
    alpha_data = np.zeros(len(decision_data) * n_iter - 3)
    alpha_data[0] = 3

    # Initialise beta
    beta_data = np.zeros(len(decision_data) * n_iter - 3)
    beta_data[0] = 70

    index = 1
    decision_data_index = 1
    print(f"Running MH-sampler with {len(decision_data)} decision chains and for n={n_iter} iterations.")
    print(f"Parameters:\n{tuning_parameters}")

    start = time.time()
    for i in range(1, n_iter):
        for j in range(len(decision_data)):
            if index % 100 == 0:
                print(f"Iteration {index} out of {n_iter * n_chains}")
            #  Update sigma
            sigma_data[index] = sigma_data[index - 1]
            sigma_proposal = sigma_data[index] * np.random.gamma(shape=tuning_parameters['a_sigma'],
                                                                 scale=tuning_parameters['b_sigma'])
            mh_ratio = estimation.sigma_mh_ratio(sigma_data[index], sigma_proposal,
                                                 decision_data[j].p_tilde_data[decision_data_index - 1],
                                                 decision_data[j].p_data[decision_data_index - 1],
                                                 tuning_parameters['a_sigma'],
                                                 tuning_parameters['b_sigma'], tuning_parameters['g_sigma'],
                                                 tuning_parameters['h_sigma'])
            if np.log(np.random.uniform(0, 1)) <= mh_ratio:
                sigma_data[index] = sigma_proposal

            #  Update tau
            tau_data[index] = tau_data[index - 1]
            tau_proposal = tau_data[index] * np.random.gamma(shape=tuning_parameters['a_tau'],
                                                             scale=tuning_parameters['b_tau'])
            tau_mh_ratio = estimation.tau_mh_ratio(tau_data[index], tau_proposal, decision_data[j].z,
                                                   decision_data[j].p_data[decision_data_index - 1],
                                                   decision_data[j].p_tilde_data[decision_data_index - 1],
                                                   decision_data[j].y_data[decision_data_index - 1],
                                                   tuning_parameters['g_tau'],
                                                   tuning_parameters['h_tau'], tuning_parameters['a_tau'],
                                                   tuning_parameters['b_tau'])
            if np.random.uniform(0, 1) <= tau_mh_ratio:
                tau_data[index] = tau_proposal

            #  Update alpha
            alpha_data[index] = alpha_data[index - 1]
            alpha_proposal = alpha_data[index] * np.random.gamma(shape=tuning_parameters['a_alpha'],
                                                                 scale=tuning_parameters['b_alpha'])
            alpha_mh_ratio = estimation.alpha_mh_ratio(alpha_data[index], alpha_proposal, beta_data[index - 1],
                                                       tuning_parameters['g_alpha'],
                                                       tuning_parameters['h_alpha'], tuning_parameters['a_alpha'],
                                                       tuning_parameters['b_alpha'],
                                                       decision_data[j].p_data[decision_data_index - 1])

            if np.log(np.random.uniform(0, 1)) <= alpha_mh_ratio:
                alpha_data[index] = alpha_proposal

            #  Update beta
            beta_data[index] = beta_data[index - 1]
            beta_proposal = beta_data[index] * np.random.gamma(shape=tuning_parameters['a_beta'],
                                                               scale=tuning_parameters['b_beta'])

            beta_mh_ratio = estimation.beta_mh_ratio(beta_data[index], beta_proposal, alpha_data[index],
                                                     tuning_parameters['g_beta'], tuning_parameters['h_beta'],
                                                     tuning_parameters['a_beta'],
                                                     tuning_parameters['b_beta'],
                                                     decision_data[j].p_data[decision_data_index - 1])
            if np.log(np.random.uniform(0, 1)) <= beta_mh_ratio:
                beta_data[index] = beta_proposal

            if i % n_chains == 0:
                decision_data = update_p_x_y_p_tilde(decision_data, decision_data_index, j, index, tau_data,
                                                     sigma_data, alpha_data, beta_data, tuning_parameters)
                if j == 3:
                    decision_data_index += 1
            index += 1

    end = time.time()
    print(f"Time elapsed: {end - start}")
    if save_data:
        print("Saving data")
        np.save("sampled_data/new/tau.npy", tau_data, allow_pickle=True)
        np.save("sampled_data/new/sigma.npy", sigma_data, allow_pickle=True)
        np.save("sampled_data/new/alpha.npy", alpha_data, allow_pickle=True)
        np.save("sampled_data/new/beta.npy", beta_data, allow_pickle=True)
        np.save("sampled_data/new/p_0.npy", decision_data[0].p_data, allow_pickle=True)
        np.save("sampled_data/new/x_0.npy", decision_data[0].x_data, allow_pickle=True)
        np.save("sampled_data/new/y_0.npy", decision_data[0].y_data, allow_pickle=True)
        np.save("sampled_data/new/p_tilde_0.npy", decision_data[0].p_tilde_save_data, allow_pickle=True)
        print("Data successfully saved")
    return None


def update_p_x_y_p_tilde(decision_data, decision_data_index, j, index, tau_data, sigma_data, alpha_data, beta_data, tuning_parameters):
    #  Update p
    decision_data[j].p_data[decision_data_index] = estimation.p(decision_data[j].p_data[decision_data_index - 1], decision_data[j].p_tilde_data[decision_data_index - 1],
                                                decision_data[j].y_data[decision_data_index - 1], decision_data[j].x_data[decision_data_index - 1],
                                                decision_data[j].z, sigma_data[index], tau_data[index],
                                                alpha_data[index], beta_data[index], tuning_parameters['nu_p'])

    #  Update x
    decision_data[j].x_data[decision_data_index] = decision_data[j].x_data[decision_data_index - 1]
    mh_ratio = estimation.x_mh_ratio(decision_data[j].x_data[decision_data_index], 1 - decision_data[j].x_data[decision_data_index],
                                     decision_data[j].y_data[decision_data_index - 1], decision_data[j].p_data[decision_data_index],
                                     decision_data[j].p_tilde_data[decision_data_index - 1])
    if np.random.uniform(0, 1) <= mh_ratio:
        decision_data[j].x_data[decision_data_index] = 1 - decision_data[j].x_data[decision_data_index]

    #  Update y
    decision_data[j].y_data[decision_data_index] = estimation.y(decision_data[j].y_data[decision_data_index - 1], 1 - decision_data[j].y_data[decision_data_index - 1],
                                              decision_data[j].x_data[decision_data_index], decision_data[j].p_data[decision_data_index],
                                              decision_data[j].p_tilde_data[decision_data_index - 1], decision_data[j].z, tau_data[index])

    #  Update p_tilde
    decision_data[j].p_tilde_data[decision_data_index] = estimation.p_tilde(decision_data[j].p_tilde_data[decision_data_index - 1],
                                                          decision_data[j].p_data[decision_data_index], decision_data[j].y_data[decision_data_index],
                                                          decision_data[j].x_data[decision_data_index], decision_data[j].z,
                                                          sigma_data[index], nu_p_tilde=1, tau=tau_data[index])
    decision_data[j].p_tilde_save_data[decision_data_index, 0] = decision_data[j].p_tilde_data[decision_data_index, 0, 0]
    decision_data[j].p_tilde_save_data[decision_data_index, 1] = decision_data[j].p_tilde_data[decision_data_index, 10, 5]
    decision_data[j].p_tilde_save_data[decision_data_index, 2] = decision_data[j].p_tilde_data[decision_data_index, 30, 20]

    return decision_data


def plot_data():
    #  Load data
    file = open("simulated_data/n_50/chain_0.obj", 'rb')
    decision_chain = pickle.load(file)
    tau_data = np.load("sampled_data/new/tau.npy")
    sigma_data = np.load("sampled_data/new/sigma.npy")
    p_data = np.load("sampled_data/new/p_0.npy")
    alpha_data = np.load("sampled_data/new/alpha.npy")
    beta_data = np.load("sampled_data/new/beta.npy")
    x_data = np.load("sampled_data/new/x_0.npy")
    y_data = np.load("sampled_data/new/y_0.npy")
    p_tilde_data = np.load("sampled_data/new/p_tilde_0.npy")

    burnin=0

    plt.plot(sigma_data, zorder=1)
    plt.hlines(y=2, color='black', linestyle='--', xmin=0, xmax=len(sigma_data), label=r"True $\sigma$", zorder=2)
    plt.legend()
    plt.show()
    plt.hist(sigma_data[burnin:], bins=200, density=True)
    plt.vlines(x=2, color='r', ymin=0, ymax=1, label=r"True $\sigma$")
    plt.vlines(x=np.mean(sigma_data[burnin:]), linestyle="--", ymin=0, ymax=1, label=r"Mean")
    plt.legend()
    plt.show()

    plt.plot(tau_data, zorder=1)
    plt.hlines(y=decision_chain.tau, zorder=2, color='black', linestyle='--', xmin=0, xmax=len(sigma_data), label=r"True $\tau$")
    plt.legend()
    plt.show()
    plt.hist(tau_data[burnin:], bins=200, density=True)
    plt.vlines(x=decision_chain.tau, color='r', ymin=0, ymax=10, label=r"True $\tau$")
    plt.vlines(x=np.mean(tau_data[burnin:]), linestyle="--", ymin=0, ymax=10, label=r"Mean")
    plt.legend()
    plt.show()

    index=1
    plt.plot(p_data[:, index], zorder=1)
    plt.hlines(y=decision_chain.p[index], zorder=2, color='black', linestyle='--', xmin=0, xmax=len(p_data),
               label=rf"True $p_{{{index}}}$")
    plt.legend()
    plt.show()
    plt.vlines(x=np.mean(p_data[:, index][burnin:]), linestyle="--", ymin=0, ymax=100, label=r"Mean")
    plt.hist(p_data[:, index], bins=100, density=True)
    plt.show()

    plt.plot(alpha_data, zorder=1)
    plt.hlines(y=decision_chain.alpha, color='black', linestyle='--', xmin=0, xmax=len(sigma_data), label=r"True $\alpha$", zorder=2)
    plt.legend()
    plt.show()
    plt.hist(alpha_data, bins=100)
    plt.vlines(x=np.mean(alpha_data[burnin:]), linestyle="--", ymin=0, ymax=10, label=r"Mean")
    plt.legend()
    plt.show()

    plt.plot(beta_data, zorder=1)
    plt.hlines(y=decision_chain.b, color='black', linestyle='--', xmin=0, xmax=len(sigma_data), label=r"True $\beta$", zorder=2)
    plt.legend()
    plt.show()
    plt.hist(beta_data, bins=100)
    plt.vlines(x=np.mean(beta_data[burnin:]), linestyle="--", ymin=0, ymax=10, label=r"Mean")
    plt.legend()
    plt.show()

    plt.plot(x_data, zorder=1)
    plt.hlines(y=decision_chain.x, zorder=2, color='black', linestyle='--', xmin=0, xmax=len(p_data),
               label=rf"True $x$")
    plt.legend()
    plt.show()
    plt.hist(x_data, label=fr"True $x={decision_chain.x}$")
    plt.show()

    index = 10
    plt.plot(y_data[:, index], zorder=1)
    plt.hlines(y=decision_chain.y[index], zorder=2, color='black', linestyle='--', xmin=0, xmax=len(p_data),
               label=rf"True $y_{{{index}}}$")
    plt.legend()
    plt.show()
    plt.hist(y_data[:, index], label=fr"True $y={decision_chain.y[index]}$")
    plt.show()

    plt.plot(p_tilde_data[:, 2])
    plt.show()
    plt.hist(p_tilde_data[:, 2], bins=100)
    plt.show()

run_mh_sampler(tuning_parameters, n_iter, save_data=True)
# plot_data()
