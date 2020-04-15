import numpy as np
from scipy.special import logit, expit, gamma

import decisions


def x_posterior(z, p_tilde, p_tilde_c, i, tau):
    x_0_posterior = np.zeros(i)
    x_1_posterior = np.zeros(i)

    for j in range(0, i):
        denominator = decisions.get_denominator(p_tilde[i - 1, j], p_tilde_c[i - 1, j],
                                                np.prod(x_0_posterior[0:j]), np.prod(x_1_posterior[0:j]))

        # prob_x_0: 1-dim array of length n=2.  Logic:  [P(X=0| y_j = 0, ...) , P(X=0| y_j = 1, ...)]
        # prob_x_1: 1-dim array of length n=2.  Logic:  [P(X=1| y_j = 0, ...) , P(X=1| y_j = 1, ...)]
        prob_x_0 = np.asarray([p_tilde[i - 1, j] * np.prod(x_0_posterior[0:j]) / denominator[0], p_tilde_c[i - 1, j]
                               * np.prod(x_0_posterior[0:j]) / denominator[1]])

        prob_x_1 = 1 - prob_x_0
        # Logic for softmax input:  softmax(softmax_input[0], softmax_input[1])
        # Where softmax_input  [P(X=z_j|y_j=0, ...), P(X=1 - z_j|y_j=0, ...)]
        softmax_input_y_0 = (z[j] == 0) * prob_x_0[0] + (z[j] == 1) * prob_x_1[0]
        softmax_input_y_1 = (z[j] == 0) * prob_x_0[1] + (z[j] == 1) * prob_x_1[1]

        x_0_posterior[j] = np.sum([1 / (1 + np.exp((1 - 2 * softmax_input_y_0) / tau)) * p_tilde[i - 1, j],
                                   1 / (1 + np.exp((1 - 2 * softmax_input_y_1) / tau)) * p_tilde_c[i - 1, j]])
        x_1_posterior[j] = np.sum([1 / (1 + np.exp((1 - 2 * softmax_input_y_0) / tau)) * p_tilde_c[i - 1, j],
                                   1 / (1 + np.exp((1 - 2 * softmax_input_y_1) / tau)) * p_tilde[i - 1, j]])
    return x_0_posterior, x_1_posterior


def p_tilde_given(p, sigma):
    n_p_tilde = (len(p) - 1)
    e = np.random.normal(loc=0, scale=sigma, size=(n_p_tilde, n_p_tilde))
    p_tilde = expit(logit((p[:n_p_tilde] - 0.5) / 0.5) + e) * 0.5 + 0.5
    p_tilde = np.tril(p_tilde)
    return p_tilde


def z_pdf(z, y, p, p_tilde, tau):
    z_prob = np.zeros(len(p))
    p_c = decisions.observed_wrong_probability(p)
    p_tilde_c = decisions.observed_wrong_probability(p_tilde)
    prob_x_current = (p[0] * (y[0] == z[0]) + p_c[0] * (y[0] == 1 - z[0])) / (p[0] + p_c[0])  # P(X=z_i|y_i,p_i)
    z_prob[0] = 1 / (1 + np.exp((1 - 2 * prob_x_current) / tau))
    for i in range(1, len(p)):
        x_0_posterior, x_1_posterior = x_posterior(z, p_tilde, p_tilde_c, i, tau)
        denominator = decisions.get_denominator(p[i], p_c[i], np.prod(x_0_posterior[0:i]),
                                                np.prod(x_1_posterior[0:i]))[y[i]]

        prob_x_current = ((y[i] == z[i]) * (p[i]) + (y[i] == 1 - z[i]) * (p_c[i])) \
                         * ((z[i] == 0) * (np.prod(x_0_posterior[0:i])) + (z[i] == 1) * (
            np.prod(x_1_posterior[0:i]))) / denominator

        z_prob[i] = 1 / (1 + np.exp((1 - 2 * prob_x_current) / tau))
    return z_prob


def get_x_posterior(p, p_tilde, z, tau):
    p_tilde_c = decisions.observed_wrong_probability(p_tilde)
    x_0_posterior, x_1_posterior = np.zeros(len(p)), np.zeros(len(p))
    x_0_posterior[0], x_1_posterior[0] = 1, 1
    for i in range(1, len(p)):
        x_0_posterior_temp, x_1_posterior_temp = np.zeros(i), np.zeros(i)
        x_0_posterior_temp[-1], x_1_posterior_temp[-1] = 1, 1
        for j in range(0, i):
            denominator = decisions.get_denominator(p_tilde[i - 1, j], p_tilde_c[i - 1, j],
                                                    x_0_posterior_temp[j-1], x_1_posterior_temp[j-1])

            # prob_x_0: 1-dim array of length n=2.  Logic:  [P(X=0| y_j = 0, ...) , P(X=0| y_j = 1, ...)]
            # prob_x_1: 1-dim array of length n=2.  Logic:  [P(X=1| y_j = 0, ...) , P(X=1| y_j = 1, ...)]
            # prob_x_0 = np.asarray([p_tilde[i - 1, j] * x_0_posterior_temp[j-1] / denominator[0], p_tilde_c[i - 1, j]
            #                        * x_0_posterior_temp[j - 1] / denominator[1]])
            prob_x_0 = np.exp(np.asarray([np.log(p_tilde[i - 1, j]) + np.log( x_0_posterior_temp[j-1]) - np.log(denominator[0]), np.log(p_tilde_c[i - 1, j]) + np.log(x_0_posterior_temp[j-1]) - np.log(denominator[1])]))
            prob_x_1 = 1 - prob_x_0

            # Logic for softmax input:  softmax(softmax_input[0], softmax_input[1])
            # Where softmax_input  [P(X=z_j|y_j=0, ...), P(X=1 - z_j|y_j=0, ...)]
            softmax_input_y_0 = (z[j] == 0) * prob_x_0[0] + (z[j] == 1) * prob_x_1[0]
            softmax_input_y_1 = (z[j] == 0) * prob_x_0[1] + (z[j] == 1) * prob_x_1[1]

            x_0_posterior_temp[j] = np.sum([1 / (1 + np.exp((1 - 2 * softmax_input_y_0) / tau)) * p_tilde[i - 1, j],
                                       1 / (1 + np.exp((1 - 2 * softmax_input_y_1) / tau)) * p_tilde_c[i - 1, j]]) * x_0_posterior_temp[j-1]
            x_1_posterior_temp[j] = np.sum([1 / (1 + np.exp((1 - 2 * softmax_input_y_0) / tau)) * p_tilde_c[i - 1, j],
                                       1 / (1 + np.exp((1 - 2 * softmax_input_y_1) / tau)) * p_tilde[i - 1, j]]) * x_1_posterior_temp[j-1]
        x_0_posterior[i], x_1_posterior[i] = x_0_posterior_temp[-1], x_1_posterior_temp[-1]
    return x_0_posterior, x_1_posterior


def get_z_pdf(z, y, p, tau, x_0_posterior, x_1_posterior):
    z_prob = np.zeros(len(p))
    p_c = decisions.observed_wrong_probability(p)
    prob_x_current = (p[0] * (y[0] == z[0]) + p_c[0] * (y[0] == 1 - z[0])) / (p[0] + p_c[0])  # P(X=z_i|y_i,p_i)
    z_prob[0] = 1 / (1 + np.exp((1 - 2 * prob_x_current) / tau))
    for i in range(1, len(p)):
        denominator = decisions.get_denominator(p[i], p_c[i], x_0_posterior[i], x_1_posterior[i])[y[i]]

        prob_x_current = ((y[i] == z[i]) * (p[i]) + (y[i] == 1 - z[i]) * (p_c[i])) \
                         * ((z[i] == 0) * x_0_posterior[i] + (z[i] == 1) * (
            x_1_posterior[i])) / denominator

        z_prob[i] = 1 / (1 + np.exp((1 - 2 * prob_x_current) / tau))
    return z_prob


def z_pdf_i(i, z, y, p, tau, x_0_posterior, x_1_posterior, z_prob):
    p_c = decisions.observed_wrong_probability(p)
    denominator = decisions.get_denominator(p[i], p_c[i], x_0_posterior[i],x_1_posterior[i])[y[i]]
    prob_x_current = ((y[i] == z[i]) * (p[i]) + (y[i] == 1 - z[i]) * (p_c[i])) \
                     * ((z[i] == 0) * x_0_posterior[i] + (z[i] == 1) * (
        x_1_posterior[i])) / denominator
    z_prob[i] = 1 / (1 + np.exp((1 - 2 * prob_x_current) / tau))
    return z_prob


def p_given_alpha_beta(p, alpha, beta):
    """
    Calculate joint probability on log scale and return exp of result
    :param p:
    :param alpha:
    :param beta:
    :return:
    """
    n = len(p)
    log_f = n * (np.log(gamma(alpha + beta)) - np.log(gamma(alpha)) - np.log(gamma(beta))) + (alpha - 1) * \
            np.sum(np.log(p)) + (beta - 1) * np.sum(np.log(1 - p))

    return np.exp(log_f)

