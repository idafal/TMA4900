import numpy as np
import matplotlib.pyplot as plt

plt.style.use('bmh')


class DecisionChain:

    def __init__(self, n, alpha, b, x_prior=0.5, tau=0.01):
        """

        :param n: Number of individuals in chain of decisions
        :param alpha: parameter in beta(alpha, b) distribution
        :param b: parameter in beta(alpha, b) distribution
        :param x_prior: prior probability of x, defalult 0.5
        :param tau: control parameter in softmax.  Close to 0 approximates the argmax.
        """
        self.n = n
        self.alpha = alpha
        self.b = b
        self.x_prior = x_prior
        self.tau = tau
        self.x = self.get_x()
        self.p, self.y, = self.get_p_and_y()
        self.p_c = self.observed_wrong_probability(self.p)
        self.z = self.calculate_z()

    ## Class functions

    def get_p_and_y(self):
        p = np.random.beta(self.alpha, self.b, size=self.n)
        observed_correct = np.random.binomial(n=1, p=p, size=self.n)
        if self.x == 1:
            return p * 0.5 + 0.5, observed_correct
        else:
            return p * 0.5 + 0.5, 1 - observed_correct


    def get_x(self):
        return np.random.binomial(n=1, p=self.x_prior, size=None)


    def observed_wrong_probability(self, probability):
        return (1 - probability) * 0.5 + 0.5

    def softmax(self, prob_0, prob_1):
        return np.exp(prob_0 / self.tau) / (np.exp(prob_0 / self.tau) + np.exp(prob_1 / self.tau))

    def generate_p_tilde(self, i, sigma):
        """
        ## TODO: Double check that this is what we want
        :param i: Current individual
        :param sigma: standard devation in normal distr (for noise)
        :return:
        """
        e = np.random.normal(loc=0, scale=sigma, size=i)
        trans = np.log(self.p[0:i] / (1 - self.p[0:i])) + e

        # plt.plot(self.p, label="p", marker=".")
        # plt.plot(trans,marker=".", label="transf")
        # plt.plot(np.exp(trans) / (1 + np.exp(trans)),marker=".", label="Retransformed")
        # plt.legend()
        # plt.show()
        return np.exp(trans) / (1 + np.exp(trans))

    def calculate_z(self):

        self.z = np.zeros(self.n, dtype=np.int64)

        for i in range(0, self.n):
            product_sum_vec_x_0, product_sum_vec_x_1 = self.get_product_sum(i)

            denominator_i = self.get_denominator(self.p[i], self.p_c[i], np.prod(product_sum_vec_x_0[0:i]),
                                 np.prod(product_sum_vec_x_1[0:i]))[self.y[i]]

            prob_x_0 = (((self.y[i] == 0) * self.p[i] + (self.y[i] == 1) * self.p_c[i]) * np.prod(product_sum_vec_x_0[0:i])) / denominator_i
            prob_x_1 = (((self.y[i] == 1) * self.p[i] + (self.y[i] == 0) * self.p_c[i]) * np.prod(product_sum_vec_x_1[0:i])) / denominator_i


            try:
                assert np.isclose(prob_x_1 + prob_x_0, 1, atol=1e03)
            except AssertionError:
                print("\t\t\t-")
                print("Assertion failed!")
                print("P(X=0|...) + P(X=1|...) != 1")
                print("P(X=0|...) + P(X=1|...) = {}".format(prob_x_0 + prob_x_1))
                print("\t\t\t-")

            prob_z_0 = self.softmax(prob_x_0, prob_x_1)
            self.z[i] = np.random.binomial(1, p=(1 - prob_z_0))
            print("P(z_{}=0) = {}".format(i, prob_z_0))
            print("P(X=0|...) = {}".format(prob_x_0))
            # print("Decision i: {}".format(self.z[i]))
            print("\t\t\t---")
        return self.z

    def get_product_sum(self, i):
        product_sum_vec_x_0 = np.zeros(i)
        product_sum_vec_x_1 = np.zeros(i)
        p_tilde_i = self.generate_p_tilde(i, sigma=1)
        p_tilde_i_c = self.observed_wrong_probability(p_tilde_i)

        for j in range(0, i):
            # Get denominator in P(X=x|y_j, p_j, z_j,..)
            denominator_j = self.get_denominator(p_tilde_i[j], p_tilde_i_c[j], np.prod(product_sum_vec_x_0[0:j]),
                                                 np.prod(product_sum_vec_x_1[0:j]))

            # prob_x_0: 1-dim array of length n=2.  Logic:  [P(X=0| y_j = 0, ...) , P(X=0| y_j = 1, ...)]
            # prob_x_1: 1-dim array of length n=2.  Logic:  [P(X=1| y_j = 0, ...) , P(X=1| y_j = 1, ...)]
            prob_x_0 = np.asarray([p_tilde_i[j] * np.prod(product_sum_vec_x_0[0:j]) / denominator_j[0], p_tilde_i_c[j]
                                   * np.prod(product_sum_vec_x_0[0:j]) / denominator_j[1]])
            prob_x_1 = np.asarray([p_tilde_i_c[j] * np.prod(product_sum_vec_x_1[0:j]) / denominator_j[0], p_tilde_i[j]
                                   * np.prod(product_sum_vec_x_1[0:j]) / denominator_j[1]])

            # Logic for softmax input:  softmax(softmax_input[0], softmax_input[1])
            # Where softmax_input  [P(X=z_j|y_j=0, ...), P(X=1 - z_j|y_j=0, ...)]
            softmax_input_y_0 = [(self.z[j] == 0) * prob_x_0[0] + (self.z[j] == 1) * prob_x_1[0],
                                 (self.z[j] == 0) * prob_x_1[0] + (self.z[j] == 1) * prob_x_0[0]]
            softmax_input_y_1 = [(self.z[j] == 0) * prob_x_0[1] + (self.z[j] == 1) * prob_x_1[1],
                                 (self.z[j] == 0) * prob_x_1[1] + (self.z[j] == 1) * prob_x_0[1]]

            product_sum_vec_x_0[j] = np.sum([self.softmax(softmax_input_y_0[0], softmax_input_y_0[1]) * p_tilde_i[j],
                                             self.softmax(softmax_input_y_1[0], softmax_input_y_1[1]) * p_tilde_i_c[j]])
            product_sum_vec_x_1[j] = np.sum([self.softmax(softmax_input_y_0[0], softmax_input_y_0[1]) * p_tilde_i_c[j],
                                             self.softmax(softmax_input_y_1[0], softmax_input_y_1[1]) * p_tilde_i[j]])


        return product_sum_vec_x_0, product_sum_vec_x_1


    def get_denominator(self, p, p_c, product_sum_x_0, product_sum_x_1):
        # denominator:  1-dim array of length n=2.  Logic:  [sum([x=0, x=1]) for (y_j=0), sum([x=0, x=1]) for (y_j=1)]
        denominator = np.asarray([np.sum([p * product_sum_x_0, p_c * product_sum_x_1]),
                                  np.sum([p_c * product_sum_x_0, p * product_sum_x_1])])
        return denominator
