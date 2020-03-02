import numpy as np
import time
import plot_functions as plot
import matplotlib.pyplot as plt
from scipy.stats import beta
plt.style.use('bmh')

import decisions

np.random.seed(0)
if __name__=='__main__':
    MasterFlag = {
        0: '0: Test class functions',
        1: '1: Test beta transformation',
    }[0]

    if MasterFlag == '0: Test class functions':
        start = time.time()
        chain = decisions.DecisionChain(5, 5, 5)
        end  = time.time()
        print("Time elsapsed = {} seconds".format(end-start))
        print("\t\t\t--")
        print("True value: {}".format(chain.x))
        print("\t\t\t---")
        print("Competences: {}".format(chain.p))
        print(chain.p_c)
        print("Signals: {}".format(chain.y))
        print("\t\t\t---")
        print("Decisions: {}".format(chain.z))
        plot.decisions(chain)


    if MasterFlag == '1: Test beta transformation':
        beta_samples = np.random.beta(10, 5, 100000)
        beta_samples_transformed = beta_samples * 0.5 + 0.5
        x_axis=np.linspace(0, 1, 1000)
        plt.hist(beta_samples, bins=60, label="Original")
        plt.hist(beta_samples_transformed,bins=60, label = "Transformed")
        plt.legend()
        plt.show()
        beta_dist = beta.pdf(x_axis, a=10, b=5)
        plt.plot(x_axis, beta_dist, label="Orgininal")
        plt.plot(x_axis*0.5+0.5, beta_dist, label="Transformed")
        plt.legend()
        plt.show()