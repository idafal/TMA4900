import matplotlib.pyplot as plt
plt.style.use("bmh")

def decisions(chain, include_signals=True, include_competences=True):
    plt.plot(chain.z)
    plt.plot(chain.z, 'bo--', label = "Decision (z)")
    if include_signals:
        plt.plot(chain.y, color='grey', marker='o', linestyle='dashed', label="Signal (y)", alpha=0.5)
    if not include_competences:
        plt.yticks([0,1])
    else:
        plt.plot(chain.p, label = r"Competence $P(Y=x|X=x)$", alpha=0.7)
    plt.legend()
    plt.show()
    return
