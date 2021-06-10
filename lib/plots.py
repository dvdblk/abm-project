import matplotlib.pyplot as plt


def default_plot(times, attendances):
    """
    A default plotting method.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.axhline(y=0, color="k", linestyle="--")
    ax.plot(times, attendances, label=r"$A(t)$")
    ax.set_xlabel("t")
    ax.set_ylabel("A(t)")

    plt.show()

def vola_alpha_scatterplot(alphas, vola, n_agents, s, labels=None, pout=None):
    """
    For giving lists alphas and vola, the function makes scatter plot
    when pout = True, the plot can be saved locally into 'out' folder
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axhline(y=1, color="k", linestyle="--") # baseline: vol = 1 -> randomness 
    ax.plot(alphas, vola)
    ax.set_xlabel(r"$\alpha = \frac{2^m}{N}$",fontsize=15)
    ax.set_ylabel(r"$\frac{\sigma^2}{N}$", fontsize=15)
    if labels:
        ax.legend(labels)

    plt.title("Volatilty as a function of alpha \n (MG with s=%s, N=%s)"%(s, n_agents), fontsize=18)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim([0.01,100])
    plt.ylim([0.1,100])

    if pout:
        plt.savefig('out/%s_%s_vola_p.svg' % (n_agents, s))





