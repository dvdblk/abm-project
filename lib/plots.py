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
