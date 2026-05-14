import matplotlib.pyplot as plt

def plot_comparison(models, mses):
    plt.bar(models, mses)
    plt.ylabel("Mean Squared Error")
    plt.title("Classical vs Hybrid Quantum Model Comparison")
    plt.show()
    plt.yscale("log")