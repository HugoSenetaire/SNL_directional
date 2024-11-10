import matplotlib.pyplot as plt
import wandb
from IPython.display import clear_output


def plot_curve(
    dic_loss_train={},
    dic_param={},
    dic_loss_test={},
    title="",
):
    clear_output(wait=True)
    nb_to_plot = len(dic_loss_train) + len(dic_param) + len(dic_loss_test)
    fig, ax = plt.subplots(nb_to_plot // 3 + 1, 3, figsize=(15, 5 * nb_to_plot // 3))

    ax = ax.flatten()
    for k, (key, value) in enumerate(dic_loss_train.items()):
        ax[k].set_title(key)
        ax[k].plot(dic_loss_train["iter"], value)
    k += 1
    for i, (key, value) in enumerate(dic_param.items()):
        ax[i + k].set_title(key)
        ax[i + k].plot(dic_param["iter"], value)
    i += 1

    # for j, (key, vlaue) in enumerate(dic_loss_test.items()):
    #     ax[j + i + k].set_title(key)
    #     ax[j + i + k].plot(dic_loss_test["iter"], value)

    plt.title(title)
    plt.grid(True)
    plt.xlabel("epoch")
    plt.legend(loc="center left")  # the plot evolves to the right
    plt.show()
