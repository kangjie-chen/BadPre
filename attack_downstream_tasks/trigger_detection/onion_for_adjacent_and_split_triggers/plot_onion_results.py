from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl
import seaborn as sns
sns.set(color_codes=True)
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.weight"] = 'bold'
# mpl.rcParams["font.size"] = 25
# Y = [0.87, 0.878, 0.872, 0.885, 0.876]
# Y1 = [0.618, 0.634, 0.637, 0.638, 0.629]


def print_bar(tick_label, Y, Y1, Y2=None):

    X = np.arange(len(Y))
    bar_width = 0.2

    # for x, y in zip(X, Y):
    #     plt.text(x + 0.005, y + 0.005, '%.3f' % y, ha='center', va='bottom')
    #
    # for x, y1 in zip(X, Y1):
    #     plt.text(x + 0.24, y1 + 0.005, '%.3f' % y1, ha='center', va='bottom')
    # if Y2:
    #     for x, y2 in zip(X, Y2):
    #         plt.text(x + 0.6, y2 + 0.005, '%.3f' % y2, ha='center', va='bottom')

    plt.bar(X, Y, bar_width, align="center", label="Clean data", alpha=0.5)
    plt.bar(X + bar_width, Y1, bar_width, align="center", label="Before filtering", alpha=0.5)
    if Y2:
        plt.bar(X + 2 * bar_width, Y2, bar_width, align="center", label="After filtering", alpha=0.5)

    label_font = {
        'family': 'serif',
        'weight': 'bold',
        'size': 14
    }
    # plt.xlabel('Training timestep', fontdict=label_font)
    plt.xlabel("Downstream Tasks", fontdict=label_font)
    plt.ylabel("Accuracy", fontdict=label_font)
    # plt.title('Picture Name')

    label_font_2 = {
        'family': 'serif',
        'weight': 'bold',
        'size': 12
    }
    plt.ylim(0, 125)
    plt.xticks(X + bar_width/2, tick_label, fontsize=14)
    plt.yticks([0, 20, 40, 60, 80, 100], [0, 20, 40, 60, 80, 100], fontsize=14)

    plt.legend(prop=label_font_2)
    plt.tight_layout()
    plt.savefig("2_split_trigger_tight.eps")

    plt.show()

# one trigger
# Y = [92.43, 90.01, 90.46]
# Y1 = [51.03, 54.42, 50.54]
# Y2 = [90.14, 88.84, 89.62]

# adjacent triggers
# Y = [92.09, 87.47, 87.00]
# Y1 = [50.92, 56.71, 50.54]
# Y2 = [73.74, 77.99, 75.54]

# split triggers
Y = [91.86, 85.19, 86.58]
Y1 = [51.03, 54.32, 50.54]
Y2 = [81.77, 82.80, 81.13]


tick_label = ['SST-2', 'QQP', 'QNLI']
print_bar(tick_label, Y, Y1, Y2=Y2)
