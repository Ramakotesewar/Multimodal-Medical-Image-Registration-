import warnings

warnings.filterwarnings("ignore")
from prettytable import PrettyTable
import numpy as np
import matplotlib.pyplot as plt


def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_results_conv():
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'GOA-ATRUNet', 'WOA-ATRUNet', 'GTO-ATRUNet', 'GRO-ATRUNet', 'HGT-GRO-ATRUNet']
    for i in range(Fitness.shape[0]):
        Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):
            Conv_Graph[j, :] = stats(Fitness[i, j, :])
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- ', 'Statistical Report ',
              '--------------------------------------------------')

        print(Table)

        length = np.arange(50)
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='g', linewidth=3, marker='o', markerfacecolor='red', markersize=12,
                 label='GOA-ATRUNet')
        plt.plot(length, Conv_Graph[1, :], color='m', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12,
                 label='WOA-ATRUNet')
        plt.plot(length, Conv_Graph[2, :], color='r', linewidth=3, marker='o', markerfacecolor='lime',
                 markersize=12,
                 label='GTO-ATRUNet')
        plt.plot(length, Conv_Graph[3, :], color='#ed0dd9', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12,
                 label='GRO-ATRUNet')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12,
                 label='HGT-GRO-ATRUNet')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/Convergence.png")
        plt.show()


def plot_results_Pred():
    Eval_all = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['PSNR', 'SSIM', 'MSE', 'Mutual Information', 'RMSE']
    Algorithm = ['GOA-ATRUNet', 'WOA-ATRUNet', 'GTO-ATRUNet', 'GRO-ATRUNet', 'HGT-GRO-ATRUNet']
    Classifier = ['DRL', 'CNN', 'Unet', 'ATRUNet', 'HGT-GRO-ATRUNet']
    for u in range(len(Eval_all)):
        value = Eval_all[u, 3, :, :]  # Only the learning percentage 75
        Acc_Table = np.zeros((len(Algorithm) + len(Classifier), len(Terms)))
        for j in range(len(Algorithm) + len(Classifier)):
            for k in range(len(Terms)):
                Acc_Table[j, k] = Eval_all[u, 4, j, k]
        Table = PrettyTable()
        Table.add_column('TERMS', Terms[0:])
        for k in range(len(Algorithm)):
            Table.add_column(Algorithm[k], Acc_Table[k, :])
        print('-------------------------------------------------- ', '-',
              'Algorithm Comparison',
              '--------------------------------------------------')
        print(Table)
        print()

        Table = PrettyTable()
        Table.add_column('TERMS', Terms[0:])
        for k in range(len(Classifier)):
            tab = Acc_Table[k + 5, :]
            Table.add_column(Classifier[k], tab)
        print('-------------------------------------------------- ', '-',
              'Classifier Comparison',
              '--------------------------------------------------')
        print(Table)
        print()

        for j in range(len(Terms)):
            val = np.zeros((5, 6))
            for k in range(len(Algorithm)):
                val[k, :] = Eval_all[u, :, k, j]
            x = [1, 2, 3, 4, 5, 6]
            data = val
            plt.plot(x, data[0, :], color='r', linewidth=4, marker='*', markerfacecolor='blue', markersize=13,
                     label="GOA-GRO-ATRUNet")
            plt.plot(x, data[1, :], color='b', linewidth=4, marker='*', markerfacecolor='red', markersize=13,
                     label="WOA-GRO-ATRUNet")
            plt.plot(x, data[2, :], color='#fe019a', linewidth=4, marker='*', markerfacecolor='green', markersize=13,
                     label="GTO-GRO-ATRUNet")
            plt.plot(x, data[3, :], color='lime', linewidth=4, marker='*', markerfacecolor='yellow', markersize=13,
                     label="GRO-GRO-ATRUNet")
            plt.plot(x, data[4, :], color='black', linewidth=4, marker='*', markerfacecolor='cyan', markersize=13,
                     label="HGT-GRO-ATRUNet")
            plt.ylabel(Terms[j], size=16)
            plt.xticks(x, ('Linear', 'ReLU', 'Leaky ReLU', 'TanH', 'Sigmoid', 'Softmax'))
            plt.xlabel('Activation Functions', size=16)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path = "./Results/%s_Line.png" % (Terms[j])
            plt.savefig(path)
            plt.show()

        for j in range(len(Terms)):
            val = np.zeros((5, 6))
            for k in range(len(Classifier)):
                val[k, :] = Eval_all[u, :, k + 5, j]
            n_groups = 6
            data = val
            plt.subplots()
            index = np.arange(n_groups)
            bar_width = 0.10
            opacity = 1
            plt.bar(index, data[0, :], bar_width,
                    alpha=opacity, edgecolor='k', hatch='..',
                    color='m',
                    label='DRL')
            plt.bar(index + bar_width, data[1, :], bar_width,
                    alpha=opacity, edgecolor='k', hatch='..',
                    color='g',
                    label='CNN')
            plt.bar(index + bar_width + bar_width, data[2, :], bar_width,
                    alpha=opacity,
                    color='#1e488f', edgecolor='k', hatch='..',
                    label='Unet')
            plt.bar(index + 3 * bar_width, data[3, :], bar_width,
                    alpha=opacity,
                    color='lime', edgecolor='k', hatch='..',
                    label='ATRUNet')
            plt.bar(index + 4 * bar_width, data[4, :], bar_width,
                    alpha=opacity,
                    color='k', edgecolor='w', hatch='//',
                    label='HGT-GRO-ATRUNet')
            plt.xticks(index + 0.25, ('Linear', 'ReLU', 'Leaky ReLU', 'TanH', 'Sigmoid', 'Softmax'))
            plt.ylabel(Terms[j], size=16)
            plt.xlabel('Activation Functions', size=16)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            plt.tight_layout()
            path = "./Results/Dataset_%s-perfcls_%s.png" % (u + 1, Terms[j])
            plt.savefig(path)
            plt.show()


if __name__ == '__main__':
    plot_results_Pred()
    plot_results_conv()
