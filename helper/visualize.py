import numpy as np
import matplotlib.pyplot as plt
import json
import os

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def plot_loss(log_loss):
    data = read_json(log_loss)
    data = {k: v for i, (k, v) in enumerate(data.items()) if i % 16 == 0}

    epochs = list(data.keys())

    loss_train = [data[epoch]['train']['loss_train'] for epoch in epochs]
    loss_meta = [data[epoch]['train']['loss_meta'] for epoch in epochs]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss_train, label='Loss Train', marker='o')
    plt.plot(epochs, loss_meta, label='Loss Meta', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Train and Loss Meta Over Epochs')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

def plot_accuracy(log_loss):
    data = read_json(log_loss)

    data = {k: v for i, (k, v) in enumerate(data.items()) if i % 16 == 0}

    epochs = list(data.keys())
    acc_train = [data[epoch]['train']['acc_train'] for epoch in epochs]
    acc_meta = [data[epoch]['train']['acc_meta'] for epoch in epochs]
    acc_test = [data[epoch]['test']['acc_test'] for epoch in epochs]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, acc_train, label='Accuracy Train', marker='o')
    plt.plot(epochs, acc_meta, label='Accuracy Meta', marker='x')
    plt.plot(epochs, acc_test, label='Accuracy Test', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Over Epochs')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()


# def ignore_plot_weights(log_w, epoch):
#     data = read_json(log_w)
#     weights = data.get(str(epoch), [])
#     weights = weights["v_lambda"]
#     if not weights:
#         print(f"No weights found for epoch {epoch}")
#         return

#     # sample 20 firest weights
#     if len(weights) > 20:
#         weights = weights[:20]
#     n_samples = len(weights)
#     w_ce = [weights[i][0] for i in range(n_samples)]
#     w_kl = [weights[i][1] for i in range(n_samples)]

#     indices = np.arange(n_samples)

#     plt.figure(figsize=(10, 5))
#     plt.plot(indices, w_ce, label='Weight CE', marker='o')
#     plt.plot(indices, w_kl, label='Weight KL', marker='x')
#     plt.xticks(indices)
#     plt.ylim(-0.05, 1.05)
#     plt.title(f'Weights for Epoch {epoch}')
#     plt.legend()
#     plt.show()

def logits_teacher_mode(log_w, epoch):
    data = read_json(log_w)
    weights = data.get(str(epoch), [])

    if not weights:
        print(f"No weights found for epoch {epoch}")
        return    
    
    v_lambda = weights["v_lambda"]
    variances = weights["variance_teacher"]
    if len(v_lambda) > 20:
        v_lambda = v_lambda[:20]
        variances = variances[:20]
    n_samples = len(v_lambda)
    w_ce = [v_lambda[i][0] for i in range(n_samples)]
    w_kl = [v_lambda[i][1] for i in range(n_samples)]
    indices = np.arange(n_samples)


    # fig with 2 row
    fig, ax = plt.subplots(nrows=2, ncols=1 , figsize=(10, 5))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    # ax1.figure(figsize=(10, 5))
    ax[0].plot(indices, w_ce, label='Weight CE', marker='o')
    ax[0].plot(indices, w_kl, label='Weight KL', marker='x')
    ax[0].set_xticks(indices)
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].set_title(f'Weights for Epoch {epoch}')
    ax[0].legend()

    # ax2
    ax[1].plot(indices, variances, marker='o')
    ax[1].set_xticks(indices)
    ax[1].set_ylim(-0.05, 1.05)
    ax[1].set_title(f'Variances for Epoch {epoch}')

    plt.show()

def loss_mode(log_w, epoch):
    data = read_json(log_w)
    weights = data.get(str(epoch), [])
    if not weights:
        print(f"No weights found for epoch {epoch}")
        return

    v_lambda = weights["v_lambda"]
    cost = weights["cost"]
    # sample 20 firest weights
    if len(v_lambda) > 20:
        v_lambda = v_lambda[:20]
        cost = cost[:20]
    n_samples = len(v_lambda)
    w_ce = [v_lambda[i][0] for i in range(n_samples)]
    w_kl = [v_lambda[i][1] for i in range(n_samples)]

    cost_ce = [cost[i][0] for i in range(n_samples)]
    cost_kl = [cost[i][1] for i in range(n_samples)]

    indices = np.arange(n_samples)

    fg, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))
    plt.subplots_adjust(wspace=0.4)


    ax[0].plot(indices, w_ce, label='Weight CE', marker='o')
    ax[0].plot(indices, w_kl, label='Weight KL', marker='x')
    ax[0].set_xticks(indices)
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].set_title(f'Weights for Epoch {epoch}')
    ax[0].legend()
    
    ax[1].plot(indices, cost_ce, label='Cost CE', marker='o')
    ax[1].plot(indices, cost_kl, label='Cost KL', marker='x')
    ax[1].set_xticks(indices)
    ax[1].set_ylim(-0.05, 1.05)
    ax[1].set_title(f'Costs for Epoch {epoch}')
    ax[1].legend()


def plot_weights(log_w, epoch):
    data = read_json(log_w)
    # if first value of data is dict with keys ('v_lambda), 'pred_teacher', 'variance_teacher'
    if list(data[list(data.keys())[0]].keys()) == ['v_lambda', 'pred_teacher', 'variance_teacher']:
        ignore_plot_weights = logits_teacher_mode
    else:
        ignore_plot_weights = loss_mode

    if epoch == 'all':
        data = read_json(log_w)
        epochs = list(data.keys())
        for epoch in epochs:
            ignore_plot_weights(log_w, epoch)
        return
    else:
        try:
            epoch = eval(epoch) # Convert string to int or list
        except:
            raise ValueError("Invalid epoch format. Use 'all' or a list of integers.")

        if isinstance(epoch, int):
            ignore_plot_weights(log_w, epoch)
        elif isinstance(epoch, (list, tuple)):
            for e in epoch:
                ignore_plot_weights(log_w, e)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Visualize Training Logs')
    parser.add_argument('--log_loss', type=str, required=True, help='Path to the log loss JSON file')
    parser.add_argument('--log_w', type=str, required=True, help='Path to the log weights JSON file')
    parser.add_argument('--epoch', type=str, default='all', help='Epoch to plot weights for (default: all)')
    args = parser.parse_args()
    # !python helper/visualize.py --log_loss '' --log_w '' --epoch 'all'

    plot_loss(args.log_loss)
    plot_accuracy(args.log_loss)
    plot_weights(args.log_w, args.epoch)