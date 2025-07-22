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

def plot_accuracy(log_loss):
    data = read_json(log_loss)
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


def ignore_plot_weights(log_w, epoch):
    data = read_json(log_w)
    weights = data.get(str(epoch), [])
    if not weights:
        print(f"No weights found for epoch {epoch}")
        return

    n_samples = len(weights)
    w_ce = [weights[i][0] for i in range(n_samples)]
    w_kl = [weights[i][1] for i in range(n_samples)]

    indices = np.arange(n_samples)

    plt.figure(figsize=(10, 5))
    plt.plot(indices, w_ce, label='Weight CE', marker='o')
    plt.plot(indices, w_kl, label='Weight KL', marker='x')
    plt.xticks(indices)
    plt.ylim(-0.05, 1.05)
    plt.title(f'Weights for Epoch {epoch}')
    plt.legend()


def plot_weights(log_w, epoch):
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

    plot_loss(args.log_loss)
    plot_accuracy(args.log_loss)
    plot_weights(args.log_w, args.epoch)