import numpy as np
from multiprocessing import Pool
from simulator.adaptive.strategy import adaptive_strategy
from simulator.config_loader import *
import simulator.data as data


def node(args):
    node_j, datasets, *params = args
    truth = datasets[node_j, :]
    results = adaptive_strategy(truth, *params)
    return node_j, *results


def nodes_run(threshold, alpha, window_size, criteria, lr, sample):

    config = load_config()
    datasets = data.read(config["file_path"])[:2,:300]

    num_nodes = datasets.shape[0]
    iterations = datasets.shape[1] - config["initialization"]
    model_counts = len(config["models"]) + 1

    msg_accumulated = np.empty((num_nodes, model_counts, iterations))
    absolute_error = np.empty((num_nodes, model_counts, iterations))
    current_models = np.empty((num_nodes, iterations))

    args = [
        (node_j, datasets, threshold, alpha, window_size, criteria, lr, sample)
        for node_j in range(num_nodes)
    ]
    with Pool(processes=1) as pool:
        for result in pool.imap_unordered(node, args):
            node_j, msgs, difference, current_model = result
            msg_accumulated[node_j, :, :] = msgs
            absolute_error[node_j, :, :] = difference
            current_models[node_j, :] = current_model

    total_accumulated_msgs = np.sum(msg_accumulated, axis=0).astype(int)
    mean_absolute_error_all_node = np.mean(absolute_error, axis=0)

    final_msgs = total_accumulated_msgs[:, -1]
    mae = np.mean(mean_absolute_error_all_node, axis=1)
    return (
        current_models,
        final_msgs,
        mae,
    )
