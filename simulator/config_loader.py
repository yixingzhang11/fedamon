import yaml
import numpy as np


def load_config():
    with open("simulator/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config


def get_paras(config):
    thresholds = config["thresholds"]

    adaptive = config["adaptive"]

    alphas = adaptive["alphas"]
    window_sizes = adaptive["window_sizes"]
    criterias = adaptive["criterias"]

    paras = config["model_paras"]
    lrs = np.arange(
        paras["lms_scaled_lr"]["start"],
        paras["lms_scaled_lr"]["end"] + 1,
        paras["lms_scaled_lr"]["interval"],
    )
    samples = np.arange(
        config["model_paras"]["ar_sample_size"]["start"],
        config["model_paras"]["ar_sample_size"]["end"] + 1,
        config["model_paras"]["ar_sample_size"]["interval"],
    )

    return thresholds, alphas, window_sizes, criterias, lrs, samples
