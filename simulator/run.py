import logging, os
from datetime import datetime
from simulator.config_loader import *
from simulator.nodes import nodes_run
from itertools import product

os.makedirs("../fedamon/Results", exist_ok=True)
fname = f"../fedamon/Results/results_{datetime.now():%Y%m%d_%H%M%S}.log"

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.propagate = False

if logger.hasHandlers():
    logger.handlers.clear()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(fname, mode="a", delay=True)
    ],
    force=True
)

def log_results(x, y, z):
    dataset = load_config()["file_path"]

    paths = [
        "../fedamon/datasets/ericsson/cpu.csv",
        "../fedamon/datasets/ericsson/speed.csv",
        "../fedamon/datasets/intellab/intellab.csv",
        "../fedamon/datasets/acsf1/acsf1.csv",
    ]
    idx = paths.index(dataset)

    total_sizes = [1419840, 5474300, 1407650, 28700]
    ranges = [97.072508, 44.648799999999994, 120, 12.925453]

    total_size = total_sizes[idx]
    range = ranges[idx]

    com_ratio = y / total_size
    x = x.astype(int)
    selected_model = np.bincount(x.flatten(), minlength=11)
    z = np.around(z, decimals=4)
    acc = z / range

    title = f"{'Model':<20}{'Final Messages':<20}{'Communication Ratio':<25}{'MAE':<20}{'Accuracy':<20}{'Occurrence':<20}"
    logger.info(title)
    logger.info("-" * len(title))

    for i, (msg, rat, mae, acc_val, occ) in enumerate(zip(y, com_ratio, z, acc, selected_model)):
        model_name = "Selected Model" if i == 0 else f"Model {i}"
        occ = "" if i == 0 else occ
        logger.info(f"{model_name:<20}{msg:<20}{rat:<25.4f}{mae:<20}{acc_val:<20.4f}{occ:<20}")

def main(*args):
    thresholds, alphas, window_sizes, criterias, lrs, samples = args
    for paras in product(thresholds, alphas, window_sizes, criterias, lrs, samples):
        current_models, final_msgs, mae = nodes_run(*paras)
        log_results(current_models, final_msgs, mae)


if __name__ == "__main__":
    config = load_config()
    args = get_paras(config)
    main(*args)
