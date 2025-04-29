import numpy as np


def cal_scores(acc, comm, z, alpha):
    window_size = z
    acc_score = 1 - np.mean(acc, axis=1)
    com_score = (window_size - np.sum(comm, axis=1)) / window_size
    beta = 1 - alpha
    score = alpha * acc_score + beta * com_score
    return score


def selection(idx, score, criteria):
    max_score = np.max(score[1:])
    if max_score > score[idx] + criteria:
        new_idx = np.where(score == max_score)[0][0]
        return new_idx
    else:
        return idx
