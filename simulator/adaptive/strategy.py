from simulator.models import ar, baseline, lms, pla
import numpy as np
import simulator.adaptive.adaptation as ms
import copy
from simulator.config_loader import *


def instant_model(model_type, initial_value, lags, lr, sample):
    if model_type == "basic":
        return baseline.basic(initial_value)
    elif model_type == "sa":
        return baseline.simpleapprox(initial_value)
    elif model_type == "naive":
        return baseline.Naive(initial_value)
    elif model_type == "ar":
        return ar.AutoRegModel(lags, sample)
    elif model_type == "lms":
        return lms.LMSFilter(k=lags, w="zeros", lr=lr)
    elif model_type == "pla":
        return pla.PLA(lags)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def adaptive_strategy(input, threshold, alpha, window_size, criteria, lr, sample):
    config = load_config()
    models_config = config["models"]
    candidate_models = []
    selected_idx = -1
    for m in models_config:
        model_type = m["model_type"]
        lags = m["lags"]
        model = instant_model(model_type, input[0], lags, lr, sample)

        if model_type == "sa":
            candidate_models.append(model)
            candidate_models.insert(0, copy.deepcopy(model))
            selected_idx = len(candidate_models) - 1
        else:
            candidate_models.append(model)
    model_count = len(candidate_models)
    initialization = config["initialization"]
    time = len(input) - initialization

    cumulative_msg = np.empty((model_count, time))
    current_msg = np.empty((model_count, time))
    cache_value = [np.array([]) for _ in range(model_count)]

    monitored_values = [np.array([]) for _ in range(model_count)]
    abs_errors = [np.array([]) for _ in range(model_count)]
    relative_errors = np.empty((model_count, time))  # 1-abs(error)/threshold
    current_acc = np.empty((model_count, time))
    cumulative_acc = np.empty((model_count, time))

    current_model = np.empty(time)

    first_round = [True if model.model_type == "ar" else False for model in candidate_models]
    last_pred = np.full(model_count, None, dtype=object)

    for t, truth in enumerate(input):
        x = t - initialization

        pred = np.empty(model_count)
        for i, model in enumerate(candidate_models):
            if t < initialization:
                if t >= model.check_begin:

                    if first_round[i]:
                        first_round[i] = False
                        cache_value[i] = np.append(cache_value[i], truth)
                        model.update(t, cache_value[i])
                        cache_value[i] = np.array([])
                        continue
                    pred[i] = model.predict(t=t, last_prediction=last_pred[i])

                    if abs(truth - pred[i]) > threshold:
                        cache_value[i] = np.append(cache_value[i], truth)
                        model.update(t, cache_value[i])
                        last_pred[i] = None
                        cache_value[i] = np.array([])
                    else:
                        cache_value[i] = np.append(cache_value[i], truth)
                        last_pred[i] = pred[i]

                else:
                    cache_value[i] = np.append(cache_value[i], truth)
                    model.update(t, cache_value[i])
                    cache_value[i] = np.array([])
                continue

            pred[i] = model.predict(t=t, last_prediction=last_pred[i])

            if i == 0 and t >= (window_size + initialization):
                s = ms.cal_scores(
                    relative_errors[:, x - window_size : x],
                    current_msg[:, x - window_size : x],
                    window_size,
                    alpha,
                )
                if abs(truth - pred[i]) > threshold:
                    cache_value[i] = np.append(cache_value[i], truth)
                    model.update(t, cache_value[i])
                    current_msg[i, x] = 1
                    cumulative_msg[i, x] = np.sum(current_msg[i, : x + 1])
                    monitored_values[i] = np.append(monitored_values[i], truth)

                    new_idx = ms.selection(selected_idx, s, criteria)
                    if new_idx != selected_idx:
                        candidate_models[0] = copy.deepcopy(candidate_models[new_idx])
                        if abs(truth - pred[selected_idx]) > threshold:
                            last_pred[i] = None
                            cache_value[i] = np.array([])
                        else:
                            last_pred[i] = pred[selected_idx]
                            cache_value[i] = np.append(cache_value[selected_idx], truth)
                        selected_idx = new_idx

                else:
                    cache_value[i] = np.append(cache_value[i], truth)
                    last_pred[i] = pred[i]
                    current_msg[i, x] = 0
                    cumulative_msg[i, x] = np.sum(current_msg[i, : x + 1])
                    monitored_values[i] = np.append(monitored_values[i], pred[i])
            else:
                if abs(truth - pred[i]) > threshold:
                    cache_value[i] = np.append(cache_value[i], truth)
                    model.update(t, cache_value[i])
                    last_pred[i] = None
                    cache_value[i] = np.array([])
                    current_msg[i, x] = 1
                    cumulative_msg[i, x] = np.sum(current_msg[i, : x + 1])
                    monitored_values[i] = np.append(monitored_values[i], truth)

                else:
                    cache_value[i] = np.append(cache_value[i], truth)
                    last_pred[i] = pred[i]
                    current_msg[i, x] = 0
                    cumulative_msg[i, x] = np.sum(current_msg[i, : x + 1])
                    monitored_values[i] = np.append(monitored_values[i], pred[i])

            abs_errors[i] = np.append(abs_errors[i], abs(monitored_values[i][-1] - truth))
            relative_errors[i, x] = abs_errors[i][-1] / threshold
            current_acc[i, x] = 1 - relative_errors[i, x]
            cumulative_acc[i, x] = np.mean(current_acc[i, : x + 1])
            current_model[x] = selected_idx

    return cumulative_msg, abs_errors, current_model
