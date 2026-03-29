import copy

import torch



def fedavg(local_models, sample_counts):
    """Weighted FedAvg aggregation."""
    assert local_models, "local_models cannot be empty"
    assert len(local_models) == len(sample_counts), "local_models and sample_counts must match"

    total_samples = sum(sample_counts)
    global_model = copy.deepcopy(local_models[0])
    global_state_dict = global_model.state_dict()

    for key in global_state_dict:
        global_state_dict[key] = torch.zeros_like(global_state_dict[key])

    for model, num_samples in zip(local_models, sample_counts):
        local_state_dict = model.state_dict()
        weight = num_samples / total_samples
        for key in global_state_dict:
            global_state_dict[key] += local_state_dict[key] * weight

    global_model.load_state_dict(global_state_dict)
    return global_model
