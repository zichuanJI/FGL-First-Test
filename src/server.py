import copy
import torch


def fedavg(local_models, sample_counts):
    """
    对多个 client 的本地模型做 FedAvg 聚合。

    参数：
    - local_models: list，每个元素是一个训练后的本地模型
    - sample_counts: list，每个元素是对应 client 的样本数

    返回：
    - global_model: 聚合后的全局模型
    """
    assert len(local_models) > 0, "local_models 不能为空"
    assert len(local_models) == len(sample_counts), "模型数和样本数必须一致"

    total_samples = sum(sample_counts)

    # 以第一个模型为模板创建全局模型
    global_model = copy.deepcopy(local_models[0])
    global_state_dict = global_model.state_dict()

    # 先把所有参数清零
    for key in global_state_dict:
        global_state_dict[key] = torch.zeros_like(global_state_dict[key])

    # 按样本数加权平均
    for model, num_samples in zip(local_models, sample_counts):
        local_state_dict = model.state_dict()
        weight = num_samples / total_samples

        for key in global_state_dict:
            global_state_dict[key] += local_state_dict[key] * weight

    global_model.load_state_dict(global_state_dict)

    return global_model