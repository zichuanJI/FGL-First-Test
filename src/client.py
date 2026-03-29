import copy
import torch
import torch.nn.functional as F


def train_local(model, data, client_mask, lr=0.01, weight_decay=5e-4, local_epochs=1):
    """
    对单个 client 做本地训练。
    
    参数：
    - model: 全局模型（会被复制，不直接原地修改）
    - data: 图数据
    - client_mask: 该 client 的训练节点 mask
    - lr: 学习率
    - weight_decay: 权重衰减
    - local_epochs: 本地训练轮数
    
    返回：
    - local_model: 本地训练后的模型
    - num_samples: 该 client 的训练样本数
    """
    local_model = copy.deepcopy(model)
    optimizer = torch.optim.Adam(local_model.parameters(), lr=lr, weight_decay=weight_decay)

    local_model.train()

    for epoch in range(local_epochs):
        optimizer.zero_grad()

        out = local_model(data.x, data.edge_index)
        loss = F.cross_entropy(out[client_mask], data.y[client_mask])

        loss.backward()
        optimizer.step()

    num_samples = int(client_mask.sum())

    return local_model, num_samples