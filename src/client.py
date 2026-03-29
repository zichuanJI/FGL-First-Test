import copy

import torch
import torch.nn.functional as F


def _fedprox_penalty(local_model, global_params):
    penalty = torch.zeros(1, device=next(local_model.parameters()).device)
    for local_param, global_param in zip(local_model.parameters(), global_params):
        penalty = penalty + torch.sum((local_param - global_param) ** 2)
    return penalty.squeeze(0)


def train_local(
    model,
    data,
    client_mask,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    local_epochs: int = 1,
    algorithm: str = "fedavg",
    prox_mu: float = 0.01,
):
    """Train one client from a deepcopy of the global model."""
    local_model = copy.deepcopy(model)
    num_samples = int(client_mask.sum())

    if num_samples == 0:
        return local_model, {
            "num_samples": 0,
            "last_loss": 0.0,
            "last_ce_loss": 0.0,
            "last_prox_penalty": 0.0,
        }

    optimizer = torch.optim.Adam(local_model.parameters(), lr=lr, weight_decay=weight_decay)
    global_params = None
    if algorithm == "fedprox":
        global_params = [param.detach().clone() for param in model.parameters()]
    elif algorithm != "fedavg":
        raise ValueError(f"Unknown algorithm: {algorithm}")

    local_model.train()
    last_loss = 0.0
    last_ce_loss = 0.0
    last_prox_penalty = 0.0

    for _ in range(local_epochs):
        optimizer.zero_grad()
        out = local_model(data.x, data.edge_index)
        ce_loss = F.cross_entropy(out[client_mask], data.y[client_mask])
        loss = ce_loss
        prox_penalty = torch.zeros(1, device=out.device).squeeze(0)

        if global_params is not None and prox_mu > 0:
            prox_penalty = _fedprox_penalty(local_model, global_params)
            loss = loss + 0.5 * prox_mu * prox_penalty

        loss.backward()
        optimizer.step()

        last_loss = float(loss.item())
        last_ce_loss = float(ce_loss.item())
        last_prox_penalty = float(prox_penalty.item())

    return local_model, {
        "num_samples": num_samples,
        "last_loss": last_loss,
        "last_ce_loss": last_ce_loss,
        "last_prox_penalty": last_prox_penalty,
    }
