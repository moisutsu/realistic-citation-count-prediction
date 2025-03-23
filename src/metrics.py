import torch


def top_k_percent_acc(k: int, golds: list[float], predicts: list[float]) -> float:
    top_k_length = int(len(golds) * k // 100)

    if top_k_length == 0:
        return 0.0

    golds = torch.argsort(torch.tensor(golds), descending=True).tolist()
    predicts = torch.argsort(torch.tensor(predicts), descending=True).tolist()

    top_k_golds = golds[:top_k_length]
    top_k_predicts = predicts[:top_k_length]

    acc = (
        sum([1 for predict in top_k_predicts if predict in top_k_golds]) / top_k_length
    )

    return acc
