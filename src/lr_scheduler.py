import torch


def create_lr_scheduler(
    method: str,
    optimizer: torch.optim.Optimizer,
    dataset_length: int,
    batch_size: int,
    epochs: int,
):
    one_epoch_steps = (dataset_length - 1) // batch_size + 1
    total_training_steps = one_epoch_steps * epochs

    if method == "warmup":
        warmup_steps = total_training_steps * 0.1

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1.0, warmup_steps))
            return 1.0

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        lr_dict = {"scheduler": lr_scheduler, "interval": "step"}

        return lr_dict

    elif method == "linear_warmup":
        warmup_steps = total_training_steps * 0.1

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1.0, warmup_steps))
            return max(
                0.0,
                float(total_training_steps - current_step)
                / float(max(1, total_training_steps - warmup_steps)),
            )

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        lr_dict = {"scheduler": lr_scheduler, "interval": "step"}

        return lr_dict

    elif method == "constant":

        def lr_lambda(current_step: int) -> float:
            return 1.0

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        lr_dict = {"scheduler": lr_scheduler, "interval": "step"}

        return lr_dict

    else:
        raise ValueError(f"No such lr schedulering method: {method}")
