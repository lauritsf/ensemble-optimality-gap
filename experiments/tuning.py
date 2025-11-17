import math

import torch
import typer
import wandb
from torch import nn

from ensemble_optimality_gap.data import (
    NUM_CLASSES,
    NUM_FEATURES,
    create_cifar_loaders,
    create_covertype_loaders,
    create_setfit_loaders,
    create_tud_loaders,
)
from ensemble_optimality_gap.ensembles import EnsembleMetrics, load_ensemble_model


def main(
    model_name: str = typer.Option(
        "wrn-16-4",
        help="Model name (wrn-16-4, GCN, wrn-16-4-be, mlp, bilstm)",
    ),
    dataset_name: str = typer.Option(
        "cifar10_subset",
        help="Dataset name (cifar10, cifar10_subset, NCI1, covertype, ag_news",
    ),
    val_pct: float = typer.Option(0.1, help="Validation percentage (0. to use test data)"),
    seed: int = typer.Option(42, help="Seed for reproducibility"),
    ensemble_size: int = typer.Option(4, help="Number of models in the ensemble"),
    num_epochs: int = typer.Option(100, help="Number of epochs to train each model"),
    learning_rate: float = typer.Option(0.1, help="Learning rate for training"),
    weight_decay_min: float = typer.Option(5e-4, help="Minimum weight decay value to evaluate"),
    weight_decay_max: float = typer.Option(5e-4, help="Maximum weight decay value to evaluate"),
    weight_decay_steps: int = typer.Option(1, help="Number of weight decay values to evaluate"),
    wandb_project: str = typer.Option("nmlb - tuning", help="Wandb project name"),
    batch_size: int = typer.Option(128, help="Batch size"),
    batch_strategy: str = typer.Option("random", help="Batch strategy (same or random)"),
    holdout_strategy: str = typer.Option("same", help="Holdout strategy (same, random, disjoint or overlapping."),
    last_batch_strategy: str = typer.Option("keep", help="Last batch strategy (drop, keep or rescale)"),
    # GNN settings
    hidden_channels: int = typer.Option(64, help="Hidden channels for GNN models"),
    num_layers: int = typer.Option(4, help="Number of layers for GNN models"),
):
    """
    Trains an ensemble of models on a dataset with a number of different
    weight decay values to determine the best value.

    We evaluate the models both individually and jointly when possible.
    """
    if batch_strategy == "same" and holdout_strategy != "same":
        raise ValueError("Cannot use same batch strategy unless holdout strategy is also same")

    valid_model_dataset_combinations = {
        "wrn-16-4": ["cifar10", "cifar10_subset"],
        "wrn-16-4-be": ["cifar10", "cifar10_subset"],
        "GCN": ["NCI1"],
        "mlp": ["covertype"],
        "bilstm": ["ag_news"],
    }

    if model_name not in valid_model_dataset_combinations:
        raise ValueError(f"Model {model_name} not supported")
    if dataset_name not in valid_model_dataset_combinations[model_name]:
        raise ValueError(f"Dataset {dataset_name} not supported for model {model_name}")
    drop_last = last_batch_strategy == "drop"
    if last_batch_strategy not in ["drop", "keep", "rescale"]:
        raise ValueError("Last batch strategy must be drop, keep or rescale")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if weight_decay_min == weight_decay_max or weight_decay_steps == 1:
        weight_decay_values = [weight_decay_min]
    else:
        weight_decay_values = torch.logspace(
            start=math.log10(weight_decay_min), end=math.log10(weight_decay_max), steps=weight_decay_steps
        )
    for i, weight_decay in enumerate(weight_decay_values):
        print(f"Weight decay ({i + 1}/{len(weight_decay_values)}): {weight_decay:.2e}")
        # Set random seed
        torch.manual_seed(seed)

        # --- Load model ---
        model = load_ensemble_model(
            model_name,
            dataset_name,
            ensemble_size,
            NUM_CLASSES[dataset_name],
            NUM_FEATURES[dataset_name],
            gnn_kwargs={
                "hidden_channels": hidden_channels,
                "num_layers": num_layers,
            }
            if model_name in ["GCN"]
            else None,
        )
        model.to(device)

        # --- Load dataset ---
        is_graph_dataset = dataset_name in ["NCI1"]
        num_classes = NUM_CLASSES[dataset_name]
        if is_graph_dataset:
            train_loader, val_loader, test_loader = create_tud_loaders(
                dataset_name,
                holdout_strategy,
                batch_strategy,
                val_pct,
                batch_size,
                seed,
                ensemble_size,
                device,
                drop_last,
            )
        elif dataset_name in ["cifar10", "cifar10_subset"]:
            train_loader, val_loader, test_loader = create_cifar_loaders(
                dataset_name,
                holdout_strategy,
                batch_strategy,
                val_pct,
                batch_size,
                seed,
                ensemble_size,
                device,
                drop_last,
            )
        elif dataset_name == "covertype":
            train_loader, val_loader, test_loader = create_covertype_loaders(
                holdout_strategy, batch_strategy, val_pct, batch_size, seed, ensemble_size, device, drop_last
            )
        elif dataset_name in ["ag_news"]:
            train_loader, val_loader, test_loader = create_setfit_loaders(
                holdout_strategy,
                batch_strategy,
                val_pct,
                batch_size,
                seed,
                ensemble_size,
                device,
                drop_last,
                dataset_name,
                tokenizer="gpt2",
            )
        else:
            raise ValueError(f"Dataset {dataset_name} not supported for loading")
        eval_loader = val_loader if val_pct > 0 else test_loader

        # --- Setup logging ---
        config = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "val_pct": val_pct,
            "seed": seed,
            "ensemble_size": ensemble_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "batch_strategy": batch_strategy,
            "holdout_strategy": holdout_strategy,
            "use_test": val_pct == 0,
            "last_batch_strategy": last_batch_strategy,
        }
        if is_graph_dataset:
            config.update(
                {
                    "hidden_channels": hidden_channels,
                    "num_layers": num_layers,
                }
            )
        run_name = f"{model_name}_{dataset_name}_wd_{weight_decay}"
        job_type = "tuning" if val_pct > 0 else "test"
        run = wandb.init(
            project=wandb_project, name=run_name, config=config, reinit="finish_previous", job_type=job_type
        )

        # --- initialize optimizer, scheduler and metrics ---
        wd_parameters = []
        non_wd_parameters = []
        for name, param in model.named_parameters():
            if model_name in ["wrn-16-4-be", "wrn-28-10-be"] and ("alpha_param" in name or "gamma_param" in name):
                non_wd_parameters.append(param)
            else:
                wd_parameters.append(param)
        optimizer = torch.optim.SGD(
            [
                {"params": wd_parameters, "weight_decay": weight_decay},
                {"params": non_wd_parameters, "weight_decay": 0.0},
            ],
            lr=learning_rate,
            momentum=0.9,
        )  # type: ignore
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

        train_metrics = EnsembleMetrics(
            ensemble_size, compute_joint_metrics=False, device=device, num_classes=num_classes
        )
        if val_pct > 0:  # uses validation set
            eval_metrics = EnsembleMetrics(
                ensemble_size,
                compute_joint_metrics=holdout_strategy in ["same", "overlapping"],
                use_masked_updates=holdout_strategy == "overlapping",
                device=device,
                num_classes=num_classes,
            )
        else:  # uses test set
            eval_metrics = EnsembleMetrics(
                ensemble_size,
                device=device,
                num_classes=num_classes,
            )

        if last_batch_strategy == "rescale":
            # To avoid putting too much weight on an underfull batch, we weight down underfull batches
            def criterion(logits, y):
                return nn.functional.cross_entropy(logits, y, reduction="sum") / batch_size
        else:
            criterion = nn.CrossEntropyLoss(reduction="mean")

        for epoch in range(num_epochs):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                if is_graph_dataset:
                    y = batch.y  # type: ignore
                    logits = model(batch)
                elif model_name in ["bilstm"]:
                    x, y, attention_mask, *_ = batch  # type: ignore
                    logits = model(x, attention_mask)
                else:
                    x, y, *_ = batch  # type: ignore
                    logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                train_metrics.update(logits, y)

            # eval
            model.eval()
            with torch.no_grad():
                for batch in eval_loader:
                    if is_graph_dataset:
                        if holdout_strategy == "overlapping" and val_pct > 0:
                            batch, mask = batch  # type: ignore
                        else:
                            mask = None
                        y = batch.y  # type: ignore
                        logits = model(batch)
                    elif model_name in ["bilstm"]:
                        x, y, attention_mask, *mask = batch  # type: ignore
                        logits = model(x, attention_mask)
                        mask = mask if mask else None
                    else:
                        x, y, *mask = batch  # type: ignore
                        logits = model(x)
                        mask = mask[0] if mask else None
                    eval_metrics.update(logits, y, mask)

            # log metrics
            eval_key = "val" if val_pct > 0 else "test"
            metrics = {f"train/{k}": v for k, v in train_metrics.compute().items()}
            metrics.update({f"{eval_key}/{k}": v for k, v in eval_metrics.compute().items()})
            metrics["lr"] = scheduler.get_last_lr()[0]
            run.log(metrics)

            # reset metrics
            train_metrics.reset()
            eval_metrics.reset()

            # update learning rate
            scheduler.step()

        run.finish()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    typer.run(main)
