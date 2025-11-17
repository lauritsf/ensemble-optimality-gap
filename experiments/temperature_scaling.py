import math

import torch
import typer
import wandb
from torch import Tensor, nn

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
    val_pct: float = typer.Option(0.1, help="Validation percentage (should be above 0.)"),
    seed: int = typer.Option(42, help="Seed for reproducibility"),
    ensemble_size: int = typer.Option(4, help="Number of models in the ensemble"),
    num_epochs: int = typer.Option(100, help="Number of epochs to train each model"),
    learning_rate: float = typer.Option(0.1, help="Learning rate for training"),
    weight_decay: float = typer.Option(7.50e-4, help="Weight decay value to use for training"),
    wandb_project: str = typer.Option("nmlb - temperature scaling", help="Wandb project name"),
    batch_size: int = typer.Option(128, help="Batch size"),
    batch_strategy: str = typer.Option("random", help="Batch strategy (same or random)"),
    holdout_strategy: str = typer.Option("same", help="Holdout strategy (same, random, disjoint or overlapping."),
    last_batch_strategy: str = typer.Option("keep", help="Last batch strategy (drop, keep or rescale)"),
    # GNN settings
    hidden_channels: int = typer.Option(64, help="Hidden channels for GNN models"),
    num_layers: int = typer.Option(4, help="Number of layers for GNN models"),
    batchensemble_fast_weight_init: float = typer.Option(
        0.5, help="Batchensemble fast weight initialization (random sign if positive, gaussian if negative)"
    ),
    run_temperature_sweep: str = typer.Option(
        "False", help="Run temperature sweep for the best early stopping models (True or False)"
    ),
):
    """First train an ensemble of models, evaluate their performance with/without temperature scaling."""
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

    # Set random seed
    torch.manual_seed(seed)

    # --- Load model ---
    model = load_ensemble_model(
        model_name,
        dataset_name,
        ensemble_size,
        NUM_CLASSES[dataset_name],
        NUM_FEATURES[dataset_name],
        gnn_kwargs={"hidden_channels": hidden_channels, "num_layers": num_layers} if model_name in ["GCN"] else None,
        batchensemble_fast_weight_init=batchensemble_fast_weight_init,
    )
    model.to(device)

    # --- Load dataset ---
    is_graph_dataset = dataset_name in ["NCI1"]
    if is_graph_dataset:
        train_loader, val_loader, test_loader = create_tud_loaders(
            dataset_name, holdout_strategy, batch_strategy, val_pct, batch_size, seed, ensemble_size, device, drop_last
        )
    elif dataset_name in ["cifar10", "cifar10_subset"]:
        train_loader, val_loader, test_loader = create_cifar_loaders(
            dataset_name, holdout_strategy, batch_strategy, val_pct, batch_size, seed, ensemble_size, device, drop_last
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
        raise ValueError(f"Dataset {dataset_name} not supported")

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
        ensemble_size, compute_joint_metrics=False, device=device, num_classes=NUM_CLASSES[dataset_name]
    )
    val_metrics = EnsembleMetrics(
        ensemble_size,
        compute_joint_metrics=holdout_strategy in ["same", "overlapping"],
        use_masked_updates=holdout_strategy == "overlapping",
        device=device,
        num_classes=NUM_CLASSES[dataset_name],
    )
    test_metrics = EnsembleMetrics(
        ensemble_size,
        device=device,
        num_classes=NUM_CLASSES[dataset_name],
    )

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
        "last_batch_strategy": last_batch_strategy,
        "run_temperature_sweep": run_temperature_sweep,
    }
    if is_graph_dataset:
        config.update(
            {
                "hidden_channels": hidden_channels,
                "num_layers": num_layers,
            }
        )
    long_run_name = "_".join([f"{k}_{v}" for k, v in config.items()])
    run = wandb.init(
        project=wandb_project, name=long_run_name, config=config, reinit="finish_previous", job_type="training"
    )

    if last_batch_strategy == "rescale":

        def criterion(logits, targets):
            return nn.functional.cross_entropy(logits, targets, reduction="sum") / batch_size
    else:
        criterion = nn.CrossEntropyLoss(reduction="mean")

    for epoch in range(num_epochs):
        model.train()
        train_metrics.reset()
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

        model.eval()
        with torch.no_grad():
            val_metrics.reset()
            for batch in val_loader:
                if is_graph_dataset:
                    if holdout_strategy == "overlapping":
                        batch, mask = batch
                    else:
                        mask = None
                    y = batch.y  # type: ignore
                    logits = model(batch)
                elif model_name in ["bilstm"]:
                    x, y, attention_mask, *mask = batch  # type: ignore
                    logits = model(x, attention_mask)
                    mask = mask[0] if mask else None
                else:
                    x, y, *mask = batch  # type: ignore
                    mask = mask[0] if mask else None
                    logits = model(x)
                val_metrics.update(logits, y, mask)
            test_metrics.reset()
            for batch in test_loader:
                if is_graph_dataset:
                    y = batch.y  # type: ignore
                    logits = model(batch)
                elif model_name in ["bilstm"]:
                    x, y, attention_mask = batch  # type: ignore
                    logits = model(x, attention_mask)
                else:
                    x, y = batch  # type: ignore
                    logits = model(x)
                test_metrics.update(logits, y)

        metrics = {f"train/{k}": v for k, v in train_metrics.compute().items()}
        metrics.update({f"val/{k}": v for k, v in val_metrics.compute().items()})
        metrics.update({f"test/{k}": v for k, v in test_metrics.compute().items()})
        metrics["lr"] = scheduler.get_last_lr()[0]
        run.log(metrics)
        scheduler.step()
    run.finish()

    # --- Temperature Scaling Evaluation ---
    print("\n--- Starting Temperature Scaling Evaluation ---")
    val_logits, val_targets, holdout_mask = get_logits(
        model, val_loader, ensemble_size, is_graph_dataset, holdout_strategy == "overlapping", model_name
    )

    is_joint_possible = holdout_strategy in ["same", "overlapping"]

    # --- Find Optimal Temperatures ---
    print("--- Finding Optimal Temperatures ---")
    temperatures_to_evaluate = {
        "none": None,
        "individual": find_temperature_for_individual_models(val_logits, val_targets, holdout_mask),
    }
    if is_joint_possible:
        temperatures_to_evaluate["ensemble"] = find_temperature_for_ensemble(
            val_logits, val_targets, holdout_mask, share_temperature=False
        )
        temperatures_to_evaluate["ensemble_shared"] = find_temperature_for_ensemble(
            val_logits, val_targets, holdout_mask, share_temperature=True
        )
        temperatures_to_evaluate["ensemble_log_probs"] = find_temperature_for_log_ensemble_probs(
            val_logits, val_targets, holdout_mask
        )

    print("--- Running Evaluation with Optimal Temperatures ---")
    test_logits_batches, test_targets_batches = get_all_logits_and_targets_in_batches(
        model, test_loader, is_graph_dataset, model_name
    )
    for temp_name, temp_value in temperatures_to_evaluate.items():
        temp_target = "ensemble_log_probs" if temp_name == "ensemble_log_probs" else "model_logits"
        print(f"Evaluating with temperature scaling: {temp_name}")
        run = wandb.init(
            project=wandb_project,
            name=f"{long_run_name}_{temp_name}",
            reinit="finish_previous",
            job_type="evaluation",
            config={**config, "temperature_scaling": temp_name},
        )
        test_metrics.reset()
        with torch.no_grad():
            for logits, y in zip(test_logits_batches, test_targets_batches):
                test_metrics.update(
                    logits,
                    y,
                    temperature=temp_value.to(device) if temp_value is not None else None,
                    temperature_target=temp_target,
                )
        metrics = {f"test/{k}": v for k, v in test_metrics.compute().items()}
        run.log(metrics)
        run.finish()

    # --- Temperature Sweep Evaluation ---
    if run_temperature_sweep.lower() == "true" and is_joint_possible:
        print("\n--- Running Temperature Sweep Evaluation ---")
        temperature_targets = ["model_logits", "ensemble_log_probs"]
        for temperature_target in temperature_targets:
            print(f"--- Sweeping temperature for target: {temperature_target} ---")
            sweep_run = wandb.init(
                project=wandb_project,
                name=f"{long_run_name}_temp_sweep_{temperature_target}",
                reinit="finish_previous",
                job_type="temperature_sweep",
                config={**config, "temperature_scaling": "sweep", "temperature_target": temperature_target},
            )
            temp_sweep_values = torch.cat([torch.arange(0.1, 1.0, 0.1), torch.arange(1.0, 5.25, 0.25)])
            for temp_val in temp_sweep_values:
                shared_temp = torch.tensor([temp_val], device=device)
                test_metrics.reset()
                with torch.no_grad():
                    for logits, y in zip(test_logits_batches, test_targets_batches):
                        test_metrics.update(logits, y, temperature=shared_temp, temperature_target=temperature_target)
                metrics = {f"test/{k}": v for k, v in test_metrics.compute().items()}
                metrics["temperature"] = temp_val
                sweep_run.log(metrics)
            sweep_run.finish()


@torch.no_grad()
def get_all_logits_and_targets_in_batches(ensemble_model, data_loader, is_graph_dataset, model_name):
    """Get all model logits and targets from a data loader and store them as lists of batches."""
    all_logits_batches = []
    all_targets_batches = []
    for batch in data_loader:
        if is_graph_dataset:
            y = batch.y
            logits = ensemble_model(batch)
        elif model_name in ["bilstm"]:
            x, y, attention_mask, *_ = batch
            logits = ensemble_model(x, attention_mask)
        else:
            x, y, *_ = batch
            logits = ensemble_model(x)
        all_logits_batches.append(logits)
        all_targets_batches.append(y)
    return all_logits_batches, all_targets_batches


@torch.no_grad()
def get_logits(
    ensemble_model, data_loader, ensemble_size, is_graph_dataset: bool, use_holdout_mask: bool, model_name: str
):
    """Get the model logits for optimizing temperature scaling."""
    logits = []
    targets = []
    holdout_masks = []
    for batch in data_loader:
        if is_graph_dataset:
            if use_holdout_mask:
                batch, holdout_mask = batch
            else:
                holdout_mask = None
            y = batch.y
            raw_logits = ensemble_model(batch)
        elif model_name in ["bilstm"]:
            x, y, attention_mask, *mask_data = batch
            holdout_mask = mask_data[0] if mask_data else None
            raw_logits = ensemble_model(x, attention_mask)
        else:
            x, y, *mask_data = batch
            holdout_mask = mask_data[0] if mask_data else None
            raw_logits = ensemble_model(x)

        ensemble_batch_size, *other_dims = raw_logits.shape
        batch_size = ensemble_batch_size // ensemble_size
        logits.append(raw_logits.view(ensemble_size, batch_size, *other_dims))
        targets.append(y.view(ensemble_size, batch_size))
        if holdout_mask is not None:
            holdout_masks.append(holdout_mask.T)

    return (
        torch.cat(logits, dim=1),
        torch.cat(targets, dim=1),
        torch.cat(holdout_masks, dim=1) if holdout_masks else None,
    )


def find_temperature_for_individual_models(
    logits: Tensor,  # shape: [ensemble_size, num_samples, num_classes]
    targets: Tensor,  # shape: [ensemble_size, num_samples]
    holdout_mask: Tensor | None = None,  # shape: [ensemble_size, num_samples]
):
    ensemble_size = logits.size(0)
    temperatures = nn.Parameter(torch.ones(ensemble_size, dtype=logits.dtype, device=logits.device))
    optimizer = torch.optim.LBFGS([temperatures], lr=0.1, max_iter=100)  # type: ignore
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        logits_scaled = logits / temperatures.view(-1, 1, 1)
        if holdout_mask is None:
            logits_scaled = logits_scaled.view(-1, logits_scaled.size(-1))
            loss = criterion(logits_scaled, targets.view(-1))
        else:
            loss = criterion(logits_scaled[holdout_mask], targets[holdout_mask])
        loss.backward()
        return loss

    optimizer.step(closure)
    return temperatures


def find_temperature_for_ensemble(
    logits: Tensor,  # shape: [ensemble_size, num_samples, num_classes]
    targets: Tensor,  # shape: [ensemble_size, num_samples]
    holdout_mask: Tensor | None = None,  # shape: [ensemble_size, num_samples]
    share_temperature: bool = False,
):
    ensemble_size = logits.size(0)
    num_temperature_params = 1 if share_temperature else ensemble_size
    temperature = nn.Parameter(torch.ones(num_temperature_params, dtype=logits.dtype, device=logits.device))
    optimizer = torch.optim.LBFGS([temperature], lr=0.1, max_iter=100)  # type: ignore

    if holdout_mask is not None:
        holdout_mask = holdout_mask.unsqueeze(-1)

    if not (targets == targets[0]).all():
        raise ValueError("Targets must be the same for all models in the ensemble")
    else:
        targets = targets[0]

    criterion = nn.NLLLoss()

    def closure():
        optimizer.zero_grad()
        logits_scaled = logits / temperature.view(-1, 1, 1)
        model_log_probs = logits_scaled.log_softmax(dim=-1)
        if holdout_mask is not None:
            model_log_probs_masked = model_log_probs.masked_fill(~holdout_mask, float("-inf"))
            num_models = holdout_mask.sum(dim=0)
            ensemble_log_probs = torch.logsumexp(model_log_probs_masked, dim=0) - num_models.log()
        else:
            ensemble_log_probs = model_log_probs.logsumexp(dim=0) - math.log(ensemble_size)
        loss = criterion(ensemble_log_probs, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return temperature


def find_temperature_for_log_ensemble_probs(
    logits: Tensor,  # shape: [ensemble_size, num_samples, num_classes]
    targets: Tensor,  # shape: [ensemble_size, num_samples]
    holdout_mask: Tensor | None,  # shape: [ensemble_size, num_samples] or None
):
    """Find a temperature that is applied to the log probabilities of the ensemble."""
    ensemble_size = logits.size(0)
    temperature = nn.Parameter(torch.ones(1, dtype=logits.dtype, device=logits.device))
    optimizer = torch.optim.LBFGS([temperature], lr=0.1, max_iter=100)  # type: ignore

    if holdout_mask is not None:
        holdout_mask = holdout_mask.unsqueeze(-1)

    if not (targets == targets[0]).all():
        raise ValueError("Targets for ensemble models must be the same")
    targets = targets[0]

    criterion = nn.NLLLoss()
    model_log_probs_unscaled = logits.log_softmax(dim=-1)

    def closure():
        optimizer.zero_grad()
        if holdout_mask is None:
            ensemble_log_probs_unscaled = model_log_probs_unscaled.logsumexp(dim=0) - math.log(ensemble_size)
        else:
            model_log_probs_unscaled_masked = model_log_probs_unscaled.masked_fill(~holdout_mask, float("-inf"))
            num_valid_models_per_sample = holdout_mask.sum(dim=0)
            ensemble_log_probs_unscaled = (
                torch.logsumexp(model_log_probs_unscaled_masked, dim=0) - num_valid_models_per_sample.log()
            )

        ensemble_log_probs_scaled = ensemble_log_probs_unscaled / temperature.view(-1, 1)
        ensemble_log_probs_scaled = ensemble_log_probs_scaled.softmax(dim=-1).clamp(min=1e-9).log()
        loss = criterion(ensemble_log_probs_scaled, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return temperature


if __name__ == "__main__":
    typer.run(main)
