import copy
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
from ensemble_optimality_gap.ensembles import Ensemble, EnsembleMetrics, GraphEnsemble, load_ensemble_model


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
    learning_rate: float = typer.Option(0.001, help="Learning rate for training"),
    patience: int = typer.Option(10, help="Early stopping patience"),
    wandb_project: str = typer.Option("nmlb - early stopping", help="Wandb project name"),
    batch_size: int = typer.Option(128, help="Batch size"),
    batch_strategy: str = typer.Option("random", help="Batch strategy (same or random)"),
    holdout_strategy: str = typer.Option("same", help="Holdout strategy (same, random, disjoint or overlapping."),
    epoch_termination: str = typer.Option(
        "early_stopping", help="Epoch termination strategy (early_stopping or max_epochs)"
    ),
    last_batch_strategy: str = typer.Option("keep", help="Last batch strategy (drop, keep or rescale)"),
    # GNN settings
    hidden_channels: int = typer.Option(64, help="Hidden channels for GNN models"),
    num_layers: int = typer.Option(4, help="Number of layers for GNN models"),
    # Batchensemble settings
    batchensemble_fast_weight_init: float = typer.Option(
        0.5, help="Batchensemble fast weight initialization (random sign if positive, gaussian if negative)"
    ),
    run_temperature_sweep: str = typer.Option(
        "False", help="Run temperature sweep for the best early stopping models (True or False)"
    ),
):
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
    if epoch_termination not in ["early_stopping", "max_epochs"]:
        raise ValueError(f"Invalid epoch_termination strategy: {epoch_termination}")
    drop_last = last_batch_strategy == "drop"
    if last_batch_strategy not in ["drop", "keep", "rescale"]:
        raise ValueError("Last batch strategy must be drop, keep or rescale")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")

    # Set random seed
    torch.manual_seed(seed)
    print(f"--- Seed set to {seed} ---")

    # --- Load model ---
    print(f"--- Loading model: {model_name} ---")
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
        batchensemble_fast_weight_init=batchensemble_fast_weight_init,
    )
    model.to(device)

    # --- Load dataset ---
    print(f"--- Loading dataset: {dataset_name} ---")
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
        raise ValueError(f"Dataset {dataset_name} not supported for early stopping experiments")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # type: ignore

    # setup early stopping
    if is_graph_dataset:
        ensemble_type = "graph"
    elif model_name.endswith("-be"):
        ensemble_type = "batchensemble"
    else:
        ensemble_type = "ensemble"
    early_stopping_manager = EarlyStoppingManager(patience, ensemble_size, holdout_strategy, ensemble_type)

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

    # --- Initialize wandb ---
    config = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "val_pct": val_pct,
        "seed": seed,
        "ensemble_size": ensemble_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "patience": patience,
        "wandb_project": wandb_project,
        "batch_size": batch_size,
        "batch_strategy": batch_strategy,
        "holdout_strategy": holdout_strategy,
        "epoch_termination": epoch_termination,
        "last_batch_strategy": last_batch_strategy,
        "batchensemble_fast_weight_init": batchensemble_fast_weight_init,
    }
    if is_graph_dataset:
        config.update(
            {
                "hidden_channels": hidden_channels,
                "num_layers": num_layers,
            }
        )
    long_run_name = "_".join([f"{k}_{v}" for k, v in config.items()])
    print("--- Initializing WandB for Training ---")
    run = wandb.init(
        project=wandb_project, name=long_run_name, config=config, reinit="finish_previous", job_type="training"
    )

    if last_batch_strategy == "rescale":
        # To avoid putting too much weight on an underfull batch, we weight down underfull batches
        def criterion(logits, y):
            return nn.functional.cross_entropy(logits, y, reduction="sum") / batch_size
    else:
        criterion = nn.CrossEntropyLoss(reduction="mean")

    print(f"--- Starting Training for {num_epochs} epochs ---")
    for epoch in range(num_epochs):
        # --- Train ---
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

        # --- Validate ---
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
                    logits = model(x)
                    mask = mask[0] if mask else None
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

        # --- Log metrics ---
        metrics = {f"train/{k}": v for k, v in train_metrics.compute().items()}
        metrics.update({f"val/{k}": v for k, v in val_metrics.compute().items()})
        metrics.update({f"test/{k}": v for k, v in test_metrics.compute().items()})
        metrics["lr"] = optimizer.param_groups[0]["lr"]
        run.log(metrics)
        train_nll = train_metrics.compute().get("model_0/nll", 0)
        train_accuracy = train_metrics.compute().get("model_0/accuracy", 0)
        val_nll = val_metrics.compute().get("model_0/nll", 0)
        val_accuracy = val_metrics.compute().get("model_0/accuracy", 0)

        print(
            f"Epoch {epoch + 1}/{num_epochs} - Model 1 - "
            f"Train NLL: {train_nll:.4f}, Train Accuracy: {train_accuracy:.4f}, "
            f"Val NLL: {val_nll:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )

        # --- Early stopping ---
        early_stopping_manager.update(val_metrics.compute(), model, epoch)
        # Check if we should stop training
        if epoch_termination == "early_stopping" and early_stopping_manager.should_stop():
            print(f"--- Early stopping triggered at epoch {epoch + 1} ---")
            break
    run.finish()

    # --- Evaluate best models ---
    print("\n--- Starting Evaluation of Best Models ---")
    early_stopping_ensembles = early_stopping_manager.get_best_models()
    early_stopping_epochs = early_stopping_manager.get_best_epochs()
    for name, ensemble in early_stopping_ensembles.items():
        print(f"\n--- Evaluating ensemble: {name} ---")
        avg_epoch = early_stopping_epochs[name.replace("model", "epoch")]
        individual_epochs = {}
        if name == "individual_best_models":  # list of epoch_values
            for i, epoch in enumerate(avg_epoch):
                individual_epochs[f"epoch_stopped_{i}"] = epoch
            avg_epoch = sum(avg_epoch) / len(avg_epoch)

        ensemble.to(device)
        ensemble.eval()

        # Pre-compute test logits and targets to speed up evaluation loops
        print("--- Pre-computing test logits and targets as batches ---")
        test_logits_batches, test_targets_batches = get_all_logits_and_targets_in_batches(
            ensemble, test_loader, is_graph_dataset, model_name
        )

        # --- Unscaled Evaluation ---
        print("--- Running Unscaled Evaluation ---")
        run = wandb.init(
            project=wandb_project,
            name=f"{long_run_name}_{name}_unscaled",
            reinit="finish_previous",
            job_type="evaluation",
            config={
                **config,
                "early_stopping_ensemble": name,
                "epoch_stopped": avg_epoch,
                **individual_epochs,
                "temperature_scaling": "none",
            },
        )
        test_metrics.reset()
        with torch.no_grad():
            for logits, y in zip(test_logits_batches, test_targets_batches):
                test_metrics.update(logits, y)

        metrics = {f"test/{k}": v for k, v in test_metrics.compute().items()}
        metrics.update({"epoch_stopped": avg_epoch})
        if individual_epochs:
            metrics.update(individual_epochs)
        run.log(metrics)
        run.finish()

        # --- Temperature Scaling Evaluation ---
        print("--- Getting validation logits for temperature scaling ---")
        val_logits, val_targets, holdout_mask = get_logits(
            ensemble, val_loader, ensemble_size, is_graph_dataset, holdout_strategy == "overlapping", model_name
        )

        is_joint_possible = holdout_strategy in ["same", "overlapping"]

        # --- Optimal Temperature Evaluation ---
        print("--- Finding Optimal Temperatures ---")
        temperatures_to_evaluate = {
            "individual": find_temperature_for_individual_models(val_logits, val_targets, holdout_mask)
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
        for temp_name, temp_value in temperatures_to_evaluate.items():
            temp_target = "ensemble_log_probs" if temp_name == "ensemble_log_probs" else "model_logits"
            run = wandb.init(
                project=wandb_project,
                name=f"{long_run_name}_{name}_{temp_name}",
                reinit="finish_previous",
                job_type="evaluation",
                config={
                    **config,
                    "early_stopping_ensemble": name,
                    "epoch_stopped": avg_epoch,
                    **individual_epochs,
                    "temperature_scaling": temp_name,
                },
            )
            test_metrics.reset()
            with torch.no_grad():
                for logits, y in zip(test_logits_batches, test_targets_batches):
                    test_metrics.update(logits, y, temperature=temp_value.to(device), temperature_target=temp_target)
            metrics = {f"test/{k}": v for k, v in test_metrics.compute().items()}
            run.log(metrics)
            run.finish()

        # --- Temperature Sweep Evaluation ---
        if run_temperature_sweep.lower() == "true":
            temperature_targets = ["model_logits", "ensemble_log_probs"]
            for temperature_target in temperature_targets:
                print("--- Running Temperature Sweep Evaluation ---")
                sweep_run = wandb.init(
                    project=wandb_project,
                    name=f"{long_run_name}_{name}_temp_sweep",
                    reinit="finish_previous",
                    job_type="temperature_sweep",
                    config={
                        **config,
                        "early_stopping_ensemble": name,
                        "epoch_stopped": avg_epoch,
                        **individual_epochs,
                        "temperature_scaling": temperature_target,
                    },
                )
                temp_sweep_values = torch.cat([torch.arange(0.1, 1.0, 0.1), torch.arange(1.0, 5.25, 0.25)])
                for temp_val in temp_sweep_values:
                    shared_temp = torch.tensor([temp_val], device=device)
                    test_metrics.reset()
                    with torch.no_grad():
                        for logits, y in zip(test_logits_batches, test_targets_batches):
                            test_metrics.update(
                                logits, y, temperature=shared_temp, temperature_target=temperature_target
                            )
                    metrics = {f"test/{k}": v for k, v in test_metrics.compute().items()}
                    metrics["temperature"] = temp_val
                    sweep_run.log(metrics)
                sweep_run.finish()


class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.best_val_loss = float("inf")
        self.num_no_improvements = 0
        self.best_model = None
        self.stopped = False
        self.best_epoch = None

    def update(self, val_loss: float, model: nn.Module, epoch: int | None = None):
        if self.stopped:
            return

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.num_no_improvements = 0
            self.best_model = copy.deepcopy(model)
            self.best_epoch = epoch
        else:
            self.num_no_improvements += 1

        if self.num_no_improvements >= self.patience:
            self.stopped = True


class EarlyStoppingManager:
    def __init__(self, patience: int, ensemble_size: int, holdout_strategy: str, ensemble_type: str):
        self.patience = patience
        self.ensemble_size = ensemble_size
        self.holdout_strategy = holdout_strategy
        self.ensemble_type = ensemble_type

        self.individual_model_stoppers = self._initialize_individual_stoppers()
        self.shared_epoch_stopper = EarlyStopping(patience)
        self.ensemble_stoppers = self._initialize_ensemble_stoppers()

    def _initialize_individual_stoppers(self):
        # Create early stoppers for individual models if not using batch ensemble
        if self.ensemble_type != "batchensemble":
            return [EarlyStopping(self.patience) for _ in range(self.ensemble_size)]
        else:
            return []

    def _initialize_ensemble_stoppers(self):
        # Create early stoppers for ensemble models based on holdout strategy
        if self.holdout_strategy not in ["same", "overlapping"]:
            return []
        elif self.holdout_strategy == "overlapping" or self.ensemble_type == "batchensemble":
            return [EarlyStopping(self.patience)]
        elif self.holdout_strategy == "same":
            return [EarlyStopping(self.patience) for _ in range(self.ensemble_size)]
        else:
            raise RuntimeError("Should never reach here")

    def update(self, val_metrics_dict: dict, model, epoch: int):
        if self.individual_model_stoppers:  # if not using batch ensemble
            for i in range(self.ensemble_size):
                self.individual_model_stoppers[i].update(
                    val_metrics_dict[f"model_{i}/nll"],
                    model.models[i],
                    epoch,
                )
                if self.holdout_strategy == "same":
                    self.ensemble_stoppers[i].update(
                        val_metrics_dict[f"ensemble_{i + 1}/nll"],
                        model,
                        epoch,
                    )
        if self.holdout_strategy == "overlapping" or (self.ensemble_type == "batchensemble" and self.ensemble_stoppers):
            self.ensemble_stoppers[0].update(
                val_metrics_dict[f"ensemble_{self.ensemble_size}/nll"],
                model,
                epoch,
            )

        self.shared_epoch_stopper.update(
            torch.stack([val_metrics_dict[f"model_{i}/nll"] for i in range(self.ensemble_size)]).mean(),
            model,
            epoch,
        )

    def should_stop(self):
        all_stoppers = self.individual_model_stoppers + [self.shared_epoch_stopper] + self.ensemble_stoppers
        return all(stopper.stopped for stopper in all_stoppers)

    def get_best_models(self) -> dict:
        result = {}
        if self.individual_model_stoppers:
            individual_best_models = [stopper.best_model for stopper in self.individual_model_stoppers]
            if self.ensemble_type == "graph":
                result["individual_best_models"] = GraphEnsemble(individual_best_models)
            else:
                result["individual_best_models"] = Ensemble(individual_best_models)

        result["shared_epoch_best_model"] = self.shared_epoch_stopper.best_model
        if len(self.ensemble_stoppers) == 1:
            result[f"ensemble_best_model_{self.ensemble_size}"] = self.ensemble_stoppers[0].best_model
        else:
            for i, stopper in enumerate(self.ensemble_stoppers):
                result[f"ensemble_best_model_{i + 1}"] = stopper.best_model
        return result

    def get_best_epochs(self) -> dict:
        result = {}
        if self.individual_model_stoppers:
            result["individual_best_epochs"] = [stopper.best_epoch for stopper in self.individual_model_stoppers]

        result["shared_epoch_best_epoch"] = self.shared_epoch_stopper.best_epoch
        if len(self.ensemble_stoppers) == 1:
            result[f"ensemble_best_epoch_{self.ensemble_size}"] = self.ensemble_stoppers[0].best_epoch
        else:
            for i, stopper in enumerate(self.ensemble_stoppers):
                result[f"ensemble_best_epoch_{i + 1}"] = stopper.best_epoch
        return result


# --- Logic for performing temperature scaling
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
            x, y = batch
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
            holdout_masks.append(holdout_mask.T)  # To batch (ensemble_size, batch_size)

    return (
        torch.cat(logits, dim=1),
        torch.cat(targets, dim=1),
        torch.cat(holdout_masks, dim=1) if holdout_masks else None,
    )


def find_temperature_for_individual_models(
    logits: Tensor,  # shape: [ensemble_size, num_samples, num_classes]
    targets: Tensor,  # shape: [ensemble_size, num_samples]
    holdout_mask: Tensor | None,  # shape: [ensemble_size, num_samples] or None
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
            # Using the mask for getting the holdout samples for the respective models
            loss = criterion(logits_scaled[holdout_mask], targets[holdout_mask])
        loss.backward()
        return loss

    optimizer.step(closure)
    return temperatures


def find_temperature_for_ensemble(
    logits: Tensor,  # shape: [ensemble_size, num_samples, num_classes]
    targets: Tensor,  # shape: [ensemble_size, num_samples]
    holdout_mask: Tensor | None,  # shape: [ensemble_size, num_samples] or None
    share_temperature: bool = False,
):
    ensemble_size = logits.size(0)
    num_temperature_params = 1 if share_temperature else ensemble_size
    temperature = nn.Parameter(torch.ones(num_temperature_params, dtype=logits.dtype, device=logits.device))
    optimizer = torch.optim.LBFGS([temperature], lr=0.1, max_iter=100)  # type: ignore

    if holdout_mask is not None:  # Match logit dims for element-wise multiplication
        holdout_mask = holdout_mask.unsqueeze(-1)  # shape: [ensemble_size, num_samples, 1]

    if not (targets == targets[0]).all():  # Ensure that all models have the same targets
        raise ValueError("Targets for ensemble models must be the same")
    else:
        targets = targets[0]

    criterion = nn.NLLLoss()

    def closure():
        optimizer.zero_grad()
        logits_scaled = logits / temperature.view(-1, 1, 1)
        model_log_probs = logits_scaled.log_softmax(dim=-1)
        if holdout_mask is None:
            ensemble_log_probs = model_log_probs.logsumexp(dim=0) - math.log(ensemble_size)
        else:
            model_log_probs_masked = model_log_probs.masked_fill(
                ~holdout_mask, float("-inf")
            )  # shape: [ensemble_size, num_samples, num_classes]
            num_models = holdout_mask.sum(dim=0)  # Count the number of models in the ensemble
            # For each sample, we essentially only include the log probabilities of the models that are not masked out
            ensemble_log_probs = torch.logsumexp(model_log_probs_masked, dim=0) - num_models.log()
            # This results in the shape of ensemble_log_probs being [num_samples, num_classes]
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

    # use unscaled model_log_probs instead of the logits directly for this approach
    model_log_probs_unscaled = logits.log_softmax(dim=-1)  # shape: [ensemble_size, num_samples, num_classes]

    def closure():
        optimizer.zero_grad()
        if holdout_mask is None:
            ensemble_log_probs_unscaled = model_log_probs_unscaled.logsumexp(dim=0) - math.log(ensemble_size)
        else:
            model_log_probs_unscaled_masked = model_log_probs_unscaled.masked_fill(
                ~holdout_mask, float("-inf")
            )  # shape: [ensemble_size, num_samples, num_classes]
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
