import math
from functools import partial

import torch
import torch_geometric as pyg
import torchmetrics
from torch import Tensor, nn
from transformers import AutoTokenizer

import ensemble_optimality_gap.models.batchensemble as be
from ensemble_optimality_gap.metrics import (
    Disagreement,
    EnsembleDiversity,
    GeneralizedEntropy,
    KLDivergence,
    NegativeLogLikelihood,
)
from ensemble_optimality_gap.models.gnn import GCNNet
from ensemble_optimality_gap.models.lstm import BiLSTM
from ensemble_optimality_gap.models.wide_resnet import WideResNet


class GraphEnsemble(nn.Module):
    def __init__(self, models: list[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, data):
        batch_size = data.num_graphs
        batch_size_per_model = batch_size // len(self.models)

        if batch_size % len(self.models) != 0:
            raise ValueError(f"Batch size {batch_size} must be divisible by number of models {len(self.models)}")

        model_outputs = []
        for i, model in enumerate(self.models):
            data_i = data[i * batch_size_per_model : (i + 1) * batch_size_per_model]
            data_i = pyg.data.Batch.from_data_list(data_i)
            model_outputs.append(model(data_i.x, data_i.edge_index, batch=data_i.batch))
        return torch.cat(model_outputs, dim=0)


class Ensemble(nn.Module):
    def __init__(self, models: list[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        batch_size = x.size(0)
        batch_size_per_model = batch_size // len(self.models)

        if batch_size % len(self.models) != 0:
            raise ValueError(f"Batch size {batch_size} must be divisible by number of models {len(self.models)}")

        x_split = x.split(batch_size_per_model, dim=0)
        if attention_mask is None:
            outputs = [model(x_split[i]) for i, model in enumerate(self.models)]
        else:
            attention_mask_split = attention_mask.split(batch_size_per_model, dim=0)
            outputs = [model(x_split[i], attention_mask=attention_mask_split[i]) for i, model in enumerate(self.models)]
        return torch.cat(outputs, dim=0)


class EnsembleMetrics:
    def __init__(
        self,
        num_models: int,
        compute_individual_metrics: bool = True,
        compute_joint_metrics: bool = True,
        use_masked_updates: bool = False,
        num_classes: int = 10,
        device: str | torch.device = "cpu",
    ):
        if not compute_individual_metrics and not compute_joint_metrics:
            raise ValueError("At least one of compute_individual_metrics or compute_joint_metrics must be True")

        self.num_models = num_models
        self.compute_individual_metrics = compute_individual_metrics
        self.compute_joint_metrics = compute_joint_metrics
        self.use_masked_updates = use_masked_updates
        self.num_classes = num_classes

        if compute_individual_metrics:
            self.standard_individual_metrics = [self._init_standard_metrics() for _ in range(num_models)]
        else:
            self.standard_individual_metrics = None
        if compute_joint_metrics:
            if use_masked_updates:
                # We only get the average overlapping joint performance
                self.masked_joint_metrics = self._init_standard_metrics()
            else:
                # We track the performance of each ensemble size 1 to num_models
                self.standard_joint_metrics = [self._init_standard_metrics() for _ in range(num_models)]
                self.ensemble_diversity = [EnsembleDiversity() for _ in range(num_models)]
                self.pairwise_metrics = [self._init_ensemble_metrics() for _ in range(num_models - 1)]
                self.model_vs_ensemble_metrics = [self._init_ensemble_metrics() for _ in range(num_models)]

        self.to(device)

    def _init_standard_metrics(self) -> torchmetrics.MetricCollection:
        return torchmetrics.MetricCollection(
            {
                "nll": NegativeLogLikelihood(),
                "accuracy": torchmetrics.Accuracy("multiclass", num_classes=self.num_classes),
                "ece": torchmetrics.CalibrationError("multiclass", num_classes=self.num_classes),
                "entropy": GeneralizedEntropy(),
                "f1": torchmetrics.F1Score("multiclass", num_classes=self.num_classes),
                "precision": torchmetrics.Precision("multiclass", num_classes=self.num_classes),
                "recall": torchmetrics.Recall("multiclass", num_classes=self.num_classes),
            }
        )

    def _init_ensemble_metrics(self) -> torchmetrics.MetricCollection:
        return torchmetrics.MetricCollection(
            {
                "disagreement": Disagreement(),
                "kl_div": KLDivergence(),
                "cross_entropy": GeneralizedEntropy(),
            }
        )

    def _reshape_inputs(self, logits: Tensor, targets: Tensor, mask: Tensor | None = None):
        """Reshape the batched inputs for the ensemble models"""
        ensemble_batch_size, *other_dims = logits.shape
        batch_size = ensemble_batch_size // self.num_models
        logits = logits.view(self.num_models, batch_size, *other_dims)
        targets = targets.view(self.num_models, batch_size)
        if mask is not None:
            mask = mask.T
        return logits, targets, mask

    @torch.no_grad()
    def update(
        self,
        model_logits: Tensor,  # shape: (ensemble_size * batch_size, num_classes)
        targets: Tensor,  # shape: (ensemble_size * batch_size)
        mask: Tensor | None = None,  # shape: (batch_size, ensemble_size)
        temperature: Tensor | None = None,  # shape: per model (ensemble_size) or shared (1)
        temperature_target: str = "model_logits",  # "model_logits" or "ensemble_log_probs"
    ):
        if self.use_masked_updates:
            assert mask is not None, "Mask must be provided for masked updates"
        else:
            assert mask is None, "Mask should not be provided unless using masked updates"

        if temperature_target not in ["model_logits", "ensemble_log_probs"]:
            raise ValueError(
                f"temperature_target must be either 'model_logits' or 'ensemble_log_probs', got {temperature_target}"
            )

        # if temperature target is ensemble_log_probs, we can only have a single temperature value
        if temperature_target == "ensemble_log_probs" and temperature.numel() != 1:
            raise ValueError("When temperature_target is 'ensemble_log_probs', temperature must be a single value")

        model_logits, targets, mask = self._reshape_inputs(model_logits, targets, mask)

        if temperature_target == "model_logits" and temperature is not None:
            # Apply temperature scaling to model logits (temperature is either [ensemble_size] or [1])
            model_logits = model_logits / temperature.view(-1, 1, 1)

        model_probs, model_log_probs = self._compute_probs_and_log_probs(model_logits)

        # --- Individual metrics ---
        if self.compute_individual_metrics:
            for i, (probs, log_probs, target) in enumerate(zip(model_probs, model_log_probs, targets)):
                if mask is not None:
                    mask_i = mask[i]
                    if mask_i.any():
                        probs = probs[mask_i]
                        log_probs = log_probs[mask_i]
                        target = target[mask_i]
                    else:
                        continue
                self._update_standard_metrics(self.standard_individual_metrics[i], probs, log_probs, target)

        # --- Joint metrics ---
        if self.compute_joint_metrics:
            assert (targets == targets[0]).all(), "Targets must be the same for all models in the ensemble"
            target = targets[0]

            if self.use_masked_updates:
                # The mask has already been applied to the probs and log_probs in _compute_probs_and_log_probs
                ensemble_probs, ensemble_log_probs = self._compute_masked_ensemble_probs_and_log_probs(
                    model_probs, model_log_probs, mask
                )
                if temperature is not None and temperature_target == "ensemble_log_probs":
                    ensemble_log_probs = ensemble_log_probs / temperature.item()
                    ensemble_probs = ensemble_log_probs.softmax(dim=-1)
                    ensemble_log_probs = torch.log(ensemble_probs.clamp(min=1e-9))

                self._update_standard_metrics(self.masked_joint_metrics, ensemble_probs, ensemble_log_probs, target)

            else:
                for i in range(self.num_models):
                    ensemble_probs, ensemble_log_probs = self._compute_ensemble_probs_and_log_probs(
                        model_probs[: i + 1], model_log_probs[: i + 1]
                    )

                    if temperature is not None and temperature_target == "ensemble_log_probs":
                        ensemble_log_probs = ensemble_log_probs / temperature.item()
                        ensemble_probs = ensemble_log_probs.softmax(dim=-1)
                        ensemble_log_probs = torch.log(ensemble_probs.clamp(min=1e-9))

                    self._update_standard_metrics(
                        self.standard_joint_metrics[i], ensemble_probs, ensemble_log_probs, target
                    )
                    if i > 0:
                        self._update_pairwise_metrics(
                            self.pairwise_metrics[i - 1], model_probs[: i + 1], model_log_probs[: i + 1]
                        )

                    self.ensemble_diversity[i].update(
                        ensemble_probs, ensemble_log_probs, model_probs[: i + 1], model_log_probs[: i + 1]
                    )

                    self._update_model_vs_ensemble_metrics(
                        self.model_vs_ensemble_metrics[i],
                        ensemble_probs,
                        ensemble_log_probs,
                        model_probs[: i + 1],
                        model_log_probs[: i + 1],
                    )

    def _update_standard_metrics(
        self, metrics: torchmetrics.MetricCollection, probs: Tensor, log_probs: Tensor, targets: Tensor
    ):
        metrics["nll"].update(log_probs, targets)
        metrics["accuracy"].update(probs, targets)
        metrics["ece"].update(probs, targets)
        metrics["f1"].update(probs, targets)
        metrics["precision"].update(probs, targets)
        metrics["recall"].update(probs, targets)
        metrics["entropy"].update(probs, log_probs)

    def _update_pairwise_metrics(self, metrics: torchmetrics.MetricCollection, probs: Tensor, log_probs: Tensor):
        """
        Update the pairwise metrics for the ensemble

        Args:
            metrics: MetricCollection containing the pairwise metrics
            probs: Probabilities from the ensemble [ensemble_size, batch_size, num_classes]
            log_probs: Log probabilities from the ensemble [ensemble_size, batch_size, num_classes]
        """
        # Efficiently compute the pairwise metrics using the upper triangular part of the matrix
        indices = torch.triu_indices(probs.size(0), probs.size(0), offset=1)
        probs_i, probs_j = probs[indices[0]], probs[indices[1]]
        log_probs_i, log_probs_j = log_probs[indices[0]], log_probs[indices[1]]

        # Update the pairwise metrics (disagreement is symmetric, so we only need to update one side)
        metrics["disagreement"].update(probs_i, probs_j)
        metrics["kl_div"].update(p=probs_i, log_p=log_probs_i, log_q=log_probs_j)
        metrics["kl_div"].update(p=probs_j, log_p=log_probs_j, log_q=log_probs_i)
        metrics["cross_entropy"].update(p=probs_i, log_q=log_probs_j)
        metrics["cross_entropy"].update(p=probs_j, log_q=log_probs_i)

    def _update_model_vs_ensemble_metrics(
        self,
        metrics: torchmetrics.MetricCollection,
        model_probs: Tensor,  # shape: (ensemble_size, batch_size, num_classes)
        model_log_probs: Tensor,  # shape: (ensemble_size, batch_size, num_classes)
        ensemble_probs: Tensor,  # shape: (batch_size, num_classes)
        ensemble_log_probs: Tensor,  # shape: (batch_size, num_classes)
    ):
        for probs_i, log_probs_i in zip(model_probs, model_log_probs):
            metrics["disagreement"].update(probs_i, ensemble_probs)
            metrics["kl_div"].update(p=ensemble_probs, log_p=ensemble_log_probs, log_q=log_probs_i)
            metrics["cross_entropy"].update(p=ensemble_probs, log_q=log_probs_i)

    def _compute_probs_and_log_probs(self, model_logits: Tensor) -> tuple[Tensor, Tensor]:
        return model_logits.softmax(dim=-1), model_logits.log_softmax(dim=-1)

    def _compute_ensemble_probs_and_log_probs(
        self, model_probs: Tensor, model_log_probs: Tensor
    ) -> tuple[Tensor, Tensor]:
        num_models = model_probs.size(0)
        return model_probs.mean(dim=0), torch.logsumexp(model_log_probs, dim=0) - math.log(num_models)

    def _compute_masked_ensemble_probs_and_log_probs(
        self, model_probs: Tensor, model_log_probs: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        mask = mask.unsqueeze(-1)  # shape: (ensemble_size, batch_size, 1)
        ensemble_probs = (model_probs * mask).sum(dim=0) / mask.sum(dim=0)
        model_log_probs_masked = model_log_probs.masked_fill(~mask, float("-inf"))
        num_models = mask.sum(dim=0)
        ensemble_log_probs = torch.logsumexp(model_log_probs_masked, dim=0) - num_models.log()
        # There should be no nan or inf
        assert not torch.isnan(ensemble_probs).any() and not torch.isinf(ensemble_probs).any()
        assert not torch.isnan(ensemble_log_probs).any() and not torch.isinf(ensemble_log_probs).any()
        return ensemble_probs, ensemble_log_probs

    def compute(self) -> dict:
        """Compute the metrics and return them as a dictionary"""
        metrics = {}

        # --- Individual metrics ---
        if self.compute_individual_metrics:
            for i, metric_collection in enumerate(self.standard_individual_metrics):
                for key, value in metric_collection.compute().items():
                    metrics[f"model_{i}/{key}"] = value

        if self.compute_joint_metrics:
            # --- Standard Joint metrics ---
            if self.use_masked_updates:
                for key, value in self.masked_joint_metrics.compute().items():
                    metrics[f"ensemble_{self.num_models}/{key}"] = value
            else:
                for i, metric_collection in enumerate(self.standard_joint_metrics):
                    for key, value in metric_collection.compute().items():
                        metrics[f"ensemble_{i + 1}/{key}"] = value

                    # --- Ensemble-specific metrics ---
                    if i > 0:
                        # Pairwise metrics
                        for key, value in self.pairwise_metrics[i - 1].compute().items():
                            metrics[f"ensemble_{i + 1}/average_pairwise_{key}"] = value
                    # Ensemble diversity
                    metrics[f"ensemble_{i + 1}/ensemble_diversity"] = self.ensemble_diversity[i].compute()
                    # Model vs ensemble metrics
                    for key, value in self.model_vs_ensemble_metrics[i].compute().items():
                        metrics[f"ensemble_{i + 1}/model_vs_ensemble_{key}"] = value

        return metrics

    def reset(self):
        for metric in self.all_metrics():
            metric.reset()

    def to(self, device):
        for metric in self.all_metrics():
            metric.to(device)

    def all_metrics(self):
        metrics = []
        if self.compute_individual_metrics:
            metrics += self.standard_individual_metrics
        if self.compute_joint_metrics:
            if self.use_masked_updates:
                metrics.append(self.masked_joint_metrics)
            else:
                metrics += self.standard_joint_metrics + self.pairwise_metrics + self.model_vs_ensemble_metrics
                metrics += self.ensemble_diversity
        return metrics


def load_ensemble_model(
    model_name: str,
    dataset_name: str,
    ensemble_size: int,
    num_classes: int,
    num_features: int,
    be_batchnorm: str = "ensemble",
    gnn_kwargs: dict | None = None,
    batchensemble_fast_weight_init: float = 0.5,
) -> nn.Module:
    if model_name in ["wrn-16-4", "wrn-16-4-be"]:
        model_params = {
            "in_channels": num_features,
            "out_features": num_classes,
            "depth": 16,
            "width_multiplier": 4,
        }
        if "be" in model_name:
            if be_batchnorm == "normal":
                model = WideResNet(
                    **model_params,
                    linear_layer=partial(
                        be.Linear,
                        ensemble_size=ensemble_size,
                        alpha_init=batchensemble_fast_weight_init,
                        gamma_init=batchensemble_fast_weight_init,
                    ),
                    conv_layer=partial(
                        be.Conv2d,
                        ensemble_size=ensemble_size,
                        alpha_init=batchensemble_fast_weight_init,
                        gamma_init=batchensemble_fast_weight_init,
                    ),
                )
            else:
                model = WideResNet(
                    **model_params,
                    linear_layer=partial(
                        be.Linear,
                        ensemble_size=ensemble_size,
                        alpha_init=batchensemble_fast_weight_init,
                        gamma_init=batchensemble_fast_weight_init,
                    ),
                    conv_layer=partial(
                        be.Conv2d,
                        ensemble_size=ensemble_size,
                        alpha_init=batchensemble_fast_weight_init,
                        gamma_init=batchensemble_fast_weight_init,
                    ),
                    norm_layer=partial(be.Ensemble_BatchNorm2d, ensemble_size=ensemble_size),
                    norm_initializer=be.ensemble_bn_init,
                )
        else:
            model = Ensemble([WideResNet(**model_params) for _ in range(ensemble_size)])
    elif model_name in ["GCN"]:
        if gnn_kwargs is None:
            gnn_kwargs = {}
        hidden_channels = gnn_kwargs.get("hidden_channels", 64)
        num_layers = gnn_kwargs.get("num_layers", 4)

        model = GraphEnsemble(
            [
                GCNNet(
                    num_features=num_features,
                    num_classes=num_classes,
                    hidden=hidden_channels,
                    num_conv_layers=num_layers,
                )
                for _ in range(ensemble_size)
            ]
        )
    elif model_name == "mlp":
        hidden_size = 1024
        bias = False
        model = Ensemble(
            [
                torch.compile(
                    nn.Sequential(
                        # Input layer
                        nn.Linear(num_features, hidden_size, bias=bias),
                        nn.BatchNorm1d(hidden_size),
                        nn.ReLU(),
                        # Hidden layers
                        nn.Linear(hidden_size, hidden_size, bias=bias),  # size: 1024 -> 1024
                        nn.BatchNorm1d(hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size, bias=bias),  # size: 1024 -> 1024
                        nn.BatchNorm1d(hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size, bias=bias),  # size: 1024 -> 1024
                        nn.BatchNorm1d(hidden_size),
                        nn.ReLU(),
                        # Output layer
                        nn.Linear(hidden_size, num_classes),  # size: 1024 -> num_classes
                    )
                )
                for _ in range(ensemble_size)
            ]
        )
    elif model_name == "bilstm":
        tokenizer = AutoTokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces=True)
        model = Ensemble(
            [
                BiLSTM(
                    vocab_size=tokenizer.vocab_size,
                    embedding_dim=128,
                    hidden_dim=256,
                    output_dim=num_classes,
                    pad_idx=tokenizer.eos_token_id,
                    num_layers=1,
                )
                for _ in range(ensemble_size)
            ]
        )

    else:
        raise ValueError(f"Model {model_name} not supported")

    return model
