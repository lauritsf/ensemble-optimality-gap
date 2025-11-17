import torch
from torch import Tensor
from torchmetrics import Metric


class NegativeLogLikelihood(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.value: Tensor
        self.total: Tensor
        self.add_state("value", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, log_probs: Tensor, target: Tensor):
        """
        Args:
            log_probs: log probabilities
            target: class index
        """
        # Assert that preds are log probabilities
        assert log_probs.max() <= 0
        assert log_probs.size(0) == target.size(0), f"{log_probs.size()} != {target.size()}"
        self.value += torch.nn.functional.nll_loss(log_probs, target, reduction="sum")
        self.total += target.shape[0]

    def compute(self):
        return self.value / self.total


class Disagreement(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.disagreements: Tensor
        self.total: Tensor
        self.add_state("disagreements", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds1: Tensor, preds2: Tensor):
        """
        Args:
            preds1: predictions from the first model
            preds2: predictions from the second model
        """
        self.disagreements += torch.sum(preds1.argmax(dim=-1) != preds2.argmax(dim=-1))
        self.total += preds1.shape[0]

    def compute(self):
        return self.disagreements.float() / self.total


class GeneralizedEntropy(Metric):
    """Calculates entropy or cross-entropy based on inputs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.value: Tensor
        self.total: Tensor
        self.add_state("value", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, p: Tensor, log_q: Tensor | None = None):
        """
        Args:
            p: probabilities of the first/true (or only) distribution
            log_q: log probabilities of the second/predicted distribution (optional)
        """
        if log_q is None:
            log_q = torch.log(p + 1e-12)
        self.value += -torch.sum(p * log_q)
        self.total += p.shape[0]

    def compute(self):
        return self.value / self.total


class KLDivergence(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.value: Tensor
        self.total: Tensor
        self.add_state("value", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, p: Tensor, log_q: Tensor, log_p: Tensor | None = None):
        """
        Args:
            p: probabilities of the first distribution
            log_q: log probabilities of the second distribution
        """
        if log_p is None:
            log_p = torch.log(p + 1e-12)
        self.value += torch.sum(p * (log_p - log_q))
        self.total += p.shape[0]

    def compute(self):
        return self.value / self.total


class EnsembleDiversity(Metric):
    """
    Calculates the diversity of an ensemble using the expected difference
    between the entropy of the average distributions and the average entropy
    of individual distributions: E[H(p_bar) - E_p[H(p)]]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.value: Tensor
        self.total: Tensor
        self.add_state("value", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        ensemble_probs: Tensor,
        ensemble_log_probs: Tensor,
        individual_probs: Tensor,
        individual_log_probs: Tensor,
    ):
        """
        Args:
            ensemble_probs: Average probabilities from the ensemble [batch_size, num_classes]
            ensemble_log_probs: Log probabilities of average probabilities [batch_size, num_classes]
            individual_probs: Probabilities from each individual model [ensemble_size, batch_size, num_classes]
            individual_log_probs: Log probabilities from each individual model [ensemble_size, batch_size, num_classes]
        """
        ensemble_entropy = -torch.sum(ensemble_probs * ensemble_log_probs, dim=-1)  # [batch_size]
        individual_entropy = -torch.mean(
            torch.sum(individual_probs * individual_log_probs, dim=-1), dim=0
        )  # [batch_size]

        self.value += torch.sum(ensemble_entropy - individual_entropy)  # Sum over batch
        self.total += ensemble_probs.shape[0]  # batch_size

    def compute(self):
        return self.value / self.total
