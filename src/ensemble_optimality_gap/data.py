import gzip
import os
import shutil
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
import torch_geometric as pyg
import torchvision.transforms.v2 as transforms
import wget
from datasets import load_dataset as hf_load_dataset
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from transformers import AutoTokenizer

NUM_CLASSES = {
    "cifar10": 10,
    "cifar10_subset": 10,
    "NCI1": 2,
    "covertype": 7,
    "ag_news": 4,
}

NUM_FEATURES = {
    "NCI1": 37,
    "cifar10": 3,
    "cifar10_subset": 3,
    "covertype": 54,
    "ag_news": None,
}


# load either cifar10 or cifar100 dataset
def load_cifar(dataset_name: str, subset_seed: int = 0) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    base_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )
    if dataset_name in ["cifar10", "cifar10_subset"]:
        train_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=base_transform)
        test_dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=base_transform)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    x_train, y_train = next(iter(DataLoader(train_dataset, batch_size=len(train_dataset))))
    x_test, y_test = next(iter(DataLoader(test_dataset, batch_size=len(test_dataset))))
    if dataset_name == "cifar10_subset":
        # Use only 5% of the data
        x_train, _, y_train, _ = train_test_split(
            x_train, y_train, train_size=0.05, stratify=y_train, random_state=subset_seed
        )
    return x_train, y_train, x_test, y_test  # type: ignore


def create_cifar_transforms(x_train: Tensor):
    """Create transforms for CIFAR dataset based on the mean and std of the training data."""
    mean = x_train.mean(dim=[0, 2, 3]).tolist()
    std = x_train.std(dim=[0, 2, 3]).tolist()
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_transform = transforms.Compose([transforms.Normalize(mean, std)])
    return train_transform, eval_transform


def create_cifar_loaders(
    dataset_name: str,
    holdout_strategy: str,
    batch_strategy: str,
    holdout_fraction: float,
    batch_size: int,
    seed: int,
    ensemble_size: int,
    device: str | torch.device,
    drop_last: bool = False,
):
    """
    Create data loaders for the CIFAR dataset.

    Args:
        dataset_name (str): Name of the dataset to load. Should be either "cifar10" or "cifar100".
        holdout_strategy (str): Strategy to use for creating holdout splits. Should be one of "same", "random",
                                "disjoint", or "overlapping".
        holdout_fraction (float): Fraction of the data to use for validation. If 0, no validation set is created.
        batch_strategy (str): Whether each model in the ensemble should get the same batch or different batches.
        batch_size (int): Batch size for the data loaders.
        seed (int): Seed for the random number generator.
        ensemble_size (int): Number of models in the ensemble.
        device (str | torch.device): Device to use for the data loaders.
        drop_last (bool): Whether to drop the last incomplete batch.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Training, validation, and test data loaders.
    """
    x_train, y_train, x_test, y_test = load_cifar(dataset_name)
    if holdout_fraction > 0:
        train_indices, val_indices = create_holdout_splits(
            holdout_strategy, y_train, holdout_fraction, seed, ensemble_size
        )
    else:
        # Use the entire training set for training and no validation set
        train_indices = torch.arange(len(y_train)).unsqueeze(0)  # shape: (1, num_samples)
        val_indices = torch.tensor([], dtype=torch.long).unsqueeze(0)  # shape: (1, 0)

    if batch_strategy == "same":
        # Use the same data loader for all models in the ensemble
        train_indices = train_indices[0].unsqueeze(0)
        val_indices = val_indices[0].unsqueeze(0)
        train_transform, eval_transform = create_cifar_transforms(x_train[train_indices[0]])
        train_transforms = [train_transform]
        eval_transforms = [eval_transform]
    else:  # The training indices are different, so the transforms should be different
        train_transforms = []
        eval_transforms = []
        for train_indices_i in train_indices:
            tr_transform, ev_transform = create_cifar_transforms(x_train[train_indices_i])
            train_transforms.append(tr_transform)
            eval_transforms.append(ev_transform)

    # Send the data to the device
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    train_indices = train_indices.to(device)
    val_indices = val_indices.to(device)

    train_datasets = [
        TensorDataset(x_train[train_indices_i], y_train[train_indices_i]) for train_indices_i in train_indices
    ]
    train_loader = EnsembleDataLoader(
        [
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
            for train_dataset in train_datasets
        ],
        train_transforms,
        ensemble_size,
    )
    test_loader = EnsembleDataLoader(
        # Use the same data loader for all models in the ensemble when testing
        [DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)],
        eval_transforms,
        ensemble_size,
    )
    if holdout_fraction == 0:
        val_loader = None
    elif holdout_strategy == "overlapping":
        num_samples = len(y_train)
        mask = torch.zeros((num_samples, ensemble_size), dtype=torch.bool, device=device)
        mask.scatter_(0, val_indices.T, True)
        x, y = x_train[mask.any(dim=1)], y_train[mask.any(dim=1)]
        mask = mask[mask.any(dim=1)]
        val_loader = EnsembleDataLoader(
            [DataLoader(TensorDataset(x, y, mask), batch_size=batch_size, shuffle=False)],
            eval_transforms,
            ensemble_size,
        )
    else:
        val_datasets = [TensorDataset(x_train[val_indices_i], y_train[val_indices_i]) for val_indices_i in val_indices]
        val_loader = EnsembleDataLoader(
            [DataLoader(val_dataset, batch_size=batch_size, shuffle=False) for val_dataset in val_datasets],
            eval_transforms,
            ensemble_size,
        )

    return train_loader, val_loader, test_loader


def create_covertype_loaders(
    holdout_strategy: str,
    batch_strategy: str,
    holdout_fraction: float,
    batch_size: int,
    seed: int,
    ensemble_size: int,
    device: str | torch.device,
    drop_last: bool = False,
):
    dataset_name_file = "forest-cover-type"
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    data_path = Path("./data") / "covertype"
    tmp_out = data_path / f"{dataset_name_file}.gz"
    csv_out = data_path / f"{dataset_name_file}.csv"

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    if not csv_out.exists():
        print(f"Downloading {dataset_name_file} data...")
        if tmp_out.exists():
            os.remove(tmp_out)
        wget.download(url, tmp_out.as_posix())
        print("Download complete. Extracting...")
        with gzip.open(tmp_out, "rb") as f_in:
            with open(csv_out, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        print("Extraction complete.")
        os.remove(tmp_out)
    else:
        print(f"{csv_out} already exists.")

    target_col_name = "Covertype"
    wilderness_cols = [f"Wilderness_Area{i}" for i in range(1, 5)]  # Wilderness_Area1 to Wilderness_Area4
    soil_type_cols = [f"Soil_Type{i}" for i in range(1, 41)]  # Soil_Type1 to Soil_Type40
    bool_columns = wilderness_cols + soil_type_cols
    int_columns = [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
    ]
    column_names = int_columns + bool_columns + [target_col_name]

    df = pd.read_csv(csv_out, header=None, names=column_names)
    df[target_col_name] = df[target_col_name] - 1  # Map 1-7 to 0-6

    # Convert to tensor
    x_all = torch.tensor(df[int_columns + bool_columns].values).float()
    y_all = torch.tensor(df[target_col_name].values).long()

    # Peform consistent train/test split of 0.8/0.2
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=0, stratify=y_all)

    if holdout_fraction > 0:
        train_indices, val_indices = create_holdout_splits(
            holdout_strategy, y_train, holdout_fraction, seed, ensemble_size
        )
    else:
        train_indices = torch.arange(len(y_train)).unsqueeze(0)  # shape: (1, num_samples)
        val_indices = torch.tensor([], dtype=torch.long).unsqueeze(0)  # shape: (1, 0)

    # Send the data to device
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    train_indices = train_indices.to(device)
    val_indices = val_indices.to(device)

    if batch_strategy == "same":
        # Use the same data_loader for all models in the ensemble
        train_indices = train_indices[0].unsqueeze(0)
        val_indices = val_indices[0].unsqueeze(0)
        x_mean = x_train[train_indices[0]].mean(dim=0)
        x_std = x_train[train_indices[0]].std(dim=0)
        input_normalization = [lambda x: (x - x_mean) / (x_std + 1e-8)]

    else:
        x_means = [x_train[train_indices_i].mean(dim=0) for train_indices_i in train_indices]
        x_stds = [x_train[train_indices_i].std(dim=0) for train_indices_i in train_indices]
        input_normalization = [lambda x: (x - x_mean) / (x_std + 1e-8) for x_mean, x_std in zip(x_means, x_stds)]

    train_datasets = [
        TensorDataset(x_train[train_indices_i], y_train[train_indices_i]) for train_indices_i in train_indices
    ]
    train_loader = EnsembleDataLoader(
        [
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
            for train_dataset in train_datasets
        ],
        input_normalization,
        ensemble_size,
    )
    test_loader = EnsembleDataLoader(
        # Use the same data loader for all models in the ensemble when testing
        [DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)],
        input_normalization,
        ensemble_size,
    )

    if holdout_fraction == 0:
        val_loader = None
    elif holdout_strategy == "overlapping":
        num_samples = len(y_train)
        mask = torch.zeros((num_samples, ensemble_size), dtype=torch.bool, device=device)
        mask.scatter_(0, val_indices.T, True)
        x, y = x_train[mask.any(dim=1)], y_train[mask.any(dim=1)]
        mask = mask[mask.any(dim=1)]
        val_loader = EnsembleDataLoader(
            [DataLoader(TensorDataset(x, y, mask), batch_size=batch_size, shuffle=False)],
            input_normalization,
            ensemble_size,
        )
    else:
        val_datasets = [TensorDataset(x_train[val_indices_i], y_train[val_indices_i]) for val_indices_i in val_indices]
        val_loader = EnsembleDataLoader(
            [DataLoader(val_dataset, batch_size=batch_size, shuffle=False) for val_dataset in val_datasets],
            input_normalization,
            ensemble_size,
        )

    return train_loader, val_loader, test_loader


def create_tud_loaders(
    dataset_name: str,
    holdout_strategy: str,
    batch_strategy: str,
    holdout_fraction: float,
    batch_size: int,
    seed: int,
    ensemble_size: int,
    device: str | torch.device,
    test_split_seed: int = 0,  # We keep this fixed, as we focus on the training/validation split
    drop_last: bool = False,
):
    if dataset_name not in ["NCI1"]:
        raise ValueError(f"Dataset {dataset_name} not supported")

    dataset = pyg.datasets.TUDataset(root="data", name=dataset_name)  # type: ignore
    train_indices, test_indices = train_test_split(
        torch.arange(len(dataset)), test_size=0.2, random_state=test_split_seed, stratify=dataset.y
    )
    train_dataset = dataset.copy(train_indices).to(device)
    test_dataset = dataset.copy(test_indices).to(device)

    if holdout_fraction > 0:
        train_indices, val_indices = create_holdout_splits(
            holdout_strategy, train_dataset.y.cpu(), holdout_fraction, seed, ensemble_size
        )
        if batch_strategy == "same":
            # Use the same data loader for all models in the ensemble
            train_indices = train_indices[0].unsqueeze(0)
            val_indices = val_indices[0].unsqueeze(0)

        # Send the data to the device
        train_indices, val_indices = train_indices.to(device), val_indices.to(device)
        train_datasets = [train_dataset[train_indices_i] for train_indices_i in train_indices]

    else:
        train_datasets = [train_dataset]

    train_loader = EnsembleGraphDataLoader(
        [
            pyg.loader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)  # type: ignore
            for train_dataset in train_datasets
        ],
        ensemble_size,
    )
    test_loader = EnsembleGraphDataLoader(
        [pyg.loader.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)],  # type: ignore
        ensemble_size,
    )
    if holdout_fraction == 0:
        # Early exit, since we don't need a validation set
        val_loader = None
        return train_loader, val_loader, test_loader

    if holdout_strategy == "overlapping":
        num_samples = len(train_dataset)
        mask = torch.zeros((num_samples, ensemble_size), dtype=torch.bool, device=device)
        mask.scatter_(0, val_indices.T, True)
        val_dataset = train_dataset.copy(mask.any(dim=1))
        mask = mask[mask.any(dim=1)]
        val_dataset._data.mask = mask  # set the mask attribute to the mask tensor
        val_dataset.slices["mask"] = train_dataset.slices["y"]  # update the slices
        val_dataset._data_list = None  # reset the cache
        val_loader = EnsembleGraphDataLoader(
            [pyg.loader.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)],  # type: ignore
            ensemble_size,
        )
    else:
        val_datasets = [train_dataset[val_indices_i] for val_indices_i in val_indices]
        val_loader = EnsembleGraphDataLoader(
            [pyg.loader.DataLoader(val_dataset, batch_size=batch_size, shuffle=False) for val_dataset in val_datasets],  # type: ignore
            ensemble_size,
        )

    return train_loader, val_loader, test_loader


def create_setfit_loaders(
    holdout_strategy: str,
    batch_strategy: str,
    holdout_fraction: float,
    batch_size: int,
    seed: int,
    ensemble_size: int,
    device: str | torch.device,
    drop_last: bool = False,
    dataset_name: str = "bbcnews",
    tokenizer: str = "distilbert-base-uncased",
):
    hf_dataset_name = {
        "ag_news": "SetFit/ag_news",
    }.get(dataset_name, None)
    if hf_dataset_name is None:
        raise ValueError(f"Dataset {dataset_name} not supported. Supported datasets: {list(hf_dataset_name.keys())}")
    max_length = {
        "ag_news": 360,
    }.get(dataset_name, None)
    if max_length is None:
        raise ValueError(f"Dataset {dataset_name} does not have a predefined max_length. Please specify it manually.")

    raw_datasets = hf_load_dataset(hf_dataset_name)
    if tokenizer == "gpt2":
        tokenizer = AutoTokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces=True)
        tokenizer.pad_token = tokenizer.eos_token  # type: ignore

        def tokenize_data(text_list: list[str]):
            tokenized = tokenizer(
                text_list, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt"
            )  # type: ignore
            return tokenized["input_ids"], tokenized["attention_mask"]

        x_train_val, attention_mask_train_val = tokenize_data(raw_datasets["train"]["text"])  # type: ignore
        x_test, attention_mask_test = tokenize_data(raw_datasets["test"]["text"])  # type: ignore
    else:
        raise ValueError(f"Tokenizer {tokenizer} not supported. Only gpt2 is supported for this paper.")

    y_train_val = torch.tensor(raw_datasets["train"]["label"])  # type: ignore
    y_test = torch.tensor(raw_datasets["test"]["label"])  # type: ignore

    if holdout_fraction > 0:
        train_indices, val_indices = create_holdout_splits(
            holdout_strategy, y_train_val, holdout_fraction, seed, ensemble_size
        )
    else:
        train_indices = torch.arange(len(y_train_val)).unsqueeze(0)  # shape: (1, num_samples)
        val_indices = torch.tensor([], dtype=torch.long).unsqueeze(0)  # shape: (1, 0)

    # Send the data to device
    x_train_val = x_train_val.to(device)
    attention_mask_train_val = attention_mask_train_val.to(device)
    y_train_val = y_train_val.to(device)
    x_test = x_test.to(device)
    attention_mask_test = attention_mask_test.to(device)
    y_test = y_test.to(device)
    train_indices = train_indices.to(device)
    val_indices = val_indices.to(device)

    if batch_strategy == "same":
        # Use the same data_loader for all models in the ensemble
        train_indices = train_indices[0].unsqueeze(0)
        val_indices = val_indices[0].unsqueeze(0)

    train_datasets = [
        TensorDataset(  # The order has to be x, y, *others
            x_train_val[train_indices_i],
            y_train_val[train_indices_i],
            attention_mask_train_val[train_indices_i],
        )
        for train_indices_i in train_indices
    ]
    train_loader = EnsembleDataLoader(
        [
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
            for train_dataset in train_datasets
        ],
        [lambda x: x],
        ensemble_size,
        handle_attention_mask=True,  # DistilBert requires attention mask
    )
    test_loader = EnsembleDataLoader(
        [DataLoader(TensorDataset(x_test, y_test, attention_mask_test), batch_size=batch_size, shuffle=False)],
        [lambda x: x],  # No transforms needed for text data
        ensemble_size,
        handle_attention_mask=True,  # DistilBert requires attention mask
    )

    if holdout_fraction == 0:
        val_loader = None
    elif holdout_strategy == "overlapping":
        num_samples = len(y_train_val)
        mask = torch.zeros((num_samples, ensemble_size), dtype=torch.bool, device=device)
        mask.scatter_(0, val_indices.T, True)
        x, y, attention_mask, mask = (
            x_train_val[mask.any(dim=1)],
            y_train_val[mask.any(dim=1)],
            attention_mask_train_val[mask.any(dim=1)],
            mask[mask.any(dim=1)],
        )
        val_loader = EnsembleDataLoader(
            [DataLoader(TensorDataset(x, y, attention_mask, mask), batch_size=batch_size, shuffle=False)],
            [lambda x: x],
            ensemble_size,
            handle_attention_mask=True,  # DistilBert requires attention mask
        )
    else:
        val_datasets = [
            TensorDataset(
                x_train_val[val_indices_i],
                y_train_val[val_indices_i],
                attention_mask_train_val[val_indices_i],
            )
            for val_indices_i in val_indices
        ]
        val_loader = EnsembleDataLoader(
            [DataLoader(val_dataset, batch_size=batch_size, shuffle=False) for val_dataset in val_datasets],
            [lambda x: x],
            ensemble_size,
            handle_attention_mask=True,  # DistilBert requires attention mask
        )

    return train_loader, val_loader, test_loader


def create_holdout_splits(
    holdout_strategy: str, y_train: Tensor, holdout_fraction: float, seed: int, ensemble_size: int
):
    num_samples = len(y_train)
    if holdout_strategy == "same":
        train_indices, val_indices = train_test_split(
            torch.arange(num_samples),
            test_size=holdout_fraction,
            random_state=seed,
            stratify=y_train,
        )
        train_indices = train_indices.unsqueeze(0).repeat(ensemble_size, 1)
        val_indices = val_indices.unsqueeze(0).repeat(ensemble_size, 1)
        return train_indices, val_indices
    elif holdout_strategy == "random":
        train_indices = []
        val_indices = []
        for i in range(ensemble_size):
            train_idx, val_idx = train_test_split(
                torch.arange(num_samples),
                test_size=holdout_fraction,
                random_state=seed + i,
                stratify=y_train,
            )
            train_indices.append(train_idx)
            val_indices.append(val_idx)
        return torch.stack(train_indices), torch.stack(val_indices)
    elif holdout_strategy == "disjoint":
        if holdout_fraction * ensemble_size > 1:
            raise ValueError("Cannot split data into disjoint sets. Reduce holdout fraction or ensemble size.")
        all_indices = torch.arange(num_samples)
        remaining_indices = all_indices
        train_indices = []
        val_indices = []
        val_size = int(holdout_fraction * num_samples)
        for i in range(ensemble_size):
            remaining_indices, val_idx = train_test_split(
                remaining_indices, test_size=val_size, random_state=seed, stratify=y_train[remaining_indices]
            )
            train_idx = all_indices[torch.isin(all_indices, val_idx, invert=True)]
            train_indices.append(train_idx)
            val_indices.append(val_idx)
        return torch.stack(train_indices), torch.stack(val_indices)
    elif holdout_strategy == "overlapping":
        if ensemble_size <= 2:
            raise ValueError("Overlapping holdout strategy requires ensemble size > 2.")
        if 0.5 * holdout_fraction * ensemble_size > 1:
            raise ValueError("Cannot split data into overlapping sets. Reduce holdout fraction or ensemble size.")
        all_indices = torch.arange(num_samples)
        remaining_indices = all_indices
        index_chunks = []
        chunk_size = int(0.5 * holdout_fraction * num_samples)
        for i in range(ensemble_size):
            # if last batch, and numnber of remaining indices is equal to chunk size, use all remaining indices
            if i == ensemble_size - 1 and len(remaining_indices) == chunk_size:
                val_chunk = remaining_indices
                index_chunks.append(val_chunk)
                break
            remaining_indices, val_chunk = train_test_split(
                remaining_indices,
                test_size=chunk_size,
                random_state=seed,
                stratify=y_train[remaining_indices],
            )
            index_chunks.append(val_chunk)
        train_indices = []
        val_indices = []
        for i in range(ensemble_size):
            val_idx = torch.cat((index_chunks[i], index_chunks[(i + 1) % ensemble_size]))
            train_idx = all_indices[torch.isin(all_indices, val_idx, invert=True)]
            train_indices.append(train_idx)
            val_indices.append(val_idx)
        return torch.stack(train_indices), torch.stack(val_indices)
    else:
        raise ValueError(f"Unknown holdout strategy: {holdout_strategy}")


class EnsembleGraphDataLoader:
    def __init__(
        self,
        data_loaders: list[DataLoader],
        ensemble_size: int,
    ):
        if len(data_loaders) != 1 and len(data_loaders) != ensemble_size:
            raise ValueError("Number of data_loaders should be 1 or equal to ensemble size")
        if len(set([len(data_loader) for data_loader in data_loaders])) != 1:
            raise ValueError("All data_loaders should have the same length")
        if any(len(data_loader) == 0 for data_loader in data_loaders):
            raise ValueError("Data loaders cannot be empty.")

        self.data_loaders = data_loaders
        self.ensemble_size = ensemble_size

    def __iter__(self):
        iterators = [iter(dataloader) for dataloader in self.data_loaders]

        while True:
            try:
                batches = [next(iterator) for iterator in iterators]
            except StopIteration:
                break

            if len(batches) == 1:
                data = batches[0]
                mask = data.get("mask")
                data_list = data.to_data_list()
                data_list = data_list * self.ensemble_size
                if mask is None:
                    yield pyg.data.Batch.from_data_list(data_list)  # type: ignore
                else:
                    # this is only relevant for overlapping holdout strategy
                    yield pyg.data.Batch.from_data_list(data_list), mask  # type: ignore

            else:
                data_list = []
                for batch in batches:
                    data = batch.to_data_list()
                    data_list.extend(data)
                yield pyg.data.Batch.from_data_list(data_list)  # type: ignore


class EnsembleDataLoader:
    def __init__(
        self,
        data_loaders: list[DataLoader],
        transforms: list[Callable],  # list of transforms, that applied t
        ensemble_size: int,
        handle_attention_mask: bool = False,  # only for distilbert
    ):
        if len(data_loaders) != 1 and len(data_loaders) != ensemble_size:
            raise ValueError("Number of data_loaders should be 1 or equal to ensemble size")
        if len(transforms) != 1 and len(transforms) != ensemble_size:
            raise ValueError("Number of transforms should be 1 or equal to ensemble size")
        if len(set([len(data_loader) for data_loader in data_loaders])) != 1:
            raise ValueError("All data_loaders should have the same length")
        if any(len(data_loader) == 0 for data_loader in data_loaders):
            raise ValueError("Data loaders cannot be empty.")

        self.data_loaders = data_loaders
        self.transforms = transforms
        self.ensemble_size = ensemble_size
        self.handle_attention_mask = handle_attention_mask

    def __iter__(self):
        """
        Iterate over the data loaders and generate batches.

        Yields:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the batched input data
                                             and the corresponding batched target data.
                                             each of shape (ensemble_size * batch_size, ...)
        """
        iterators = [iter(dataloader) for dataloader in self.data_loaders]

        while True:
            try:
                batches = [next(iterator) for iterator in iterators]
            except StopIteration:
                break

            if len(batches) == 1:
                x, y, *others = batches[0]
                y = torch.cat([y for _ in range(self.ensemble_size)], dim=0)

                if len(self.transforms) == 1:
                    x = torch.cat([self.transforms[0](x) for _ in range(self.ensemble_size)], dim=0)
                else:
                    x = torch.cat([transform(x) for transform in self.transforms], dim=0)

                if self.handle_attention_mask:  # Repeat attention mask for ensemble size and put back again
                    attention_mask = others.pop(0)
                    attention_mask = torch.cat([attention_mask for _ in range(self.ensemble_size)], dim=0)
                    yield x, y, attention_mask, *others
                else:
                    yield x, y, *others

            else:
                if len(self.transforms) == 1:
                    x = torch.cat([self.transforms[0](batch[0]) for batch in batches], dim=0)
                else:
                    x = torch.cat([transform(batch[0]) for batch, transform in zip(batches, self.transforms)], dim=0)

                y = torch.cat([batch[1] for batch in batches], dim=0)
                if self.handle_attention_mask:
                    attention_mask = torch.cat([batch[2] for batch in batches], dim=0)
                    if len(batches[0]) > 3:  # handle additional tensors in the batch (eg. holdout mask)
                        yield x, y, attention_mask, *zip(*[batch[3:] for batch in batches])
                    else:
                        yield x, y, attention_mask

                else:
                    if len(batches[0]) > 2:  # If there are additional tensors in the batch
                        yield x, y, *zip(*[batch[2:] for batch in batches])
                    else:
                        yield x, y

    def __len__(self):
        return len(self.data_loaders[0])
