import itertools
from pathlib import Path

import datasets
import torch
from pytest_cases import fixture, param_fixture
from torch.utils.data import DataLoader

from cogelot.data.collate import collate_preprocessed_instances_from_hf_dataset
from cogelot.data.parse import (
    VIMAInstanceParser,
    get_all_raw_instance_directories,
)
from cogelot.modules.instance_preprocessor import InstancePreprocessor
from cogelot.structures.model import PreprocessedInstance
from cogelot.structures.vima import VIMAInstance

mission_task = param_fixture(
    "mission_task",
    [
        "follow_order",
        "manipulate_old_neighbor",
        "novel_adj",
        "novel_noun",
        "pick_in_order_then_restore",
        "rearrange",
        "rearrange_then_restore",
        "rotate",
        "same_shape",
        "scene_understanding",
        "sweep_without_exceeding",
        "twist",
        "visual_manipulation",
    ],
    scope="session",
)


@fixture(scope="session")
def fixture_storage_dir() -> Path:
    """Fixture storage directory."""
    return Path("storage/fixtures")


@fixture(scope="session")
def raw_instance_dir(fixture_storage_dir: Path, mission_task: str) -> Path:
    """Fixture storage directory."""
    mission_id = "000000"
    return fixture_storage_dir.joinpath(mission_task, f"{mission_id}/")


@fixture(scope="session")
def vima_instance_parser() -> VIMAInstanceParser:
    return VIMAInstanceParser(keep_null_action=False)


@fixture(scope="session")
def vima_instance(
    raw_instance_dir: Path, vima_instance_parser: VIMAInstanceParser
) -> VIMAInstance:
    """A single normalized VIMA instance."""
    return vima_instance_parser(raw_instance_dir)


@fixture(scope="session")
def all_vima_instances(
    fixture_storage_dir: Path, vima_instance_parser: VIMAInstanceParser
) -> list[VIMAInstance]:
    """All normalized VIMA instances."""
    parsed_instances = (
        vima_instance_parser(instance_dir)
        for instance_dir in get_all_raw_instance_directories(fixture_storage_dir)
    )
    return list(parsed_instances)


@fixture(scope="session")
def vima_instances_dataset(all_vima_instances: list[VIMAInstance]) -> datasets.Dataset:
    """All normalized VIMA instances."""
    dataset = datasets.Dataset.from_list(
        [instance.model_dump() for instance in all_vima_instances],
        features=VIMAInstance.dataset_features(),
    )
    dataset = dataset.with_format("torch")
    return dataset


@fixture(scope="session")
def preprocessed_instance(
    vima_instance: VIMAInstance,
    instance_preprocessor: InstancePreprocessor,
    torch_device: torch.device,
) -> PreprocessedInstance:
    """A single preprocessed instance."""
    return instance_preprocessor.preprocess(vima_instance).transfer_to_device(torch_device)


@fixture(scope="session")
def all_preprocessed_instances(
    vima_instance_parser: VIMAInstanceParser,
    fixture_storage_dir: Path,
    instance_preprocessor: InstancePreprocessor,
    torch_device: torch.device,
) -> list[PreprocessedInstance]:
    """All preprocessed instances."""
    parsed_instances = (
        vima_instance_parser(instance_dir)
        for instance_dir in get_all_raw_instance_directories(fixture_storage_dir)
    )
    preprocessed_instances = (
        instance_preprocessor.preprocess(instance).transfer_to_device(torch_device)
        for instance in parsed_instances
    )
    return list(preprocessed_instances)


@fixture(scope="session")
def preprocessed_instances_dataset(
    all_preprocessed_instances: list[PreprocessedInstance],
) -> datasets.Dataset:
    # Repeat the input data multiple times
    num_cycles = 5
    all_preprocessed_instances = list(
        itertools.chain.from_iterable([all_preprocessed_instances for _ in range(num_cycles)])
    )
    dataset = datasets.Dataset.from_list(
        [instance.model_dump() for instance in all_preprocessed_instances],
        features=PreprocessedInstance.dataset_features(),
    )
    dataset = dataset.with_format(
        "torch", columns=PreprocessedInstance.hf_tensor_fields, output_all_columns=True
    )
    return dataset


@fixture(scope="session")
def vima_dataloader(
    preprocessed_instances_dataset: datasets.Dataset,
) -> DataLoader[list[PreprocessedInstance]]:
    return DataLoader(
        preprocessed_instances_dataset,  # pyright: ignore[reportArgumentType]
        batch_size=2,
        collate_fn=collate_preprocessed_instances_from_hf_dataset,
        num_workers=0,
    )
