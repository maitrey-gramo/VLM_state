import os

import pytest
from huggingface_hub.hf_api import HfApi
from pytest_cases import fixture

DATASET_REPO_NAME = "amitkparekh/vima"
MODEL_REPO_NAME = "amitkparekh/cogelot"


@fixture(scope="module")
def hf_api() -> HfApi:
    return HfApi()


@pytest.mark.skipif(os.environ.get("CI") is not None, reason="Skipping HF API tests in CI")
def test_hf_token_has_write_permission(hf_api: HfApi) -> None:
    assert hf_api.get_token_permission() == "write"


@pytest.mark.skipif(os.environ.get("CI") is not None, reason="Skipping HF API tests in CI")
def test_can_access_hf_dataset_repo(hf_api: HfApi) -> None:
    assert hf_api.repo_exists(DATASET_REPO_NAME, repo_type="dataset")


@pytest.mark.skipif(os.environ.get("CI") is not None, reason="Skipping HF API tests in CI")
def test_can_access_hf_model_repo(hf_api: HfApi) -> None:
    assert hf_api.repo_exists(MODEL_REPO_NAME, repo_type="model")
