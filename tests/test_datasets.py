import numpy as np
import pytest

from optim_env.datasets.pmlb import PMLBDataModule
from optim_env.datasets.sklearn import SklearnDataModule
from optim_env.datasets.tabm import TabMDataModule


def test_sklearn_dataset_manager_loads_california_housing(tmp_path):
    manager = SklearnDataModule(
        name="california_housing",
        num_features=8,
        num_classes=0,
        root=tmp_path,
        val_split=0.2,
        batch_size=4,
        seed=0,
    )

    assert manager.X.shape[1] == 8
    assert manager.train_data["X"].shape[0] > 0
    assert manager.val_data["X"].shape[0] > 0
    assert manager.train_data["y"].shape[0] == manager.train_data["X"].shape[0]
    assert manager.val_data["y"].shape[0] == manager.val_data["X"].shape[0]


def test_pmlb_datamodule_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_dataset.pmlb"):
        PMLBDataModule(
            name="mushroom",
            num_features=22,
            num_classes=2,
            root=tmp_path,
            val_split=0.2,
            batch_size=2,
            seed=0,
        )


def test_tabm_datamodule_loads_split_npy_files(tmp_path):
    ds_dir = tmp_path / "microsoft"
    ds_dir.mkdir()

    rng = np.random.default_rng(0)
    for split, n in (("train", 6), ("val", 2), ("test", 2)):
        np.save(ds_dir / f"X_num_{split}.npy", rng.normal(size=(n, 3)).astype(np.float32))
        np.save(ds_dir / f"X_bin_{split}.npy", rng.integers(0, 2, size=(n, 1)).astype(np.float32))
        np.save(ds_dir / f"X_cat_{split}.npy", np.array([["a"], ["b"]] * (n // 2)))
        np.save(ds_dir / f"Y_{split}.npy", rng.normal(size=(n,)).astype(np.float32))

    manager = TabMDataModule(
        name="microsoft",
        num_features=6,
        num_classes=0,
        root=tmp_path,
        val_split=0.2,
        batch_size=2,
        seed=0,
    )

    assert manager.X.shape == (10, 6)
    assert manager.y.shape == (10,)
    # The original train/val/test split boundaries are preserved as-is, not re-shuffled.
    assert manager.train_data["X"].shape[0] == 6
    assert manager.val_data["X"].shape[0] == 2
    assert manager.test_data["X"].shape[0] == 2


def test_tabm_datamodule_test_split_is_none_when_absent(tmp_path):
    ds_dir = tmp_path / "microsoft"
    ds_dir.mkdir()

    rng = np.random.default_rng(0)
    for split, n in (("train", 6), ("val", 2)):
        np.save(ds_dir / f"X_num_{split}.npy", rng.normal(size=(n, 3)).astype(np.float32))
        np.save(ds_dir / f"Y_{split}.npy", rng.normal(size=(n,)).astype(np.float32))

    manager = TabMDataModule(
        name="microsoft",
        num_features=3,
        num_classes=0,
        root=tmp_path,
        val_split=0.2,
        batch_size=2,
        seed=0,
    )

    assert manager.test_data is None
    with pytest.raises(NotImplementedError):
        manager.get_test_ds()


def test_tabm_datamodule_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_dataset.huggingface"):
        TabMDataModule(
            name="microsoft",
            num_features=136,
            num_classes=0,
            root=tmp_path,
            val_split=0.2,
            batch_size=2,
            seed=0,
        )


def _fake_tabm_tar(tmp_path, monkeypatch, extra_member=None):
    import io
    import tarfile

    from optim_env.prepare_dataset import huggingface

    tar_path = tmp_path / "data.tar"
    with tarfile.open(tar_path, "w") as tar:
        for name in ("california/Y_test.npy", "adult/Y_test.npy", extra_member):
            if name:
                tar.addfile(tarfile.TarInfo(name), io.BytesIO())
    monkeypatch.setattr(huggingface, "hf_hub_download", lambda *a, **k: str(tar_path))
    return huggingface


def test_tabm_download_extracts_only_named_datasets(tmp_path, monkeypatch):
    huggingface = _fake_tabm_tar(tmp_path, monkeypatch)
    huggingface.download(["california"], tmp_path / "out")
    assert (tmp_path / "out" / "california" / "Y_test.npy").exists()
    assert not (tmp_path / "out" / "adult").exists()
    with pytest.raises(ValueError, match="house"):
        huggingface.download(["house"], tmp_path / "out")


def test_tabm_download_refuses_paths_outside_root(tmp_path, monkeypatch):
    huggingface = _fake_tabm_tar(tmp_path, monkeypatch, extra_member="california/../../evil.txt")
    with pytest.raises(ValueError, match="unsafe"):
        huggingface.download(["california"], tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()
