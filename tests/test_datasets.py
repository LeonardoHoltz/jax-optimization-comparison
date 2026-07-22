from pathlib import Path

import pandas as pd

from jax_optim_env.datasets.adult_income import AdultIncomeDataModule
from jax_optim_env.datasets.sklearn import SklearnDataModule


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


def test_adult_income_datamodule_loads_csv(tmp_path):
    csv_path = tmp_path / "adult.csv"
    pd.DataFrame(
        {
            "age": [39, 50],
            "workclass": ["State-gov", "Private"],
            "education": ["Bachelors", "HS-grad"],
            "income": [">50K", "<=50K"],
        }
    ).to_csv(csv_path, index=False)

    manager = AdultIncomeDataModule(
        name="adult_income",
        num_features=8,
        num_classes=1,
        root=tmp_path,
        val_split=0.5,
        batch_size=2,
        seed=0,
    )

    assert manager.X.shape[0] == 2
    assert manager.y.shape[0] == 2
    assert manager.train_data["X"].shape[0] == 1
    assert manager.val_data["X"].shape[0] == 1
