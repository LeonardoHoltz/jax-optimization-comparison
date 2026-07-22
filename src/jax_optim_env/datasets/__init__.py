from .adult_income import AdultIncomeDataModule
from .default import DataModule, XYDataset
from .pmlb import PMLBDataModule
from .sklearn import SklearnDataModule

__all__ = [
    "DataModule",
    "XYDataset",
    "PMLBDataModule",
    "SklearnDataModule",
    "AdultIncomeDataModule",
]
