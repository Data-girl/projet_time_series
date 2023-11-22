import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline


def create_temporalities(dataframe: pd.DataFrame):
    """créer des variables temporelles"""

    dataframe["heure"] = dataframe.index.hour
    dataframe["jour"] = dataframe.index.day
    dataframe["semaine"] = dataframe.index.isocalendar().week
    dataframe["mois"] = dataframe.index.month
    dataframe["trimestre"] = dataframe.index.quarter
    dataframe["annee"] = dataframe.index.year


def create_lags(dataframe: pd.DataFrame):
    """créer une liste de lags"""

    list_lags = np.arange(364, 1457, 364)

    for lag in list_lags:
        dataframe["lags_" + str(lag)] = dataframe["PJME_MW"].shift(lag)

    return dataframe


def create_rolling_average(dataframe: pd.DataFrame):
    """créer une liste de rolling average"""
    """sequence de 6 heures"""

    list_rolling = np.arange(6, 7, 6)

    for rolling in list_rolling:
        dataframe["rolling_" + str(rolling)] = (
            dataframe["PJME_MW"].rolling(rolling).mean()
        )


def cyclical_encoding(dataframe: pd.DataFrame, col: pd.Series):
    """creer des variations cycliques"""

    max_val = dataframe[col].max()

    dataframe[col + "_sin"] = np.sin(2 * np.pi * dataframe[col] / max_val)
    dataframe[col + "_cos"] = np.cos(2 * np.pi * dataframe[col] / max_val)

    return dataframe.drop([col], axis=1)
