from .time_series_data import ITimeSeriesData, TimeSeriesData, LazyTimeSeriesData
from .handle_coodinate import *

import numpy as np
from data_loader import TableLoader, PathList
from func_helper import identity, pip
import dataframe_helper as dataframe


class CompactEM:
    @staticmethod
    def of(pathlike, preprocessor=identity, lazy=False)->ITimeSeriesData:
        constructor = LazyTimeSeriesData if lazy else TimeSeriesData
        return constructor(TableLoader(
            pathlike,
            {"header": 34},
            [
                dataframe.setTimeSeriesIndex(
                    "YYYY/MM/DD", "hh:mm:ss", inplace=False),
                preprocessor
            ]
        ))

    @staticmethod
    def to_XY_coordinate(x_direction, y_direction, xname="X_converted", yname="Y_converted"):
        old_coordinate = basis_by_North(0, 90)
        new_coordinate = basis_by_North(
            x_direction,
            y_direction
        )
        return lambda df: pip(
            vectors_from_dataframe("Vel NS[cm/s]", "Vel EW[cm/s]"),
            lambda vectors: map(transform_coordinate(
                new_coordinate, old_coordinate), vectors),
            list,
            dataframe_from_vectors(
                columns=[xname, yname], index=df.index)
        )(df)
