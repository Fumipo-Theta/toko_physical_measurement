from .time_series_data import ITimeSeriesData, TimeSeriesData, LazyTimeSeriesData

import numpy as np
from data_loader import TableLoader, PathList
from func_helper import identity
import dataframe_helper as dataframe


class Vector:
    @staticmethod
    def of(pathlike, preprocessor=identity, lazy=False)->ITimeSeriesData:
        constructor = LazyTimeSeriesData if lazy else TimeSeriesData
        return constructor(TableLoader(
            pathlike,
            {},
            [
                dataframe.setTimeSeriesIndex(
                    "datetime_by_vhd", inplace=False),
                preprocessor
            ]
        ))

    @staticmethod
    def to_beam_coordinate(transform_matrix):
        """
        Return transformation function for Vector DataFrame.

        Parameters
        ----------
        transform_matrix: np.ndarray (shape is (3,3))
            Transform matrix from beam coordinate to XYZ coordinate.
            Recorded in hdr file.
        """
        def transformer(df, velocity_column_names=["Velocity X [m/s]", "Velocity Y [m/s]", "Velocity Z [m/s]"]):
            beam1 = []
            beam2 = []
            beam3 = []

            for xyz in np.array(df[velocity_column_names]):
                solution = np.linalg.solve(transform_matrix, xyz)
                beam1.append(solution[0])
                beam2.append(solution[1])
                beam3.append(solution[2])

            return (beam1, beam2, beam3)
        return transformer
