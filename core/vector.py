from .time_series_data import ITimeSeriesData, TimeSeriesData, LazyTimeSeriesData

import numpy as np
from data_loader import TableLoader, PathList
from func_helper import identity
import dataframe_helper as dataframe


class Vector:
    """
    Helper class for handling vector data.

    classmethods
    ------------
    of
    to_beam_coordinate
    burst_of
    reduce_noise
    enumerate_burst_in
    """

    @staticmethod
    def of(pathlike, preprocessor=identity, lazy=False)->ITimeSeriesData:
        """
        Create ITimeseriesData of vector data.

        Example
        -------
        vector = Vector.of(PathList.match("dat.csv")("./data/vector"))
        """
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
        Return transformation function for Vector DataFrame

        Example
        -------
        beam1, beam2, beam3 = Vector.to_beam_coordinate(np.ndarray([
            [1,0,0],
            [0,1,0],
            [0,0,1]
        ]))(vector_data)

        Parameters
        ----------
        transform_matrix: np.ndarray (shape is (3,3))
            Transform matrix from beam coordinate to XYZ coordinate.
            Recorded in hdr file.

        return
        ------
            pandas.DataFrame, List[str] -> Tuple(List[int|float])
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

    @staticmethod
    def burst_of(num: int):
        """
        Extract data of a given burst count.

        Example
        -------
        Vector.burst_of(16)(vector_data)

        Parameters
        ----------
        num: int
            Value of the burst count.
        """
        print(f"Use vector data of Burst count: {num}")

        def apply(df):
            return df[df["Burst counter []"] == num]
        return apply

    @staticmethod
    def reduce_noise(lowest_corr=0, lowest_snr=-10):
        """
        Select data having higher correlation and S/N ratio.

        Example
        -------
        Vector.reduce_noise(70, 30)(vector_data)

        Parameters
        ----------
        lowest_corr: float
            Lower threshould of beam correlations [%]
        lowest_snr: float
            Lower threshould of S/N ratio of beams
        """
        print(
            f"Use vector data only \n All Beam Correlation > {lowest_corr} [%]\n All Beam S/N > {lowest_snr} [db]")

        def apply(df):
            return df[(df["Correlation Beam1 [%]"] > lowest_corr)
                      & (df["Correlation Beam2 [%]"] > lowest_corr)
                      & (df["Correlation Beam3 [%]"] > lowest_corr)
                      & (df["SNR Beam1 [dB]"] > lowest_snr)
                      & (df["SNR Beam2 [dB]"] > lowest_snr)
                      & (df["SNR Beam3 [dB]"] > lowest_snr)
                      ]
        return apply

    @staticmethod
    def enumerate_burst_in(time_range):
        """
        Enumerate burst count at least overwrapped with the given time range.

        Example
        -------
        burst_list = Vector.enumerate_burst_in(["2019/1/1 00:00", "2019/1/1 10:20:30"])
        burst_count, time_range = burst_list[0]

        Parameter
        ---------
        time_range: List[str]
            List having length of 2 composed of strings of time stamp.

        Return
        ------
        List[Tuple[int,List[str]]]
            List of tuple of burst count and
                list of time stamp string of time range of the count,
                such as,
        [
            (0, ["2019/1/1 00:00:000000", "2019/1/1 00:02:237500"]),
            (1, ["2019/1/1 00:10:000000", "2019/1/1 00:12:237500"]),
            ...
        ]
        """
        def apply(df):
            return [(
                bc,
                [_df.head(1).index.strftime("%y/%m/%d %H:%M:%f").values[0],
                 _df.tail(1).index.strftime("%y/%m/%d %H:%M:%f").values[0]]
            ) for bc, _df in dataframe.time_range(dataframe.close_interval)(*time_range)(df).groupby("Burst counter []")
            ]
        return apply
