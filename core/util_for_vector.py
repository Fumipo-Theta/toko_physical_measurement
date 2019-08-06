import os
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
from IPython.display import display
#import pandas.tseries.offsets as offsets
from func_helper import pip
import dataframe_helper as dataframe
from data_loader import PathList
from data_loader.data_loader.csv_reader import CsvReader
import datetime

from typing import List


class VectorReadOption:
    def __init__(self,
                 read_directory: str,
                 output_directory: str,
                 start_datetime: datetime.datetime,
                 observed_time_window: List[str],
                 time_window: List[str],
                 burst_time: int=2880,
                 header_row: int=None,
                 unit_timeshift: dict={
                     "minutes": 10,
                     "seconds": 1/16
                 },
                 column_name_timeshift: dict={
                     "minutes": "Burst counter []",
                     "seconds": "Ensemble counter []"
                 },
                 ):
        self.read_directory = read_directory
        self.output_directory = output_directory
        self.start_datetime = start_datetime
        self.observed_time_window = observed_time_window
        self.time_window = time_window
        self.burst_time = burst_time
        self.header_row = header_row
        self.unit_timeshift = unit_timeshift
        self.column_name_timeshift = column_name_timeshift

    def tlim(self):
        return pd.to_datetime(self.time_window)

    def tlim_observed(self):
        return pd.to_datetime(self.observed_time_window)


class TableConverter:
    def __init__(self,
                 option: VectorReadOption,
                 reader=CsvReader,
                 vhd_columns=[
                     "Month",
                     "Day",
                     "Year",
                     "Hour",
                     "Minute",
                     "Second",
                     "Burst counter",
                     "No of velocity samples",
                     "Noise amplitude (Beam1)",
                     "Noise amplitude (Beam2)",
                     "Noise amplitude (Beam3)",
                     "Noise correlation (Beam1)",
                     "Noise correlation (Beam2)",
                     "Noise correlation (Beam3)"
                 ],
                 sen_columns=[
                     "Month",
                     "Day",
                     "Year",
                     "Hour",
                     "Minute",
                     "Second",
                     "Error code",
                     "Status code",
                     "Battery voltage",
                     "Sound speed",
                     "Heading",
                     "Pitch",
                     "Roll",
                     "Temperature",
                     "Analog input",
                     "Checksum"
                 ]
                 ):
        self.read_option = option
        self.Reader = reader
        self.vhd_columns = vhd_columns
        self.sen_columns = sen_columns
        self.dat_columns = {
            "columns": [
                "Burst counter []",
                "Ensemble counter []",
                "Velocity X [m/s]",
                "Velocity Y [m/s]",
                "Velocity Z [m/s]",
                "Amplitude Beam1 []",
                "Amplitude Beam2 []",
                "Amplitude Beam3 []",
                "SNR Beam1 [dB]",
                "SNR Beam2 [dB]",
                "SNR Beam3 [dB]",
                "Correlation Beam1 [%]",
                "Correlation Beam2 [%]",
                "Correlation Beam3 [%]",
                "Pressure [m]",
                "Analog input1",
                "Analog input2",
                "Checksum (1=failed)",
            ],
            "drop": [
                "Pressure [m]",
                "Analog input1",
                "Analog input2"
            ]
        }

    @staticmethod
    def convert_csv(path, column_names, output):
        df = pd.read_csv(path, names=column_names, sep=r"\s+")

        if not os.path.isdir(output):
            os.makedirs(output)
        output_path = os.path.join(output, os.path.basename(path)+".csv")

        df.to_csv(output_path)

    def convert_vhd(self):
        TableConverter.convert_csv(
            PathList.match(r"\.vhd$")(self.read_option.read_directory).files()[0],
            self.vhd_columns,
            self.read_option.output_directory
        )
        print("vhd file converted to csv.")

    def convert_sen(self):
        TableConverter.convert_csv(
            PathList.match(r"\.sen$")(self.read_option.read_directory).files()[0],
            self.sen_columns,
            self.read_option.output_directory
        )
        print("sen file converted to csv.")

    def convert_dat(self):
        """
        datファイルをcsvファイルに変換していない場合
        Burst counterと Ensemble counterに基づく時刻を追加して
        datファイルからDataFrameを構築する.
        """
        df = read_original_dat(
            self.read_option,
            self.dat_columns,
            sep=r"\s+",
            preprocess=[
                resetCounter,
                timeShifter,
                #lambda _,__: tee(display),
                lambda meta, _: setShiftedTime(
                    meta.start_datetime,
                    units=meta.unit_timeshift,
                    columns={
                        "minutes": "burstCount",
                        "seconds": "ensembleCount"
                    },
                    new="datetime_reset"
                ),
            ]
        )

        df.set_index(pd.to_datetime(df.datetime), inplace=True)
        print(len(df)/2880/6/24, "days observation")

        """
        vhd ファイルに基づく時間の振り直し
        """
        vhd = self.read_vhd()
        datetime_by_vhd = []
        total_burst = df["Burst counter []"].max()
        start_time = self.read_option.start_datetime
        for i in tqdm(range(total_burst)):
            # vhd データからburst開始時刻をもとめる
            start_time = start_time + \
                datetime.timedelta(minutes=vhd["delta_burst_time"][i])
            shift_seconds = shiftTimeBy(start_time, seconds=1/16)

            ensemble_count = df[df["Burst counter []"]
                                == i+1]["Ensemble counter []"]
            for e in ensemble_count:
                # ensemble カウントごとに1/16秒足す
                datetime_by_vhd.append(shift_seconds(seconds=e))

        df = df.assign(datetime_by_vhd=datetime_by_vhd)
        df.set_index("datetime_by_vhd", inplace=True)
        df_count = df.groupby("Burst counter []").tail(1)
        print(df_count.index)

        df.to_csv(self.read_option.output_directory+"dat.csv")

    def read_vhd(self):
        path = PathList.match(r"\.vhd\.csv$")(
            self.read_option.output_directory).files()[0]

        vhd = self.Reader.create()\
            .setPath(path)\
            .read(0)\
            .assemble(
                toDateTime,
                dataframe.setTimeSeriesIndex("datetime")
        ).df

        # バースト時刻の差 (min)
        delta_burst_time = [0]
        vhd_index = vhd.index

        for i in range(1, len(vhd_index)):
            time_delta = vhd_index[i] - vhd_index[i-1]
            delta_burst_time.append(time_delta.seconds/60)

        vhd = vhd.assign(delta_burst_time=delta_burst_time)
        return vhd

    def read_sen(self):
        path = PathList.match(r"\.sen\.csv$")(
            self.read_option.output_directory).files()[0]

        sen = self.Reader.create()\
            .setPath(path)\
            .read(0)\
            .assemble(
                toDateTime,
                dataframe.setTimeSeriesIndex("datetime")
        ).df
        return sen

    def read_dat(self):
        dat = pd.read_csv(PathList.match(r"dat\.csv")(
            self.read_option.output_directory).files(True)[0], index_col=0, parse_dates=[0])
        return dat

    def get_dataframes(self):
        vhd = self.read_vhd()
        sen = self.read_sen()
        dat = self.read_dat()

        return (vhd, sen, dat)


def shiftTimeBy(start, **shift_units):
    """
    Shift time by unit and counts.

    Parameters
    ----------
    start: datatime.datetime
        Initial date time.
    **shift_units:
        days: float
        hours: float
        minutes: float
        seconds: float
            Unit of time shift by each count

    Return
    ------
    coef: [**shift_counts] -> [datetime.datetime]
        generate 1d ndarray of shifted time from 1d ndarray of shift counts.

        **shift_counts:
            days: int
            hours int
            minutes: int
            seconds: int
                Shift counts.

    Usage
    -----

    timeShift = shiftTimeBy(
        start, minutes=10, seconds=1/4)(
        minutes=df.minute_count, seconds=df.second_count)

    """
    def coef(**shift_counts):
        if (set(shift_units.keys()) != set(shift_counts.keys())):
            raise SystemError("Different keywords !")

            # Keywords of shift_units and shift_counts must be identical.

        shift = {
            key: (shift_units.get(key) if key in shift_units else 0) *
            (int(shift_counts.get(key))-1 if key in shift_counts else 0)
            for key in ["days", "hours", "minutes", "seconds"]
        }

        return (start + datetime.timedelta(**shift)).strftime("%Y/%m/%d %H:%M:%S.%f")
    return np.vectorize(coef)


def setShiftedTime(start, units, columns, new="datetime"):
    """
    Parameters
    ----------
    start: datetime.datetime
        Initial date time.
    units: dict[str: float]
        Dictionary of unit of time shift.
    columns: dict[str: str]
        Dictionary of column name for time shift counts.
    new: str, optional
        Name of time series column.
        Default value is "datetime"

    Return
    ------
    assignShiftedTime: pandas.DataFrame -> pandas.DataFrame
        Add new column of time series made by time shfting.

    Usage
    -----
    s = datetime.datetime(2018,8,18,hour=8,minute=40,second=0)
    d = pd.DataFrame({
        "c" : [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3],
        "e" : [1,2,3,4,5,1,2,3,4,5,1,2,3,4,5]
    })

    new_df = setShiftedTime(
        s,
        units={
            "minutes" : 10,
            "seconds" : 1/4
        },
        columns ={
            "minutes" : "c",
            "seconds" : "e"
        }
    )(df)
    """

    timeShift = shiftTimeBy(start, **units)

    def assignShiftedTime(df):
        arg = {
            key: df[value].values for key, value in columns.items()
        }
        return pd.concat([df, pd.Series(timeShift(**arg), index=df.index, name=new)], axis=1)

    return assignShiftedTime


def test1():
    s = datetime.datetime(2018, 8, 18, hour=8, minute=40, second=0)
    d = pd.DataFrame({
        "c": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        "e": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    })

    timeShift = shiftTimeBy(s, minutes=10, seconds=1/4)

    t = timeShift(minutes=d.c.values, seconds=d.e.values)

    return d.assign(t=t)


def test2():
    s = datetime.datetime(2018, 8, 18, hour=8, minute=40, second=0)
    d = pd.DataFrame({
        "c": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        "e": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    })

    print(d.index.values + 1)

    timeShifter = setShiftedTime(
        s,
        units={
            "minutes": 10,
            "seconds": 1/4
        },
        columns={
            "minutes": "c",
            "seconds": "e"
        }
    )

    return timeShifter(d)


def test3():
    d = {
        "a": 0, "b": 1
    }

    e = {
        "a": 0, "b": 1
    }
    print(set(d.keys()) == set(e.keys()))


def timeShifter(meta: VectorReadOption, common):
    return setShiftedTime(
        meta.start_datetime,
        units=meta.unit_timeshift,
        columns=meta.column_name_timeshift
    )


def resetCounter(meta: VectorReadOption, common):
    def resetter(df):

        burstCount = np.ceil((df.index.values + 1) / meta.burst_time)
        ensembleCount = df.index.values - \
            (burstCount - 1) * meta.burst_time + 1
        # print(burstCount)
        # print(ensembleCount)
        return df.assign(burstCount=burstCount, ensembleCount=ensembleCount)
    return resetter


def read_original_dat(meta: VectorReadOption, common, sep=",", preprocess=[timeShifter]):

    files = PathList.match(r"\.dat$")(meta.read_directory).files(verbose=True)

    dfs = []
    for file in files:
        reader = pd.read_csv(
            file,
            sep=sep,
            chunksize=50000,
            names=common["columns"],
            header=meta.header_row
        )

        df = pd.concat(
            pip(
                *[f(meta, common) for f in preprocess]
            )(r) for r in tqdm(reader)
        )

        display(df.head(10))
        dfs.append(df)

    df = pd.concat(dfs)

    df_droped = df.drop(common["drop"], axis=1)

    return df_droped


def toDateTime(df):
    def f(y, m, d, H, M, S):
        return datetime.datetime(y, m, d, hour=H, minute=M, second=S).strftime("%Y/%m/%d %H:%M:%S.%f")
    vFunc = np.vectorize(f)
    dt = vFunc(
        df.Year.values,
        df.Month.values,
        df.Day.values,
        df.Hour.values,
        df.Minute.values,
        df.Second.values,
    )
    return pd.concat([df, pd.Series(dt, index=df.index, name="datetime")], axis=1)


if __name__ is "__main__":
    display(test1())
    display(test2())
    print(test3())
