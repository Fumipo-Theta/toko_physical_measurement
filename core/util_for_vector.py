import os
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
from IPython.display import display
#import pandas.tseries.offsets as offsets
from func_helper import pip
from matdat import getFileList


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


def timeShifter(meta, common):
    return setShiftedTime(
        meta["start"],
        units=meta["units_timeShift"],
        columns=meta["columns_timeShift"]
    )


def resetCounter(meta, common):
    def resetter(df):

        burstCount = np.ceil((df.index.values + 1) / meta["burstTime"])
        ensembleCount = df.index.values - \
            (burstCount - 1) * meta["burstTime"] + 1
        # print(burstCount)
        # print(ensembleCount)
        return df.assign(burstCount=burstCount, ensembleCount=ensembleCount)
    return resetter


def read_original_dat(meta, common, sep=",", preprocess=[timeShifter]):

    files = getFileList(r"\.dat$")(meta["directory"]).files()

    dfs = []
    for file in files:
        reader = pd.read_csv(
            file,
            sep=sep,
            chunksize=50000,
            names=common["columns"],
            header=meta["header"]
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
