import datetime
import numpy as np
import pandas as pd
import re
from func_helper import pip


def remove_edge_spaces(line: str)->str:
    return pip(
        lambda s: re.sub(r"^\s+", "", s),
        lambda s: re.sub(r"\s+$", "", s)
    )(line)


def split_by_spaces(line: str)->list:
    return re.split(r"\s+", line)


def get_fields(line: str)->list:
    return pip(
        remove_edge_spaces,
        split_by_spaces
    )(line)


def split_datetime(dt_row: str)->list:
    return pip(
        get_fields,
        lambda s: s[0:5]
    )(dt_row)


def to_datetime(y, m, d, H="00", M="00", S="00"):
    return pd.to_datetime(f"{y}/{m}/{d} {H}:{M}:{S}", format="%y/%m/%d %H:%M:%S")


def shift_datetime(timedelta):
    return lambda datetime: datetime + timedelta


def dibisible_by(divider: int):
    return lambda divided: (divided) % divider is 0


assert(
    split_datetime(" 18 8 18 0 0 2400 1\n") ==
    ["18", "8", "18", "0", "0"]
)

assert(
    get_fields("  1 2 3 4 5 6 7 8 9 10 \n") ==
    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
)


class Nowphas:
    """

    """

    def __init__(self, data_path, machine_sensitivity, interval_sec):
        self.data_path = data_path
        self.machine_sensitivity = machine_sensitivity
        self.interval_sec = interval_sec

    def __new__(cls, *arg, **kwargs):
        return super().__new__(cls)

    def __repr__(self):
        return f"""
        Data path: {self.data_path}
        Sensitivity: {self.machine_sensitivity} [m]
        Interval of measurement: {self.interval_sec} [s]
        """

    def dataframe(self):
        return Nowphas.to_dataframe(
            *Nowphas.parse_file(self.data_path),
            self.interval_sec,
            self.machine_sensitivity
        )

    @staticmethod
    def to_dataframe(start_datetime, data_array, interval_sec, sensitivity):
        df = pd.DataFrame({
            "tide [m]": data_array
        }, index=pd.date_range(start_datetime, periods=len(data_array), freq=f"{interval_sec}S"))

        df.replace([999, 9999], np.nan, inplace=True)
        df["tide [m]"] = df["tide [m]"] * sensitivity
        return df

    @staticmethod
    def parse_file(path):
        data_row_num = 151
        data_separator = r"\s"
        start_datetime = None

        data = []

        is_time_meta_row = dibisible_by(data_row_num)

        with open(path) as file:
            for line_num, line in enumerate(file):

                if line_num is 0:
                    start_datetime = shift_datetime(datetime.timedelta(
                        minutes=-10))(to_datetime(*split_datetime(line)))
                if (not is_time_meta_row(line_num)):
                    data += [float(d) for d in get_fields(line)]

        return (start_datetime, data)
