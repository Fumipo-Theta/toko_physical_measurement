import abc
from data_loader import IDataLoader
from func_helper import identity


def widest_time_range(df):
    return [f"{df.index[0]}", f"{df.index[-1]}"]


class ITimeSeriesData(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_data(self, **kwargs):
        pass

    @abc.abstractmethod
    def get_time_range(self, **kwargs):
        pass


class TimeSeriesData(ITimeSeriesData):
    def __init__(self, source: IDataLoader):
        self.data = source.query()

    def get_data(self, filter_func=identity):
        return filter_func(self.data)

    def get_time_range(self, filter_func=identity, range_selector=widest_time_range):
        return range_selector(filter_func(self.data))


class LazyTimeSeriesData(ITimeSeriesData):
    def __init__(self, source: IDataLoader):
        self.data = source

    def get_data(self, **query_option):
        return self.data.query(**query_option)

    def get_time_range(self, range_selector=widest_time_range, **query_option):
        return range_selector(self.data.query(**query_option))
