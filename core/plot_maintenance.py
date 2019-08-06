from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np

from structured_plot import Figure, SubplotTime, save_plot, FigureSizing
from structured_plot.plot_action.action import DuplicateLast
from data_loader import PathList
from func_helper import pip, tee, identity
import iter_helper as it
import dict_helper as d

matchCsv = r"\.[cC](sv|SV)$"


class PresetSetup:
    @staticmethod
    def create():
        styler = PresetSetup()
        return styler

    def __init__(self):
        self.figure_sizing = FigureSizing()
        self._xlim = []
        self.ylim = []

    def get_figure_geometry(self):
        return self.figure_sizing

    def set_preset(self, preset):
        self.preset = preset
        return self

    def get_preset(self):
        return self.preset

    @property
    def xlim(self):
        return self._xlim

    @xlim.setter
    def xlim(self, value):
        self._xlim = value

    def set_limit(self, xlim=None, ylim=None):
        self.xlim = xlim if xlim is not None else []
        self.ylim = ylim if ylim is not None else []
        return self

    def get_xlim(self):
        return self.xlim

    def get_ylim(self):
        return self.ylim

    def get_limit(self, axis="both"):
        if axis is "x":
            return {"xlim": self.get_xlim()}
        elif axis is "y":
            return {"ylim": self.get_ylim()}
        else:
            return {
                "xlim": self.xlim,
                "ylim": self.get_ylim()
            }

    def set_figure_size(self, **kwargs):
        self.figure_sizing.set_figure_style(**kwargs)
        return self

    def get_figure_size(self):
        return self.figure_sizing.get_figsize()

    def get_figure_padding(self):
        return self.figure_sizing.get_padding()

    def set_figure_style(self, **kwargs):
        self.figure_style = kwargs
        return self

    def get_figure_style(self):
        return self.figure_style

    def set_axes_size(self, **kwargs):
        self.figure_sizing.set_axes_style(**kwargs)
        return self

    def get_axes_margin(self):
        return self.figure_sizing.get_margin()

    def set_axes_style(self, **kwargs):
        self.axes_style = kwargs
        return self

    def get_axes_style(self):
        return self.axes_style


class IPeriod_with_window:
    def set_period(self, period: List[str]):
        pass

    def get_period(self) -> Tuple[List[str], List[str]]:
        pass

    def get_start_date(self)->str:
        pass

    def get_end_date(self) -> str:
        pass

    def is_start_date(self, date: str) -> bool:
        pass

    def is_end_date(self, date: str) -> bool:
        pass


class Period_with_window(IPeriod_with_window):

    def __init__(self, period, start_window_time: str="00:00:00", end_window_time: str="23:59:59"):
        self.start_window_time = start_window_time
        self.end_window_time = end_window_time
        self.set_period(period)

    def is_start_date(self, date: str)-> bool:
        return self.start_date == date

    def is_end_date(self, date: str) -> bool:
        return self.end_date == date

    def get_start_date(self) -> str:
        return self.start_date

    def get_end_date(self) -> str:
        return self.end_date

    def set_period(self, period: List[str]):
        """
        period = [
            yyyy/mm/dd HH:MM:SS,
            yyyy/mm/dd HH:MM:SS
        ]
        """

        self.start_date = period[0].split()[0]
        self.end_date = period[1].split()[0]
        self.period = period
        self.window = [
            f"{self.start_date} {self.start_window_time}",
            f"{self.end_date} {self.end_window_time}"
        ]

    def get_period(self)-> Tuple[List[str], List[str]]:
        return (
            self.period,
            self.window
        )


class IPeriodStorage:
    def __init__(self):
        pass

    def set_a_period(self):
        pass

    def get_periods(self)-> List[IPeriod_with_window]:
        pass

    def get_start_dates(self) -> List[str]:
        pass

    def get_end_dates(self) -> List[str]:
        pass

    def get_date_pairs(self) -> List[Tuple[str, str]]:
        pass


class PeriodStorage(IPeriodStorage):
    """
    _storage = {
        "2018/10/10 07:15:00-2018/10/10 07:43:00" : period_with_window
    }
    """

    def __init__(self):
        self.Period_with_window = PeriodStorage.IPeriod_with_window()
        self._storage = {}

    def set_a_period(self, period: List[str]):
        period_and_window = self.Period_with_window(period)
        self._storage.update({self._hash(period): period_and_window})

    def get_periods(self)-> List[IPeriod_with_window]:
        return self._storage.values()

    def _hash(self, period: List[str]) -> str:
        return period[0]+"-"+period[1]

    def get_start_dates(self) -> List[str]:
        return pip(
            it.mapping(lambda p: p.get_start_date()),
            list
        )(self.get_periods())

    def get_end_dates(self) -> List[str]:
        return pip(
            it.mapping(lambda p: p.get_end_date()),
            list
        )(self.get_periods())

    def get_date_pairs(self) -> List[Tuple[str, str]]:
        return pip(
            it.mapping(lambda p: (
                p.get_start_date(),
                p.get_end_date()
            )),
            list
        )(self.get_periods())

    @staticmethod
    def IPeriod_with_window()-> IPeriod_with_window:
        return Period_with_window


class SiteObject:
    """
    異なる地点間で, 同じ日付に行われたメンテナンス時の記録が必要となる.
    そのとき, メンテナンスが行われた地点とそうでない地点をフィルタリングする必要がある.
        ->フィルタリングのためのkeyが必要
    メンテナンス期間を含むtimestampの窓が必要.
    メンテナンス期間の情報から自動生成することが望ましい.
        ->メンテンナンス開始日の00:00:00から, 終了日の24:00:00までとする.
    """

    def __new__(cls, *arg, **kwargs):
        return super().__new__(cls)

    def __init__(self, site_name: str):
        self.name = site_name
        self.maintenance = {
            "default": SiteObject.IPeriodStorage()
        }
        self.interval = {
            "default": SiteObject.IPeriodStorage()
        }

    def get_name(self) -> str:
        return self.name

    def get_file_selector(self) -> List[str]:
        return [self.get_name()]

    def set_maintenance(self, list_of_periods: List[List[str]], key: str="default"):
        """
        arg = [[str,str], [str,str],...]
        """
        if key not in self.maintenance:
            self.maintenance[key] = SiteObject.IPeriodStorage()

        for p in list_of_periods:
            self.maintenance[key].set_a_period(p)
        return self

    def get_maintenance(
        self,
        start_dates: Optional[List[str]]=None,
        end_dates: Optional[List[str]]=None,
        key: str="default"
    ) -> List[Tuple[List[str], List[str]]]:
        _key = "default" if key not in self.maintenance else key
        return SiteObject._get_periods(
            self.maintenance[_key],
            start_dates,
            end_dates
        )

    def set_interval(self, list_of_periods: List[List[str]], key: str="default"):
        if key not in self.interval:
            self.interval[key] = SiteObject.IPeriodStorage()

        for p in list_of_periods:
            self.interval[key].set_a_period(p)
        return self

    def get_interval(
        self,
        start_dates: Optional[List[str]]=None,
        end_dates: Optional[List[str]]=None,
        key: str="default"
    ) -> List[Tuple[List[str], List[str]]]:
        _key = "default" if key not in self.interval else key
        return SiteObject._get_periods(
            self.interval[_key],
            start_dates,
            end_dates
        )

    def get_start_dates_of_maintenance(self, key="default") -> List[str]:
        return SiteObject._collect_start_dates(self.maintenance[key])

    def get_start_dates_of_interval(self, key="default") -> List[str]:
        return SiteObject._collect_start_dates(self.interval[key])

    def get_end_dates_of_maintenance(self, key="default") -> List[str]:
        return SiteObject._collect_end_dates(self.maintenance[key])

    def get_end_dates_of_interval(self, key="default") -> List[str]:
        return SiteObject._collect_end_dates(self.interval[key])

    def get_date_pairs_of_maintenance(self, key="default") -> List[Tuple[str, str]]:
        return SiteObject._collect_date_pairs(self.maintenance[key])

    def get_date_pairs_of_interval(self, key="default") -> List[Tuple[str, str]]:
        return SiteObject._collect_date_pairs(self.interval[key])

    @staticmethod
    def _collect_start_dates(storage: IPeriodStorage) -> List[str]:
        return storage.get_start_dates()

    @staticmethod
    def _collect_end_dates(storage: IPeriodStorage) -> List[str]:
        return storage.get_end_dates()

    @staticmethod
    def _collect_date_pairs(storage: IPeriodStorage) -> List[Tuple[str, str]]:
        return storage.get_date_pairs()

    @staticmethod
    def _get_periods(
        storage: IPeriodStorage,
        start_dates: Optional[List[str]]=None,
        end_dates: Optional[List[str]]=None
    ) -> List[IPeriod_with_window]:

        start_date_filter = it.filtering(lambda p: p.get_start_date(
        ) in start_dates) if start_dates is not None else identity

        end_date_filter = it.filtering(lambda p: p.get_end_date(
        ) in end_dates) if end_dates is not None else identity

        return pip(
            start_date_filter,
            end_date_filter,
            list
        )(storage.get_periods())

    @staticmethod
    def IPeriodStorage()-> IPeriodStorage:
        return PeriodStorage()


def bandPlot(xpos: Tuple[list], **kwargs):
    import structured_plot.plot_action as plot

    return plot.yband(xpos=plot.multiple(*xpos), **kwargs)


def genMaintenanceBox(maintenancePeriods: List[IPeriod_with_window], **kwargs):
    """
    Parameters
    ----------
    maintenancePeriods: List[str]
        メンテナンスの開始日時と終了日時を表す文字列のリスト.

    Returns
    -------
    maintenancePlot: Callable Dict -> pandas.DataFrame -> ax -> ax
        メンテナンス期間を表す長方形をプロットするためのアクション.

    Example
    -------
    from src.plot_util import Figure
    from src.setting import default, maintenance
    from src.plot_maintenance import getMaintenanceBox, presetSubplot

    maintenanceBox = genMaintenanceBox(*maintenance["st1-1"])
    figure = Figure()
    figure.add_subplot(
        presetSubplot(default)(
            name,
            fileSelector=[site],
            plot=[*maintenanceBox]
        )
    )
    figure.show()
    """
    return bandPlot(
        pip(
            it.mapping(lambda p: p.get_period()[0] if isinstance(
                p, IPeriod_with_window) else p),
            it.mapping(pd.to_datetime),
            tuple
        )(maintenancePeriods), **kwargs
    ) if len(maintenancePeriods) > 0 else bandPlot([None])


def presetSubplot(default: PresetSetup):
    """
    Parameters
    ----------
    default: PresetSetup
        機器ごとのcsvファイルのメタデータや，
        デフォルトでのプロット設定を定義し,
        また, プロットスタイルやx軸範囲のデフォルトを定義したクラス

    Returns
    -------
    generate: Callable

        Parameters
        ----------
        name: str
            setting のpresetに登録された項目(測定項目)の名前.
        fileSelector: List[str], optional:
            プロットするファイルのパスを絞り込むための文字列および
            正規表現リスト.
            リストの全ての文字列を含むパスが選択される.
        plot: Callable: Dict -> pandas.DataFrame -> ax -> ax, optional
            プロットに用いるアクションのリスト.
            何も指定しなければpresetに登録されたアクションが実行される.
        option: Dict, optional
            プロット時の設定の辞書.
            presetに登録されたプロット列やy軸範囲, yラベルを上書きする.
        limit: Dict, optional
            x軸, y軸の範囲の辞書.
            presetに登録された設定を上書きする.
        style: dict, optional
            フォントサイズやx軸目盛りフォーマットのリスト.
            デフォルト設定を上書きする.

        **kwd
        -----
        transformer: List[Callable[[pandas.Dataframe],pandas.Dataframe]]
            読み込んだDataframeを加工するための関数のリスト.

        Returns
        ------
        subplot: SubplotTime
            preset情報と， その一部を引数で上書きした情報がセットされたSubplotTimeインスタンス

        Example
        -------
        from src.plot_util import Figure
        from src.plot_maintenance import presetSubplot
        from src.setting import setting

        figure = Figure({"width" : 15, "height" : 5})

        # ctSalinityのプリセットを使用， st1-3のファイルをプロットする．
        # また， y軸ラベルを書き換える．
        # x軸範囲を設定する．
        subplot = presetSubplot(default)(
            "ctSalinity",
            fileSelector=["st1-3"],
            option={
                "ylabel" : "塩分"
            },
            limit = {
                "xlim" : ["2018/07/25 00:00:00", "2018/08/15 00:00:00"]
            }
        )

        figure.add_subplot(subplot)
        figure.show()

    """
    def as_tuple(ite):
        return ite if type(ite) is tuple else tuple(ite)

    def get_csv_file_list(directorys, file_selector):
        if type(directorys) is tuple:
            return [PathList.match(matchCsv, *file_selector)(directory) for directory in directorys]
        else:
            return [PathList.match(matchCsv, *file_selector)(directorys)]

    def generate(preset_name: str, fileSelector: list=[], plot=[], option={}, limit={}, style={}, plotOverwrite=[], **kwargs):
        subplotStyle = {**default.get_axes_style(), **style}
        subplotLimit = {**default.get_limit("x"), **limit}

        preset = default.get_preset().get(preset_name)

        subplot = SubplotTime(**subplotStyle)\
            .add(
            data=DuplicateLast(*get_csv_file_list(preset["directory"], fileSelector)),
            dataInfo=preset["dataInfo"],
            index=preset.get("index", None),
            plot=[*preset["plot"], *plot] if not plotOverwrite else plotOverwrite,
            limit=subplotLimit,
            **d.mix(
                preset["option"],
                option,
                kwargs
            )

        )
        """
        subplot = SubplotTime.create(**subplotStyle)\
            .setPreset(default.get_preset())\
            .usePreset(
                preset_name,
                fileSelector=[matchCsv, *fileSelector],
                plot=[*plot],
                option={**option},
                limit=subplotLimit,
                **kwd
        )
        """
        return subplot
    return generate


def totalPeriodAtSite(default: PresetSetup):
    """
    Parameters
    ----------
    default: PresetSetup
        機器ごとのcsvファイルのメタデータや，
        デフォルトでのプロット設定を定義し,
        また, プロットスタイルやx軸範囲のデフォルトを定義したクラス

    Returns
    -------
    generate: Callable

        Parameters
        ----------
        site: SiteObject
            プロットするsite名．
        machineNames: List[str]
            presetに登録された測定項目のうち， プロットに利用するもののリスト.
        limit: Dict, optional
            x軸, y軸の範囲の辞書.
            presetに登録された設定を上書きする.
        style: dict, optional
            フォントサイズやx軸目盛りフォーマットのリスト.
            デフォルト設定を上書きする.
        saveDir: str, optional
            画像を保存するディレクトリのパス．
            指定しなければ， デフォルトで"./image/{site}/"．
        file: str, optional
            画像ファイル名のpost-fix．
            "{site}{file}.png"という画像ファイルが生成される．
            デフォルトでは空文字．
        figure: Figure, optional
            予め作成したFigureインスタンス上にプロットするときに使用する．
            デフォルトでは新規にFigureインスタンスを作成し，プロットする．

        Returns
        -------
        figure: Figure
            Subplotを登録したFigureインスタンス．
        save: Callable: str -> None
            png画像を保存するアクションを起こす関数．
        maitenanceBox : List[Callable: Dict -> pandas.DataFrame -> ax -> ax]
            次以降に追加するSubplot上にメンテナンス期間をプロットするためのアクションを表す関数．
        style: Dict
            次以降に追加するSubplotに継承するための，プロットスタイルを定義したDict．

    Example
    -------
    from src.plot_maintenance import totalPeriodAtSite, presetSubplot
    from src.setting import default, maintenance

    figure, save, plotMaintenanceBox, inheritedStyle = \
        totalPeriodAtSite(default,maintenance)(
            "st1-3",
            ["photon","ctSalinity","ctTemperature"],
            limit = {
                "xlim" : ["2018/07/25 00:00:00", "2018/08/15 00:00:00"]
            }
        )
    figure.add_subplot(気象庁plot(plotMaintenanceBox,inheritedStyle))
    figure.show()
    save()
    """
    def generate(site: SiteObject, machineNames: List[str], limit={}, style={}, saveDir=None, file="", figure=None):

        if figure == None:
            figure = Figure()

        subplotStyle = {**default.get_axes_style(), **style}
        subplotLimit = {**default.get_limit("x"), **limit}

        for name in machineNames:
            maintenanceBox = genMaintenanceBox(site.get_maintenance(key=name))
            intervalBox = genMaintenanceBox(site.get_interval(key=name))

            figure.add_subplot(
                presetSubplot(default)(
                    name,
                    fileSelector=site.get_file_selector(),
                    plot=[maintenanceBox, intervalBox],
                    style=subplotStyle,
                    limit=subplotLimit
                ),
                names=name
            )

        save = save_plot(
            "./image/"+site.get_name()+"/" if saveDir == None else saveDir,
            site.get_name()+file
        )

        return (figure, save, (maintenanceBox, intervalBox),
                {"style": subplotStyle, "limit": subplotLimit})
    return generate


def maintenanceAtSite(default: PresetSetup):
    """
    Parameters
    ----------
    default: PresetSetup
        機器ごとのcsvファイルのメタデータや，
        デフォルトでのプロット設定を定義し,
        また, プロットスタイルやx軸範囲のデフォルトを定義したクラス


    Returns
    -------
    generate: Callable

        Parameters
        ----------
        site: SiteObject
            プロットするsite名．
        machineNames: List[str]
            presetに登録された測定項目のうち， プロットに利用するもののリスト.
        limit: Dict, optional
            x軸, y軸の範囲の辞書.
            presetに登録された設定を上書きする.
        style: dict, optional
            フォントサイズやx軸目盛りフォーマットのリスト.
            デフォルト設定を上書きする.
        saveDir: str, optional
            画像を保存するディレクトリのパス．
            指定しなければ， デフォルトで"./image/{site}/"．
        file: str, optional
            画像ファイル名のpost-fix．
            "{site}{file}.png"という画像ファイルが生成される．
            デフォルトでは空文字．
        figure: Figure, optional
            予め作成したFigureインスタンス上にプロットするときに使用する．
            デフォルトでは新規にFigureインスタンスを作成し，プロットする．

        Returns
        -------
        figure: Figure
            Subplotを登録したFigureインスタンス．
        save: Callable: str -> None
            png画像を保存するアクションを起こす関数．
        maitenanceBox : List[Callable: Dict -> pandas.DataFrame -> ax -> ax]
            次以降に追加するSubplot上にメンテナンス期間をプロットするためのアクションを表す関数．
        style: Dict
            次以降に追加するSubplotに継承するための，プロットスタイルを定義したDict．

    Example
    -------
    from src.plot_maintenance import maintenanceAtSite, presetSubplot
    from src.setting import default, maintenance

    for mainteNum in range(len(maintenance[site])):
        figure,save,maintenanceBox, inheritedStyle = \
            maintenaceAtSite(default,maintenance)(
                site, machineNames, mainteNum)
            figure.add_subplot(気象庁plot(default)(
                maintenanceBox, inheritedStyle))
            figure.show()
            save()
    """
    def generate(
        site: SiteObject,
        machineNames: List[str],
        limit={},
        style={},
        saveDir=None,
        file=""
    ):

        subplotStyle = {
            **default.get_axes_style(),
            "xFmt": "%m/%d\n%H:%M",
            **style
        }

        unique_maintenance_date_pairs = pip(
            it.mapping(lambda so: so.get_date_pairs_of_maintenance()),
            it.mapping(lambda pairs: it.reducing(lambda acc, e: [
                       *acc, e] if e not in acc else acc)([])(pairs)),
            it.reducing(lambda acc, e: [
                *acc, *it.filtering(lambda v: v not in acc)(e)
            ])([]),
            lambda dates: sorted(
                dates, key=lambda item: pd.to_datetime(item[0]))
        )([site])

        returns = []

        for start_date, end_date in unique_maintenance_date_pairs:

            subplotLimit = {"xlim": site.get_maintenance([start_date], [end_date])[0].get_period()[1],
                            **limit}

            figure = Figure()

            for name in machineNames:
                maintenanceBox = genMaintenanceBox(
                    site.get_maintenance([start_date], [end_date], key=name))

                figure.add_subplot(
                    presetSubplot(default)(
                        name,
                        fileSelector=site.get_file_selector(),
                        plot=[maintenanceBox],
                        style=subplotStyle,
                        limit=subplotLimit
                    ),
                    names=name
                )

            save = save_plot(
                "./image/"+site.get_name()+"/" if saveDir == None else saveDir,
                f'{site.get_name()+file}-maintenance-{start_date}-{end_date}'
            )

            returns.append(
                (figure, save, [maintenanceBox],
                 {"style": subplotStyle, "limit": subplotLimit}
                 )
            )

        return returns
    return generate


def totalPeriodForMachine(default: PresetSetup):
    """
    指定した複数の地点につき, ある特定の機器の時系列データをプロットする.

    Parameters
    ----------
    default: PresetSetup
        機器ごとのcsvファイルのメタデータや，
        デフォルトでのプロット設定を定義し,
        また, プロットスタイルやx軸範囲のデフォルトを定義したクラス

    Returns
    -------
    generate: Callable

        Parameters
        ----------
        sites: List[SiteObject]
            プロットするsite名のリスト．
        machineName: str
            presetに登録された測定項目のうち， プロットするものの名称.
        limit: Dict, optional
            x軸, y軸の範囲の辞書.
            presetに登録された設定を上書きする.
        style: dict, optional
            フォントサイズやx軸目盛りフォーマットのリスト.
            デフォルト設定を上書きする.
        saveDir: str, optional
            画像を保存するディレクトリのパス．
            指定しなければ， デフォルトで"./image/{site}/"．
        file: str, optional
            画像ファイル名のpost-fix．
            "{site}{file}.png"という画像ファイルが生成される．
            デフォルトでは空文字．
        figure: Figure, optional
            予め作成したFigureインスタンス上にプロットするときに使用する．
            デフォルトでは新規にFigureインスタンスを作成し，プロットする．

        Returns
        -------
        figure: Figure
            Subplotを登録したFigureインスタンス．
        save: Callable: str -> None
            png画像を保存するアクションを起こす関数．
        style: Dict
            次以降に追加するSubplotに継承するための，プロットスタイルを定義したDict．

    Example
    -------
    from src.plot_maintenance import totalPeriodForMachine, presetSubplot
    from src.setting import default, maintenance

    figure, save, inheritedStyle = totalPeriodForMachine(default, maintenance)(
        ["st1-1","st1-3","st4-1","st4-3"],
        "photon",
        limit={
            "xlim": ["2018/07/25 00:00:00", "2018/08/15 00:00:00"]
        }
    )
    figure.add_subplot(気象庁plot({},inheritedStyle))
    figure.show()
    save()
    """
    def generate(sites: List[SiteObject], machineName: str, limit={}, style={}, saveDir=None, file="", figure=None):

        if figure == None:
            figure = Figure()

        subplotStyle = {**default.get_axes_style(), **style}
        subplotLimit = {**default.get_limit("x"), **limit}

        for site in sites:
            maintenanceBox = genMaintenanceBox(
                site.get_maintenance(key=machineName))
            intervalBox = genMaintenanceBox(site.get_interval(key=machineName))

            figure.add_subplot(
                presetSubplot(default)(
                    machineName,
                    fileSelector=site.get_file_selector(),
                    plot=[maintenanceBox, intervalBox],
                    option={},
                    ylabel=site.get_name(),
                    style=subplotStyle,
                    **subplotLimit
                ),
                names=machineName+"-"+site.get_name()
            )

        save = save_plot(
            "./image/"+machineName+"/" if saveDir == None else saveDir,
            machineName+file+"-all"
        )

        return (figure, save,
                {"style": subplotStyle, "limit": subplotLimit})
    return generate


def maintenanceForMachine(default: PresetSetup):
    """
    Parameters
    ----------
    default: PreseSetup
        機器ごとのcsvファイルのメタデータや，
        デフォルトでのプロット設定を定義し,
        また, プロットスタイルやx軸範囲のデフォルトを定義したクラス

    Returns
    -------
    generate: Callable

        Parameters
        ----------
        sites: List[SiteObject]
            プロットするsite名のリスト．
        machineName: str
            presetに登録された測定項目のうち， プロットするものの名称.
        mianteNum: int
            何番目のmaintenance期間をプロットするかを指定する.
        limit: Dict, optional
            x軸, y軸の範囲の辞書.
            presetに登録された設定を上書きする.
        style: dict, optional
            フォントサイズやx軸目盛りフォーマットのリスト.
            デフォルト設定を上書きする.
        saveDir: str, optional
            画像を保存するディレクトリのパス．
            指定しなければ， デフォルトで"./image/{site}/"．
        file: str, optional
            画像ファイル名のpost-fix．
            "{site}{file}.png"という画像ファイルが生成される．
            デフォルトでは空文字．
        figure: Figure, optional
            予め作成したFigureインスタンス上にプロットするときに使用する．
            デフォルトでは新規にFigureインスタンスを作成し，プロットする．

        Returns
        -------
        figure: Figure
            Subplotを登録したFigureインスタンス．
        save: Callable: str -> None
            png画像を保存するアクションを起こす関数．
        style: Dict
            次以降に追加するSubplotに継承するための，プロットスタイルを定義したDict．

    Example
    -------
    from src.plot_maintenance import maintenanceForMachine, presetSubplot
    from src.setting import default, maintenance

    for n in range(mainteTime):
        figure, save, inheritedStyle = maintenanceForMachine(default, maintenance)(
            sites,
            "photon",
            n,
            limit={"ylim":[0,2000]}
        )
        ax=figure.show()
        ax["photon-気中部"].set_ylim([0,4000])
        save()
    """
    def generate(site_objects: List[SiteObject], machineName: str, limit={}, style={}, saveDir=None, file=""):

        subplotStyle = {
            **default.get_axes_style(),
            "xFmt": "%m/%d\n%H:%M",
            **style
        }

        unique_maintenance_date_pairs = pip(
            it.mapping(lambda so: so.get_date_pairs_of_maintenance()),
            it.mapping(lambda pairs: it.reducing(lambda acc, e: [
                       *acc, e] if e not in acc else acc)([])(pairs)),
            it.reducing(lambda acc, e: [
                *acc, *it.filtering(lambda v: v not in acc)(e)
            ])([]),
            lambda dates: sorted(
                dates, key=lambda item: pd.to_datetime(item[0]))
        )(site_objects)

        returns = []

        for start_date, end_date in unique_maintenance_date_pairs:

            figure = Figure()

            # siteごとにメンテナンスの回数が異なる.
            subplotLimit = {
                "xlim": pip(
                    it.mapping(lambda so: so.get_maintenance(
                        [start_date], [end_date])),
                    it.filtering(lambda p: len(p) > 0),
                    list
                )(site_objects)[0][0].get_period()[1], **limit}

            for so in site_objects:

                maintenanceBox = genMaintenanceBox(so.get_maintenance(
                    [start_date], [end_date], key=machineName))

                figure.add_subplot(
                    presetSubplot(default)(
                        machineName,
                        fileSelector=so.get_file_selector(),
                        plot=[maintenanceBox],
                        option={},
                        ylabel=so.get_name(),
                        style=subplotStyle,
                        limit=subplotLimit
                    ),
                    names=machineName+"-"+so.get_name()
                )

            save = save_plot(
                f'./image/{machineName if saveDir is None else saveDir}/',
                f'{machineName}{file}-maintenance-{start_date}-{end_date}'
            )

            returns.append(
                (
                    figure,
                    save,
                    {
                        "style": subplotStyle,
                        "limit": subplotLimit
                    }
                )
            )

        return returns
    return generate


def addLayeredPlot(default: PresetSetup, maintenance: dict, siteSymbol: dict):
    """
    Parameters
    ----------
    default: PresetSetup
        機器ごとのcsvファイルのメタデータや，
        デフォルトでのプロット設定を定義し,
        また, プロットスタイルやx軸範囲のデフォルトを定義したクラス1
    maintenance: Dict
        各siteのメンテナンス期間のリストと,
        メンテナンス前後のx軸範囲からなるDict.
    siteSymbol: Dict
        各サイトごとにプロットシンボルを定義したDict

    Returns
    -------
    generate: Callable

        Parameters
        ----------
        sites: List[str]
            プロットするsite名のリスト．
        machine: str
            presetに登録された測定項目のうち， プロットするものの名称.
        option: Dict, optional
            y軸範囲やy軸ラベルを定義したDict
        limit: Dict, optional
            x軸, y軸の範囲の辞書.
            presetに登録された設定を上書きする.
        style: dict, optional
            フォントサイズやx軸目盛りフォーマットのリスト.
            デフォルト設定を上書きする.
        saveDir: str, optional
            画像を保存するディレクトリのパス．
            指定しなければ， デフォルトで"./image/{site}/"．
        file: str, optional
            画像ファイル名のpost-fix．
            "{site}{file}.png"という画像ファイルが生成される．
            デフォルトでは空文字．
        figure: Figure, optional
            予め作成したFigureインスタンス上にプロットするときに使用する．
            デフォルトでは新規にFigureインスタンスを作成し，プロットする．

        Returns
        -------
        figure: Figure
            Subplotを登録したFigureインスタンス．
        save: Callable: str -> None
            png画像を保存するアクションを起こす関数．
        style: Dict
            次以降に追加するSubplotに継承するための，プロットスタイルを定義したDict．

    Example
    -------
    from src.plot_maintenance import addLayeredPlot
    from src.setting import default, maintenance, siteSymbol

    figure, save, inheritedStyle = totalPeriodForMachine(default, maintenance)(
        "st1-3",
        ["photon", "ctSalinity", "ctTemperature"],
        limit={
            "xlim": ["2018/07/25 00:00:00", "2018/08/15 00:00:00"]
        }
    )
    figure.add_subplot(気象庁plot({},inheritedStyle))
    figure.show()
    save()
    """
    def generate(sites: List[str], machine: str, option={}, limit={}, style={}, saveDir=None, file="", figure=None):

        if figure == None:
            figure = Figure()

        subplotLimit = {**default.get_limit("x"), **limit}
        subplotStyle = {**default.get_axes_style(), **style}

        subplot = SubplotTime(subplotStyle)
        subplot.setPreset(default.get_preset())

        for site in sites:
            maintenanceBox = bandPlot(tuple(maintenance[site]["maintenance"]))

            subplot.usePreset(
                machine,
                fileSelector=[matchCsv, site],
                plot=[maintenanceBox],
                option={**option, **siteSymbol[site], **subplotLimit},
                # limit=subplotLimit
            )

        figure.add_subplot(subplot)

        save = save_plot(
            "./image/"+machine+"/" if saveDir == None else saveDir,
            machine+file+"-all-layered"
        )
        return (figure, save,
                {"limit": subplotLimit, "style": subplotStyle})
    return generate
