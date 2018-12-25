import dataclasses
from typing import List, Dict
from matdat.matdat.plot.action import PlotAction


class PresetSetting(dataclasses):
    dataInfo: dict
    index: List[str]
    option: dict
    plot: List[PlotAction]
    directory: str
