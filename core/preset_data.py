import dataclasses
from typing import List, Dict
from structured_plot.plot_action.action import PlotAction


class PresetSetting(dataclasses):
    dataInfo: dict
    index: List[str]
    option: dict
    plot: List[PlotAction]
    directory: str
