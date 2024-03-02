import collections
from typing import Sequence

from tempurranet.util.registry import build_from_cfg
from tempurranet.tempurra_global_registry import PROCESS


class Process(object):

    def __init__(self, processes: Sequence[dict], cfg) -> None:
        """
        Compose multiple process sequentially.
        :param processes: Sequence of process object or config dict to be composed.
        :param cfg: The configuration for the current model.
        :return: None
        """
        assert isinstance(processes, collections.abc.Sequence)
        self.processes = []
        for process in processes:
            if isinstance(process, dict):
                process = build_from_cfg(process,
                                         PROCESS,
                                         default_args=dict(cfg=cfg))
                self.processes.append(process)
            elif callable(process):
                self.processes.append(process)
            else:
                raise TypeError('process must be callable or a dict')

    def __call__(self, data: dict) -> Sequence:
        """
        Compose multiple process sequentially
        :param data: A result dict contains the data to process.
        :return: Processed data
        """

        for t in self.processes:
            data = t(data)
            if data is None:
                return []
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.processes:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
