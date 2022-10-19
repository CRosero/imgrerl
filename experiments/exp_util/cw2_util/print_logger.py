from cw2.cw_data.cw_logging import AbstractLogger
import logging
import sys
import os
import numpy as np

from exp_util.logging import get_log_lines


class _CWFormatter(logging.Formatter):

    def __init__(self):
        super(_CWFormatter, self).__init__()
        self.std_formatter = logging.Formatter('[%(name)s] %(message)s')
        self.red_formatter = logging.Formatter('[%(asctime)s] %(message)s')

    def format(self, record: logging.LogRecord):
        if record.levelno <= logging.ERROR:
            return self.std_formatter.format(record)
        else:
            return self.red_formatter.format(record)


class PrintLogger(AbstractLogger):

    def initialize(self, config: dict, rep: int, rep_log_path: str) -> None:
        formatter = _CWFormatter()

        self._exclude_keys = ["ts", "rep", "iter"]

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        sh.setLevel(logging.INFO)

        fh = logging.FileHandler(os.path.join(config["_rep_log_path"], "cw_print.log"))
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)

        self._logger = logging.getLogger(name="CW")
        self._logger.propagate = False
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(fh)
        self._logger.addHandler(sh)

        self._logger.info("------------------------------------------")
        self._logger.info("Starting Repetition {:03d}".format(rep))
        self._logger.info("------------------------------------------")

    def preprocess(self, log_dict: dict) -> None:
        line_list = str(log_dict).split("\n")[1:-1]
        for line in line_list:
            self._logger.info(line)

    def process(self, log_data: dict) -> None:
        iteration = log_data["iter"]
        lines = get_log_lines(iteration=iteration, log_dict=log_data)
        for l in lines:
            self._logger.info(l)

    def finalize(self) -> None:
        for handler in self._logger.handlers:
            handler.close()
            self._logger.removeHandler(handler)

    def load(self):
        pass

    @property
    def raw_logger(self):
        return self._logger
