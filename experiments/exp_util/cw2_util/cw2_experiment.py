import os
import sys
myDir = os.getcwd()
sys.path.append(myDir)

from pathlib import Path
path = Path(myDir)
a = str(path.parent.absolute())

sys.path.append(a)

from cw2.experiment import AbstractIterativeExperiment
from cw2.cw_data import cw_logging
from cw2.cw_data.cw_wandb_logger import WandBLogger
from cw2.cw_error import ExperimentSurrender
from typing import Optional
from experiments.exp_util.config_dict import ConfigDict
from experiments.exp_util.cw2_util.print_logger import PrintLogger



class Cw2Experiment(AbstractIterativeExperiment):

    def __init__(self):
        super(Cw2Experiment, self).__init__()
        self._experiment = None

    @staticmethod
    def check_and_update(params: dict,
                         conf_dict: ConfigDict,
                         ignore_keys: Optional[list[str]] = None):

        val_key_list = list(conf_dict.keys()) + list(conf_dict.subconfig_names()) + \
                       ([] if ignore_keys is None else ignore_keys)
        for k in params.keys():
            if k not in val_key_list:
                raise AssertionError("Unused key: ", k)
        conf_dict.rec_update(params)

    @staticmethod
    def setup_experiment(seed: int = 0,
                         save_path: str = None):
        raise NotImplementedError

    @property
    def save_interval(self) -> int:
        return -1

    def initialize(self, config: dict, rep: int, logger: cw_logging.AbstractLogger) -> None:

        self._log_path = config["_rep_log_path"]

        params = config.get("params", None)

        self._experiment_cls = self.setup_experiment(seed=rep,
                                                     save_path=self._log_path)

        conf_dict = self._experiment_cls.get_default_config()
        self.check_and_update(params, conf_dict)

        for l in logger:
            if isinstance(l, PrintLogger):
                l.preprocess(conf_dict)
            if isinstance(l, WandBLogger):
                if l.run is not None:
                    l.run.config.update(conf_dict.get_raw_dict(), allow_val_change=True)
        print("params: \n", params, "\n")
        # print("conf_dict: ", conf_dict)
        self._experiment = self._experiment_cls(conf_dict)

    def iterate(self, config: dict, rep: int, n: int) -> dict:
        # how convenient
        log_dict = self._experiment.iterate(n)
        return log_dict

    def finalize(self, surrender: ExperimentSurrender = None, crash: bool = False):
        pass

    def save_state(self, config: dict, rep: int, n: int) -> None:
        pass