import sys

from cw2.cluster_work import ClusterWork
from cw2.cw_data.cw_pd_logger import PandasLogger
from cw2.cw_data.cw_wandb_logger import WandBLogger

from exp_util.cw2_util.cw2_experiment import Cw2Experiment
from exp_util.cw2_util.print_logger import PrintLogger
from exp_util.experiment import Experiment


class _Cw2Experiment(Cw2Experiment):

    @staticmethod
    def setup_experiment(seed: int = 0,
                         save_path: str = None):
        class _Experiment(Experiment):

            @staticmethod
            def get_default_config():
                config = Experiment.get_default_config()
                config.seed = seed

                config.agent.encoder = "mlp"

                return config

        return _Experiment


    if not any([".yml" in arg for arg in sys.argv]):
        sys.argv.append("experiments/configs/exp_config.yml")
        sys.argv.append("-o")


if __name__ == "__main__":

    cw = ClusterWork(_Cw2Experiment)

    cw.add_logger(PrintLogger())
    cw.add_logger(PandasLogger())
    cw.add_logger(WandBLogger(ignore_keys=["ts"]))

    cw.run()


