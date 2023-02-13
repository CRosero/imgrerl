import sys
import os

myDir = os.getcwd()
sys.path.append(myDir)

from pathlib import Path
path = Path(myDir)
a = str(path.parent.absolute())

sys.path.append(a)

from cw2.cluster_work import ClusterWork
from cw2.cw_data.cw_pd_logger import PandasLogger
from cw2.cw_data.cw_wandb_logger import WandBLogger

from exp_util.cw2_util.cw2_experiment import Cw2Experiment
from exp_util.cw2_util.print_logger import PrintLogger
from exp_util.experiment import Experiment
from pomdp_baselines.utils import augmentation


class _Cw2Experiment(Cw2Experiment):

    @staticmethod
    def setup_experiment(seed: int = 0,
                         save_path: str = None):
        class _Experiment(Experiment):

            @staticmethod
            def get_default_config():
                config = Experiment.get_default_config()
                config.env.env = "cartpole-swingup"
                config.seed = seed


                use_image = True;

                if use_image:
                    config.env.obs_type = "image"
                    config.agent.observ_embedding_size = 0
                    config.rl.buffer_size = 100000 # 1e6
                    config.rl.batch_size = 4 # 32
                    config.rl.sampled_seq_len = 64 # 64
                    
                    config.agent.image_augmentation_type = augmentation.AugmentationType.SAME_OVER_TIME;
                    config.agent.image_augmentation_K = 3;
                    config.agent.image_augmentation_M = 2;
                    config.agent.image_augmentation_actor_critic_same_aug = False;

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


