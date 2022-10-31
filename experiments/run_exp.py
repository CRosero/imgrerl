from exp_util.experiment import Experiment
from exp_util.logging import print_log
config = Experiment.get_default_config()
config.env.env = "cheetah-run"
print(config)

experiment = Experiment(config)


for i in range(1000):
    log_dict = experiment.iterate(i)
    print_log(i, log_dict)

