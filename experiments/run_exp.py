from exp_util.experiment import Experiment
from exp_util.logging import print_log
config = Experiment.get_default_config()
config.env.env = "cartpole-swingup"
config.env.obs_type = "position"

config.agent.encoder = "gru"
print(config)

experiment = Experiment(config)

for i in range(1000):
    log_dict = experiment.iterate(i)
    print_log(i, log_dict)

