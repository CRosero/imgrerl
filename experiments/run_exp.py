import sys

sys.path.append("/home/i53/student/c_rosero/workspace/pomdp_baselines")
sys.path.append("/home/i53/student/c_rosero/workspace/pomdp_baselines/pomdp_baselines")

from exp_util.experiment import Experiment
from exp_util.logging import print_log
config = Experiment.get_default_config()
config.env.env = "cartpole-swingup"


#config.env.obs_type = "state"


config.env.obs_type = "image"
config.agent.observ_embedding_size = 0
config.rl.buffer_size = 500
config.rl.batch_size = 2
config.rl.sampled_seq_len = 4



config.agent.encoder = "gru"
#config.env.image_input.width = 64
#config.env.image_input.height = 64
#config.env.image_input.channels = 3
print(config)

experiment = Experiment(config)

for i in range(1000):
    log_dict = experiment.iterate(i)
    print_log(i, log_dict)

