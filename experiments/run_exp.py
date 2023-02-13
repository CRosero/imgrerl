import sys
import os 
os.environ["MUJOCO_GL"] = "egl"

sys.path.append("/home/i53/student/c_rosero/workspace/pomdp_baselines")
sys.path.append("/home/i53/student/c_rosero/workspace/pomdp_baselines/pomdp_baselines")

from exp_util.experiment import Experiment
from exp_util.logging import print_log
from pomdp_baselines.utils import augmentation

config = Experiment.get_default_config()
config.env.env = "cartpole-swingup"


use_image = True;

if use_image:
    config.env.obs_type = "image"
    config.agent.observ_embedding_size = 0
    config.rl.buffer_size = 100000 # 1e6
    config.rl.batch_size = 4 # 32
    config.rl.sampled_seq_len = 64 # 64
    
    config.agent.image_augmentation_type = augmentation.AugmentationType.SAME_OVER_TIME;
    config.agent.image_augmentation_K = 1;
    config.agent.image_augmentation_M = 1;
    config.agent.image_augmentation_actor_critic_same_aug = False;
    
else: 
    config.env.obs_type = "state"






#config.agent.encoder = "gru"
#config.env.image_input.width = 64
#config.env.image_input.height = 64
#config.env.image_input.channels = 3
print(config)

experiment = Experiment(config)

for i in range(1000):
    log_dict = experiment.iterate(i)
    print_log(i, log_dict)

