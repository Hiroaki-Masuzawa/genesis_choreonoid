import sys
import os
# sys.path.append(os.path.dirname(__file__) + '/../irsl_rl')

from irsl_rl.rl_env_gs import RLEnvGenesis
import genesis as gs

class Bex24Env(RLEnvGenesis):
    def __init__(self,
                 num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=True,
                 device="cuda", dt=0.02, substeps=2,
                 #robot_urdf_path='urdfs/bex24/bex24.urdf'
                 robot_urdf_path='fake_bex24.urdf'
                 ):
        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer,
                         device, dt, substeps, robot_urdf_path)

    def build_environment(self): ## override
        self.scene.add_entity(gs.morphs.URDF(file="env.urdf", fixed=True))

