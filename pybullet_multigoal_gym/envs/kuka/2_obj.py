from pybullet_multigoal_gym.envs.env_bases import BaseBulletMGEnv
from pybullet_multigoal_gym.robots.kuka import Kuka


class Kuka2ObjEnv(BaseBulletMGEnv):
    def __init__(self, render=True):
        robot = Kuka()
        BaseBulletMGEnv.__init__(self, robot=robot, render=render, seed=0,
                                 use_real_time_simulation=False,
                                 gravity=9.81, timestep=0.002, frame_skip=20, num_solver_iterations=5)
        self.reset()

    def _step_callback(self):
        pass

    def _reset_callback(self):
        pass
