from pybullet_multigoal_gym.envs.env_bases import BaseBulletEnv
from pybullet_multigoal_gym.robots.kuka import Kuka
from pybullet_multigoal_gym.scenes.scene_bases import SingleRobotEmptyScene
import pybullet
from pybullet_utils import bullet_client


class Kuka2ObjEnv(BaseBulletEnv):
    def __init__(self, use_real_time_simulation=False, render=True):
        self.robot = Kuka()
        BaseBulletEnv.__init__(self, self.robot, render=render)
        self.reset()
        self._p.setRealTimeSimulation(use_real_time_simulation)

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=9.81, timestep=0.0020, frame_skip=5)

    def _step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()

        state = self.robot.calc_state()  # sets self.to_target_vec

        reward = self.robot.calc_reward()

        return state, reward, False, {}
