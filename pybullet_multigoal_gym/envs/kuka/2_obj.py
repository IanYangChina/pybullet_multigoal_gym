from pybullet_multigoal_gym.envs.env_bases import BaseBulletMGEnv
from pybullet_multigoal_gym.robots.kuka import Kuka
from pybullet_multigoal_gym.scenes.scene_bases import SingleRobotEmptyScene


class Kuka2ObjEnv(BaseBulletMGEnv):
    def __init__(self, use_real_time_simulation=False, render=True):
        self.robot = Kuka()
        BaseBulletMGEnv.__init__(self, self.robot, render=render)
        self.reset()
        self._p.setRealTimeSimulation(use_real_time_simulation)

    def create_single_player_scene(self, bullet_client):
        # fixed time step = $timestep * $frame_skip
        return SingleRobotEmptyScene(bullet_client, gravity=9.81, timestep=0.002, frame_skip=20)

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()

        state = self.robot.calc_state()  # sets self.to_target_vec

        reward = self.robot.calc_reward()

        return state, reward, False, {}
