import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import pybullet
from pybullet_utils import bullet_client

from pkg_resources import parse_version


class BaseBulletMGEnv(gym.Env):
	"""
	Base class for Bullet physics simulation environments in a Scene.
	These environments create single-player scenes and behave like normal Gym environments, if
	you don't use multiplayer.
	"""

	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second': 60
		}

	def __init__(self, robot, render=False, seed=0):
		self.scene = None
		self.physicsClientId = -1
		self.ownsPhysicsClient = 0
		self.isRender = render
		self.robot = robot
		self.seed(seed=seed)
		self._cam_dist = 1.2
		self._cam_yaw = -90
		self._cam_pitch = -30
		self._render_width = 320
		self._render_height = 240

		self.action_space = robot.action_space
		self.observation_space = robot.observation_space

	def configure(self, args):
		self.robot.args = args

	def seed(self, seed=None):
		self.np_random, seed = gym.utils.seeding.np_random(seed)
		self.robot.np_random = self.np_random # use the same np_randomizer for robot as for env
		return [seed]

	def reset(self):
		if self.physicsClientId < 0:
			self.ownsPhysicsClient = True

			if self.isRender:
				self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
			else:
				self._p = bullet_client.BulletClient()

			self.physicsClientId = self._p._client
			self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
			self._p.resetDebugVisualizerCamera(self._cam_dist+0.2, self._cam_yaw-30, self._cam_pitch, [0, 0, 0.4])

		if self.scene is None:
			self.scene = self.create_single_player_scene(self._p)
		if not self.scene.multiplayer and self.ownsPhysicsClient:
			self.scene.episode_restart(self._p)

		self.robot.scene = self.scene

		self.frame = 0
		self.done = 0
		self.reward = 0
		dump = 0
		s = self.robot.reset(self._p)
		self.potential = self.robot.calc_potential()
		return s

	def render(self, mode="human"):
		if mode == "human":
			self.isRender = True
		if mode != "rgb_array":
			return np.array([])

		base_pos = [0, 0, 0.4]
		if hasattr(self,'robot'):
			if hasattr(self.robot,'body_xyz'):
				base_pos = self.robot.body_xyz

		view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
			cameraTargetPosition=base_pos,
			distance=self._cam_dist,
			yaw=self._cam_yaw,
			pitch=self._cam_pitch,
			roll=0,
			upAxisIndex=2)
		proj_matrix = self._p.computeProjectionMatrixFOV(
			fov=60, aspect=float(self._render_width)/self._render_height,
			nearVal=0.1, farVal=100.0)
		(_, _, px, _, _) = self._p.getCameraImage(
		width = self._render_width, height=self._render_height, viewMatrix=view_matrix,
			projectionMatrix=proj_matrix,
			renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
			)
		rgb_array = np.array(px)
		rgb_array = rgb_array[:, :, :3]
		return rgb_array

	def close(self):
		if self.ownsPhysicsClient:
			if self.physicsClientId >= 0:
				self._p.disconnect()
		self.physicsClientId = -1

	def HUD(self, state, a, done):
		pass
