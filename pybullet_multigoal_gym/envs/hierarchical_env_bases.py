import gym
import numpy as np
import pybullet
from gym.utils import seeding
from gym import spaces
from pybullet_utils import bullet_client


class HierarchicalBaseBulletMGEnv(gym.Env):
    """
    Base class for hierarchical multi-goal RL task, based on PyBullet and Gym.
    """
    def __init__(self,
                 robot, render=False, seed=0,
                 use_real_time_simulation=False,
                 gravity=9.81, timestep=0.002, frame_skip=20, num_solver_iterations=5):
        self.physicsClientId = -1
        self.ownsPhysicsClient = 0
        self.isRender = render
        self.robot = robot
        self.seed(seed=seed)
        self._p = None
        self._cam_dist = 1.2
        self._cam_yaw = -90
        self._cam_pitch = -30
        self._render_width = 320
        self._render_height = 240
        self._gravity = gravity
        self._timestep = timestep
        self._frame_skip = frame_skip
        self._num_solver_iterations = num_solver_iterations
        self._use_real_time_simulation = use_real_time_simulation
        self.configure_bullet_client()

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        obs = self.reset()
        self.action_space = robot.action_space
        self.observation_space = spaces.Dict(dict(
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
            state=spaces.Box(-np.inf, np.inf, shape=obs['state'].shape, dtype='float32'),
            achieved_sub_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_sub_goal'].shape, dtype='float32'),
            achieved_sub_goal_image=spaces.Box(-np.inf, np.inf, shape=obs['achieved_sub_goal_image'].shape, dtype='float32'),
            desired_sub_goal=spaces.Box(-np.inf, np.inf, shape=obs['desired_sub_goal'].shape, dtype='float32'),
            desired_sub_goal_image=spaces.Box(-np.inf, np.inf, shape=obs['desired_sub_goal_image'].shape, dtype='float32'),
            achieved_final_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_final_goal'].shape, dtype='float32'),
            desired_final_goal=spaces.Box(-np.inf, np.inf, shape=obs['desired_final_goal'].shape, dtype='float32'),
            desired_final_goal_image=spaces.Box(-np.inf, np.inf, shape=obs['desired_final_goal_image'].shape, dtype='float32'),
        ))

    @property
    def dt(self):
        # simulation timestep
        # the product of these two values, dt, reflects how much real time it takes to execute a robot action
        # at default, dt = 0.002 * 20 = 0.04 seconds
        # also see the last line of method configure_bullet_client() below
        return self._timestep * self._frame_skip

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.robot.reset(self._p)
        self.task_reset()
        obs = self._get_obs()
        return obs

    def step(self, action):
        self.robot.apply_action(action, self._p)
        self._p.stepSimulation()
        obs = self._get_obs()
        sub_reward, sub_goal_achieved = self._compute_reward(obs['achieved_sub_goal'], obs['desired_sub_goal'])
        final_reward, final_goal_achieved = self._compute_reward(obs['achieved_final_goal'], obs['desired_final_goal'])
        self._step_callback()
        info = {
            'sub_goal_achieved': sub_goal_achieved,
            'final_goal_achieved': final_goal_achieved
        }
        reward = {
            'sub_reward': sub_reward,
            'final_reward': final_reward
        }
        return obs, reward, False, info

    def render(self, mode="human"):
        if mode == "human":
            self.isRender = True
        if mode != "rgb_array":
            return np.array([])

        base_pos = [0, 0, 0.4]
        if hasattr(self, 'robot'):
            if hasattr(self.robot, 'body_xyz'):
                base_pos = self.robot.body_xyz

        # view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
        #     cameraTargetPosition=base_pos,
        #     distance=self._cam_dist,
        #     yaw=self._cam_yaw,
        #     pitch=self._cam_pitch,
        #     roll=0,
        #     upAxisIndex=2)
        view_matrix = self._p.computeViewMatrix(
            cameraEyePosition=[-0.80, 0.0, 0.65],
            cameraTargetPosition=[-0.42, 0.0, 0.3],
            cameraUpVector=[0, 0, 1])
        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=60, aspect=float(self._render_width) / self._render_height,
            nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(
            width=self._render_width, height=self._render_height, viewMatrix=view_matrix,
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

    def configure_bullet_client(self):
        if self.physicsClientId < 0:
            self.ownsPhysicsClient = True
            if self.isRender:
                self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            else:
                self._p = bullet_client.BulletClient()
            self.physicsClientId = self._p._client
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
            self._p.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw - 30, self._cam_pitch+10, [0, 0, 0.3])
            self._p.setGravity(0, 0, -self._gravity)
            self._p.setDefaultContactERP(0.9)
            self._p.setPhysicsEngineParameter(fixedTimeStep=self._timestep * self._frame_skip,
                                              numSolverIterations=self._num_solver_iterations,
                                              numSubSteps=self._frame_skip)
            self._p.setRealTimeSimulation(self._use_real_time_simulation)

    def task_reset(self):
        # method to override, purposed to task specific reset
        #   e.g., object random spawn
        raise NotImplementedError

    def _step_callback(self):
        # method to override, purposed to some task-specific computations
        #   e.g., update goals
        raise NotImplementedError

    def _get_obs(self):
        # method to override, purposed to configure task-specific parameters,
        #   e.g., goal generations
        raise NotImplementedError

    def _compute_reward(self, achieved_goal, desired_goal):
        # method to override, purposed to compute goal-conditioned task-specific reward
        raise NotImplementedError
