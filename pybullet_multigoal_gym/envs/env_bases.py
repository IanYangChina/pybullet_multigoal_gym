import gym
import numpy as np
import pybullet
from gym.utils import seeding
from gym import spaces
from pybullet_utils import bullet_client


class BaseBulletMGEnv(gym.Env):
    """
    Base class for non-hierarchical multi-goal RL task, based on PyBullet and Gym.
    """
    def __init__(self, robot,
                 render=False, image_observation=False, goal_image=False,
                 seed=0, gravity=9.81, timestep=0.002, frame_skip=20):
        self.robot = robot

        self.isRender = render
        self.image_observation = image_observation
        self.goal_image = goal_image

        self.seed(seed=seed)
        self._gravity = gravity
        self._timestep = timestep
        self._frame_skip = frame_skip

        self.physicsClientId = -1
        self.ownsPhysicsClient = 0
        self._p = None
        self._cam_dist = 1.2
        self._cam_yaw = -90
        self._cam_pitch = -30
        self._render_width = 128
        self._render_height = 128
        self._num_solver_iterations = 5
        self._use_real_time_simulation = False
        self.configure_bullet_client()

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        obs = self.reset()
        self.action_space = robot.action_space
        if not self.image_observation:
            self.observation_space = spaces.Dict(dict(
                state=spaces.Box(-np.inf, np.inf, shape=obs['state'].shape, dtype='float32'),
                policy_state=spaces.Box(-np.inf, np.inf, shape=obs['policy_state'].shape, dtype='float32'),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
                desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['desired_goal'].shape, dtype='float32'),
            ))
        elif not self.goal_image:
            self.observation_space = spaces.Dict(dict(
                observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
                state=spaces.Box(-np.inf, np.inf, shape=obs['state'].shape, dtype='float32'),
                policy_state=spaces.Box(-np.inf, np.inf, shape=obs['policy_state'].shape, dtype='float32'),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
                desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['desired_goal'].shape, dtype='float32'),
            ))
        else:
            self.observation_space = spaces.Dict(dict(
                observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
                state=spaces.Box(-np.inf, np.inf, shape=obs['state'].shape, dtype='float32'),
                policy_state=spaces.Box(-np.inf, np.inf, shape=obs['policy_state'].shape, dtype='float32'),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
                achieved_goal_img=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal_img'].shape, dtype='float32'),
                desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['desired_goal'].shape, dtype='float32'),
                desired_goal_img=spaces.Box(-np.inf, np.inf, shape=obs['desired_goal_img'].shape, dtype='float32'),
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
        obs = self._get_obs()
        reward, goal_achieved = self._compute_reward(obs['achieved_goal'], obs['desired_goal'])
        self._step_callback()
        info = {
            'goal_achieved': goal_achieved
        }
        return obs, reward, False, info

    def render(self, mode="human"):
        if mode == "human":
            self.isRender = True
        else:
            view_matrix = self._p.computeViewMatrix(
                cameraEyePosition=[-1.0, 0.25, 0.6],
                cameraTargetPosition=[-0.6, 0.05, 0.2],
                cameraUpVector=[0, 0, 1])
            proj_matrix = self._p.computeProjectionMatrixFOV(
                fov=60, aspect=float(self._render_width) / self._render_height,
                nearVal=0.1, farVal=100.0)
            (_, _, px, depth, _) = self._p.getCameraImage(
                width=self._render_width, height=self._render_height, viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
            )
            rgb_array = px[:, :, :3]
            if mode == "rgb_array":
                return rgb_array
            elif mode == "rgbd_array":
                # transform depth value into [0 255] as type uint8
                depth = np.array([depth*255]).transpose((1, 2, 0)).astype('uint8')
                rgbd_array = np.concatenate((rgb_array, depth), axis=-1)
                return rgbd_array

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
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0, lightPosition=[0.0, 0.0, 4])
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
