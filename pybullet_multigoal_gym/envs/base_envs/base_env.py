import gym
import numpy as np
import pybullet
import warnings
from gym.utils import seeding
from gym import spaces
from pybullet_utils import bullet_client


class BaseBulletMGEnv(gym.Env):
    """
    Base class for non-hierarchical multi-goal RL task, based on PyBullet and Gym.
    """

    def __init__(self, robot, chest=False,
                 render=False, image_observation=False, goal_image=False, camera_setup=None,
                 seed=0, gravity=9.81, timestep=0.002, frame_skip=20):
        self.robot = robot

        self.isRender = render
        self.image_observation = image_observation
        self.goal_image = goal_image

        self.seed(seed=seed)
        self._gravity = gravity
        self._timestep = timestep
        self._frame_skip = frame_skip

        # bullet client setup
        self.physicsClientId = -1
        self.ownsPhysicsClient = 0
        self._p = None
        # debug viewer camera
        self._debug_cam_dist = 1.2
        self._debug_cam_yaw = -90
        self._debug_cam_pitch = -30
        self._num_solver_iterations = 5
        self._use_real_time_simulation = False
        # configure client
        self._configure_bullet_client()
        self.robot._p = self._p
        self.robot.reset()
        if chest:
            self.chest_robot._p = self._p
            self.chest_robot.reset()
        # observation camera setup
        if camera_setup is None:
            # default camera
            self.camera_setup = [{
                'cameraEyePosition': [-1.0, 0.25, 0.6],
                'cameraTargetPosition': [-0.6, 0.05, 0.2],
                'cameraUpVector': [0, 0, 1],
                'render_width': 128,
                'render_height': 128
            }]
        else:
            # user cameras
            self.camera_setup = camera_setup
        # append the top-down view camera setup
        self.camera_setup.append({
            'cameraEyePosition': [-0.52, 0.0, 0.63],
            'cameraTargetPosition': [-0.52, 0.0, 0.02],
            'cameraUpVector': [1, 0, 0],
            # resolution: 0.002 meters per pixel for the 0.7x0.7m workspace in assets/objects/assembling_shape
            # 0.7 / 0.002 = 350
            'render_width': 350,
            'render_height': 350
        })
        # append the hand camera setup
        self.camera_setup.append({
            'cameraEyePosition': self.robot.parts['iiwa_hand_cam_origin'].get_position(),
            'cameraTargetPosition': self.robot.parts['iiwa_gripper_tip'].get_position(),
            'cameraUpVector': [0, 0, 1],
            'render_width': 128,
            'render_height': 128
        })
        self.camera_matrices = self._get_camera_matrix()
        self._render_width = 128
        self._render_height = 128

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        obs = self.reset()
        self.action_space = robot.action_space
        if not self.image_observation:
            self.observation_space = spaces.Dict(dict(
                state=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
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

    def reset(self, test=False):
        self.robot.reset()
        self._task_reset(test=test)
        obs = self._get_obs()
        return obs

    def step(self, action):
        self.robot.apply_action(action)
        obs = self._get_obs()
        reward, goal_achieved = self._compute_reward(obs['achieved_goal'], obs['desired_goal'])
        self._step_callback()
        info = {
            'goal_achieved': goal_achieved
        }
        return obs, reward, False, info

    def render(self, mode="human", camera_id=0):
        assert mode in ['human', 'pcd', 'rgb_array', 'depth', 'rgbd_array'], "make sure you use a supported rendering mode"
        if mode == "human":
            warnings.warn("Users should not call env.render() with mode=\"human\" with pybullet backend."
                          "Users should make the env instance with render=True if a GUI window is desired.")
            return None
        else:
            if camera_id == -1:
                self._update_hand_camera_matrix()
            (_, _, px, depth, _) = self._p.getCameraImage(
                width=self.camera_setup[camera_id]['render_width'],
                height=self.camera_setup[camera_id]['render_height'],
                viewMatrix=self.camera_matrices[camera_id]['view_matrix'],
                projectionMatrix=self.camera_matrices[camera_id]['proj_matrix'],
                renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
            )

            if mode == 'pcd':
                return self._render_pcd(depth=depth, camera_id=camera_id)
            rgb_array = px[:, :, :3]
            if mode == "rgb_array":
                return rgb_array
            # transform depth value into [0 255] as type uint8
            depth_uint8 = np.array([depth * 255]).transpose((1, 2, 0)).astype('uint8')
            if mode == 'depth':
                return depth_uint8
            if mode == 'rgbd_array':
                rgbd_array = np.concatenate((rgb_array, depth_uint8), axis=-1)
                return rgbd_array

    def _render_pcd(self, depth, camera_id=0):
        img_width = self.camera_setup[camera_id]['render_width']
        img_height = self.camera_setup[camera_id]['render_height']

        depth = np.array(depth)

        # adapted from https://stackoverflow.com/a/62247245
        stepX = 2
        stepY = 2
        points = []
        pointCloud = np.empty([np.int(img_height / stepY), np.int(img_width / stepX), 4])
        projectionMatrix = np.asarray(self.camera_matrices[camera_id]['proj_matrix']).reshape([4, 4], order='F')
        viewMatrix = np.asarray(self.camera_matrices[camera_id]['view_matrix']).reshape([4, 4], order='F')
        tran_pix_world = np.linalg.inv(np.matmul(projectionMatrix, viewMatrix))

        for h in range(0, img_height, stepY):
            for w in range(0, img_width, stepX):
                x = (2 * w - img_width) / img_width
                y = -(2 * h - img_height) / img_height
                z = 2 * depth[h, w] - 1
                pixPos = np.asarray([x, y, z, 1])
                position = np.matmul(tran_pix_world, pixPos)
                points.append(position / position[3])
                # pointCloud[np.int(h / stepY), np.int(w / stepX), :] = position / position[3]

        return np.array(points)[:, :-1]

    def close(self):
        if self.ownsPhysicsClient:
            if self.physicsClientId >= 0:
                self._p.disconnect()
        self.physicsClientId = -1

    def _configure_bullet_client(self):
        if self.physicsClientId < 0:
            self.ownsPhysicsClient = True
            if self.isRender:
                self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
                self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0, lightPosition=[0.0, 0.0, 4])
                self._p.resetDebugVisualizerCamera(self._debug_cam_dist,
                                                   self._debug_cam_yaw - 30,
                                                   self._debug_cam_pitch + 10, [0, 0, 0.3])
            else:
                self._p = bullet_client.BulletClient()
            self.physicsClientId = self._p._client
            self._p.setGravity(0, 0, -self._gravity)
            self._p.setDefaultContactERP(0.9)
            self._p.setPhysicsEngineParameter(fixedTimeStep=self._timestep * self._frame_skip,
                                              numSolverIterations=self._num_solver_iterations,
                                              numSubSteps=self._frame_skip)
            self._p.setRealTimeSimulation(self._use_real_time_simulation)

    def _get_camera_matrix(self):
        cam_matrices = []
        for cam_dict in self.camera_setup:
            view_matrix = self._p.computeViewMatrix(
                cameraEyePosition=cam_dict['cameraEyePosition'],
                cameraTargetPosition=cam_dict['cameraTargetPosition'],
                cameraUpVector=cam_dict['cameraUpVector'])
            proj_matrix = self._p.computeProjectionMatrixFOV(
                fov=60, aspect=float(cam_dict['render_width']) / cam_dict['render_height'],
                nearVal=0.1, farVal=100.0)
            cam_matrices.append({
                'view_matrix': view_matrix,
                'proj_matrix': proj_matrix,
            })
        return cam_matrices

    def _update_hand_camera_matrix(self):
        cam_target = self.robot.parts['iiwa_gripper_tip'].get_position()
        cam_target[-1] -= 0.05
        self.camera_setup[-1] = {
            'cameraEyePosition': self.robot.parts['iiwa_hand_cam_origin'].get_position(),
            'cameraTargetPosition': self.robot.parts['iiwa_gripper_tip'].get_position(),
            'cameraUpVector': [0, 0, 1],
            'render_width': 128,
            'render_height': 128
        }
        view_matrix = self._p.computeViewMatrix(
            cameraEyePosition=self.camera_setup[-1]['cameraEyePosition'],
            cameraTargetPosition=self.camera_setup[-1]['cameraTargetPosition'],
            cameraUpVector=self.camera_setup[-1]['cameraUpVector'])
        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=60, aspect=float(self.camera_setup[-1]['render_width']) / self.camera_setup[-1]['render_height'],
            nearVal=0.1, farVal=100.0)
        self.camera_matrices[-1] = {'view_matrix': view_matrix,
                                    'proj_matrix': proj_matrix}

    def _task_reset(self, test=False):
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
