import os
import numpy as np
import quaternion as quat
from pybullet_multigoal_gym.envs.base_envs.base_env import BaseBulletMGEnv
from pybullet_multigoal_gym.robots.kuka import Kuka


class KukaBulletPrimitiveEnv(BaseBulletMGEnv):
    """
    Base class for the shape assemble manipulation tasks with a Kuka iiwa 14 robot
    """

    def __init__(self, render=True, binary_reward=True,
                 image_observation=False, goal_image=False, depth_image=False, pcd=False,
                 visualize_target=True, regenerate_goal_when_step=False,
                 manipulated_object_keys=None, goal_object_key='cube', orientation_informed_goal=False,
                 camera_setup=None, observation_cam_id=0, goal_cam_id=0,
                 gripper_type='parallel_jaw', obj_range=0.15, target_range=0.15,
                 end_effector_start_on_table=False,
                 distance_threshold=0.05, grasping=False,
                 primitive=None):
        if manipulated_object_keys is None:
            manipulated_object_keys = ['cube', 'slot']
        self.binary_reward = binary_reward
        self.image_observation = image_observation
        self.goal_image = goal_image
        self.render_pcd = pcd
        if depth_image:
            self.render_mode = 'rgbd_array'
        else:
            self.render_mode = 'rgb_array'
        self.visualize_target = visualize_target
        self.observation_cam_id = observation_cam_id
        self.goal_cam_id = goal_cam_id
        self.regenerate_goal_when_step = regenerate_goal_when_step

        self.distance_threshold = distance_threshold
        self.grasping = grasping
        self.obj_range = obj_range
        self.target_range = target_range

        self.object_assets_path = os.path.join(os.path.dirname(__file__),
                                               "..", "..", "assets", "objects", "assembling_shape")
        self.objects_urdf_loaded = False
        self.object_bodies = {
            'workspace': None,
            'cube': None,
            'slot': None,
            'rectangle': None,
            'target': None
        }
        self.object_initial_pos = {
            'workspace': [-0.55, 0.0, 0.01, 0.0, 0.0, 0.0, 1.0],
            'cube': [-0.60, 0.0, 0.035, 0.0, 0.0, 0.0, 1.0],
            'slot': [-0.50, 0.0, 0.035, 0.0, 0.0, 0.0, 1.0],
            'rectangle': [-0.50, 0.0, 0.035, 0.0, 0.0, 0.0, 1.0],
            'target': [-0.55, 0.0, 0.035, 0.0, 0.0, 0.0, 1.0]
        }
        self.manipulated_object_keys = manipulated_object_keys
        self.goal_object_key = goal_object_key
        self.orientation_informed_goal = orientation_informed_goal

        self.desired_goal = None
        self.desired_goal_image = None

        robot = Kuka(gripper_type=gripper_type, grasping=grasping,
                     primitive=primitive, workspace_range={"upper_xy": (-0.4, 0.2),
                                                           "lower_xy": (-0.7, -0.2)},
                     end_effector_start_on_table=end_effector_start_on_table, table_surface_z=0.04,
                     obj_range=self.obj_range, target_range=self.target_range)

        BaseBulletMGEnv.__init__(self, robot=robot, render=render,
                                 image_observation=image_observation, goal_image=goal_image,
                                 camera_setup=camera_setup,
                                 seed=0, timestep=0.002, frame_skip=20)

    def _task_reset(self, test=False):
        if not self.objects_urdf_loaded:
            # don't reload object urdf
            self.objects_urdf_loaded = True
            self.object_bodies['workspace'] = self._p.loadURDF(
                os.path.join(self.object_assets_path, "workspace.urdf"),
                basePosition=self.object_initial_pos['workspace'][:3],
                baseOrientation=self.object_initial_pos['workspace'][3:])

            for object_key in self.manipulated_object_keys:
                self.object_bodies[object_key] = self._p.loadURDF(
                    os.path.join(self.object_assets_path, object_key+".urdf"),
                    basePosition=self.object_initial_pos[object_key][:3],
                    baseOrientation=self.object_initial_pos[object_key][3:])

            self.object_bodies[self.goal_object_key+'_target'] = self._p.loadURDF(
                os.path.join(self.object_assets_path, self.goal_object_key+"_target.urdf"),
                basePosition=self.object_initial_pos['target'][:3],
                baseOrientation=self.object_initial_pos['target'][3:])

            if not self.visualize_target:
                self.set_object_pose(self.object_bodies[self.goal_object_key+'_target'],
                                     [0.0, 0.0, -3.0],
                                     self.object_initial_pos['target'][3:])

        # randomize object positions
        object_poses = []
        object_quats = []
        for object_key in self.manipulated_object_keys:
            done = False
            while not done:
                new_object_xy = self.np_random.uniform(self.robot.object_bound_lower[:-1],
                                                       self.robot.object_bound_upper[:-1])
                object_not_overlap = []
                for pos in object_poses + [self.robot.end_effector_tip_initial_position]:
                    object_not_overlap.append((np.linalg.norm(new_object_xy - pos[:-1]) > 0.06))
                if all(object_not_overlap):
                    object_poses.append(np.concatenate((new_object_xy.copy(), [0.035])))
                    done = True

            orientation_euler = quat.as_euler_angles(quat.as_quat_array([1., 0., 0., 0.]))
            orientation_euler[-1] = self.np_random.uniform(-1.0, 1.0) * np.pi
            orientation_quat_new = quat.as_float_array(quat.from_euler_angles(orientation_euler))
            orientation_quat_new = np.concatenate([orientation_quat_new[1:], [orientation_quat_new[0]]], axis=-1)
            object_quats.append(orientation_quat_new.copy())

            self.set_object_pose(self.object_bodies[object_key],
                                 object_poses[-1],
                                 orientation_quat_new)

        # generate goals & images
        self._generate_goal()
        if self.goal_image:
            self._generate_goal_image()

    def _generate_goal(self):
        raise NotImplementedError()

    def _generate_goal_image(self):
        raise NotImplementedError()

    def _step_callback(self):
        pass

    def _get_obs(self):
        # re-generate goals & images
        if self.regenerate_goal_when_step:
            self._generate_goal()
            if self.goal_image:
                self._generate_goal_image()

        assert self.desired_goal is not None

        state = []
        achieved_goal = []

        for object_key in self.manipulated_object_keys:
            # object state: (x, y, z), (a, b, c, w)
            obj_xyz, (a, b, c, w) = self._p.getBasePositionAndOrientation(self.object_bodies[object_key])
            obj_euler = quat.as_euler_angles(quat.as_quat_array([w, a, b, c]))
            state.append(obj_xyz)
            state.append(obj_euler)
            if object_key == self.goal_object_key:
                achieved_goal.append(obj_xyz)
                if self.orientation_informed_goal:
                    achieved_goal.append(obj_euler)

        state = np.concatenate(state)
        achieved_goal = np.concatenate(achieved_goal)
        assert achieved_goal.shape == self.desired_goal.shape

        obs_dict = {
            'observation': state.copy(),
            'policy_state': state.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.desired_goal.copy(),
        }

        if self.image_observation:
            self.robot.set_kuka_joint_state(self.robot.kuka_away_pose)

            observation = self.render(mode=self.render_mode, camera_id=self.observation_cam_id)
            obs_dict['observation'] = observation.copy()
            obs_dict.update({'state': state.copy()})

            if self.goal_image:
                achieved_goal_img = self.render(mode=self.render_mode, camera_id=self.goal_cam_id)
                obs_dict.update({
                    'achieved_goal_img': achieved_goal_img.copy(),
                    'desired_goal_img': self.desired_goal_image.copy(),
                })

            if self.render_pcd:
                pcd = self.render(mode='pcd', camera_id=self.observation_cam_id)
                obs_dict.update({'pcd': pcd.copy()})

            self.robot.set_kuka_joint_state(self.robot.kuka_rest_pose)

        return obs_dict

    def _compute_reward(self, achieved_goal, desired_goal):
        assert achieved_goal.shape == desired_goal.shape
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        not_achieved = (d > self.distance_threshold)
        if self.binary_reward:
            return -not_achieved.astype(np.float32), ~not_achieved
        else:
            return -d, ~not_achieved

    def set_object_pose(self, body_id, position, orientation=None):
        if orientation is None:
            orientation = self.object_initial_pos['workspace'][3:]
        self._p.resetBasePositionAndOrientation(body_id, position, orientation)
