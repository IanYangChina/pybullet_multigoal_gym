import os
import numpy as np
import quaternion as quat
from pybullet_multigoal_gym.envs.base_envs.base_env import BaseBulletMGEnv
from pybullet_multigoal_gym.robots.kuka import Kuka


class KukaBulletShapeAssembleEnv(BaseBulletMGEnv):
    """
    Base class for the shape assemble manipulation tasks with a Kuka iiwa 14 robot
    """

    def __init__(self, render=True, binary_reward=True,
                 image_observation=False, goal_image=False, depth_image=False, pcd=False,
                 visualize_target=True,
                 camera_setup=None, observation_cam_id=0, goal_cam_id=0,
                 gripper_type='parallel_jaw', obj_range=0.15, target_range=0.15,
                 end_effector_start_on_table=False,
                 distance_threshold=0.05, grasping=False,
                 primitive=None):
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
            'target': None
        }
        self.object_initial_pos = {
            'workspace': [-0.55, 0.0, 0.01, 0.0, 0.0, 0.0, 1.0],
            'cube': [-0.60, 0.0, 0.035, 0.0, 0.0, 0.0, 1.0],
            'slot': [-0.50, 0.0, 0.035, 0.0, 0.0, 0.0, 1.0],
            'target': [-0.55, 0.0, 0.035, 0.0, 0.0, 0.0, 1.0]
        }
        self.manipulated_object_keys = ['cube', 'slot']

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
            self.object_bodies['target'] = self._p.loadURDF(
                os.path.join(self.object_assets_path, "target.urdf"),
                basePosition=self.object_initial_pos['target'][:3],
                baseOrientation=self.object_initial_pos['target'][3:])
            self.object_bodies['cube'] = self._p.loadURDF(
                os.path.join(self.object_assets_path, "cube.urdf"),
                basePosition=self.object_initial_pos['cube'][:3],
                baseOrientation=self.object_initial_pos['cube'][3:])
            self.object_bodies['slot'] = self._p.loadURDF(
                os.path.join(self.object_assets_path, "slot.urdf"),
                basePosition=self.object_initial_pos['slot'][:3],
                baseOrientation=self.object_initial_pos['slot'][3:])
            if not self.visualize_target:
                self.set_object_pose(self.object_bodies['target'],
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
        (x, y, z), (a, b, c, w), _, _, _, _ = self._p.getLinkState(self.object_bodies['slot'], 2)
        orientation_euler = quat.as_euler_angles(quat.as_quat_array([w, a, b, c]))

        self.desired_goal = np.concatenate([np.array([x, y, z]), orientation_euler], axis=-1)

        if self.visualize_target:
            self.set_object_pose(self.object_bodies['target'],
                                 self.desired_goal[:3],
                                 self.desired_goal[3:])

    def _generate_goal_image(self):
        self.robot.set_kuka_joint_state(self.robot.kuka_away_pose)

        # Push task
        original_obj_pos, original_obj_quat = self._p.getBasePositionAndOrientation(self.object_bodies['cube'])
        target_obj_pos = self.desired_goal.copy()[:3]
        target_obj_euler = self.desired_goal.copy()
        target_obj_quat = quat.as_float_array(quat.from_euler_angles(target_obj_euler))
        target_obj_quat = np.concatenate([target_obj_quat[1:], [target_obj_quat[0]]], axis=-1)
        self.set_object_pose(self.object_bodies['cube'],
                             target_obj_pos,
                             target_obj_quat)
        self.desired_goal_image = self.render(mode=self.render_mode, camera_id=self.goal_cam_id)
        self.set_object_pose(self.object_bodies['cube'],
                             original_obj_pos,
                             original_obj_quat)

        self.robot.set_kuka_joint_state(self.robot.kuka_rest_pose)

    def _step_callback(self):
        pass

    def _get_obs(self):
        # re-generate goals & images
        self._generate_goal()
        if self.goal_image:
            self._generate_goal_image()

        assert self.desired_goal is not None

        # slot state: (x, y, z), (a, b, c, w)
        slot_xyz, slot_quat = self._p.getBasePositionAndOrientation(self.object_bodies['slot'])
        slot_euler = quat.as_euler_angles(quat.as_quat_array(slot_quat))
        # cube state: (x, y, z), (a, b, c, w)
        cube_xyz, cube_quat = self._p.getBasePositionAndOrientation(self.object_bodies['cube'])
        cube_euler = quat.as_euler_angles(quat.as_quat_array(cube_quat))

        achieved_goal = np.concatenate([cube_xyz, cube_euler])

        state = np.concatenate((cube_xyz, cube_euler, slot_xyz, slot_euler))
        policy_state = np.concatenate((cube_xyz, cube_euler, slot_xyz, slot_euler))

        obs_dict = {
            'observation': state.copy(),
            'policy_state': policy_state.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.desired_goal.copy(),
        }

        if self.image_observation:
            self.robot.set_kuka_joint_state(self.robot.kuka_away_pose)

            observation = self.render(mode=self.render_mode, camera_id=self.observation_cam_id)
            obs_dict['observation'] = observation.copy()
            obs_dict.update({'state': state.copy()})

            if self.goal_image:
                obs_dict.update({
                    'achieved_goal_img': observation.copy(),
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
