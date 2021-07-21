from pybullet_multigoal_gym.robots.robot_bases import URDFBasedRobot


class Chest(URDFBasedRobot):
    def __init__(self, base_position, door='revolving', rest_door_state=0.0):
        self.door = door
        if self.door == 'revolving':
            assert 0.0 <= rest_door_state <= 1.57
            model_urdf = 'objects/chest_revolving_door.urdf'
            self.door_joint_name = 'chest_back_wall_bearing_joint'
            self.chest_door_opened_state = 1.57
        elif self.door == 'up_sliding':
            assert 0.0 <= rest_door_state <= 0.1
            model_urdf = 'objects/chest_up_sliding_door.urdf'
            self.door_joint_name = 'chest_back_wall_door_joint'
            self.chest_door_opened_state = 0.1
        elif self.door == 'front_sliding':
            assert 0.0 <= rest_door_state <= 0.12
            model_urdf = 'objects/chest_front_sliding_door.urdf'
            self.door_joint_name = 'chest_back_wall_door_joint'
            self.chest_door_opened_state = 0.12
        else:
            raise ValueError('invalid door %s' % door, 'only support \'revolving\' and \'up_ or front_sliding\'.')
        URDFBasedRobot.__init__(self,
                                model_urdf=model_urdf,
                                robot_name='chest',
                                base_position=base_position,
                                self_collision=True)
        self.body_id = None
        self.joint_id = None
        self.rest_joint_state = rest_door_state
        self.orientation = [0, 0, 0, 1]
        self.keypoint_part_name = [
            'chest_door_left_keypoint',
            'chest_door_right_keypoint',
            'chest_door_handle_keypoint'
        ]

    def robot_specific_reset(self, bullet_client):
        if self.body_id is None:
            self.body_id = self.jdict[self.door_joint_name].bodies[self.jdict[self.door_joint_name].bodyIndex]
        if self.joint_id is None:
            self.joint_id = self.jdict[self.door_joint_name].jointIndex
        self.jdict[self.door_joint_name].reset_position(self.rest_joint_state, 0.0)

    def calc_robot_state(self):
        door_joint_pos, door_joint_vel = self.jdict[self.door_joint_name].get_state()
        keypoint_state = []
        for keypoint in self.keypoint_part_name:
            xyz = self.parts[keypoint].get_position()
            vel_xyz = self.parts[keypoint].get_linear_velocity()
            keypoint_state = keypoint_state + [xyz, vel_xyz]
            if self.door == 'revolving':
                rpy = self.parts[keypoint].get_orientation_eular()
                vel_rpy = self.parts[keypoint].get_angular_velocity()
                keypoint_state = keypoint_state + [rpy, vel_rpy]
        return door_joint_pos, door_joint_vel, keypoint_state

    def apply_action(self, action, bullet_client):
        bullet_client.setJointMotorControlArray(bodyUniqueId=self.body_id,
                                                jointIndices=[self.joint_id],
                                                controlMode=bullet_client.POSITION_CONTROL,
                                                targetPositions=action,
                                                targetVelocities=[0],
                                                forces=[500],
                                                positionGains=[0.03],
                                                velocityGains=[1])

    def set_base_pos(self, bullet_client, position, orientation=None):
        if orientation is None:
            orientation = self.orientation
        bullet_client.resetBasePositionAndOrientation(self.body_id, position, orientation)

    def get_base_pos(self, bullet_client):
        xyz, quat = bullet_client.getBasePositionAndOrientation(self.body_id)
        return xyz, quat

    def get_part_xyz(self, part_name):
        return self.parts[part_name].get_position()