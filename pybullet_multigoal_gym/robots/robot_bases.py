import numpy as np
import os


class XmlBasedRobot(object):
    """Base class for .xml based agents."""

    def __init__(self, bullet_client, robot_name, self_collision=True):
        self._p = bullet_client
        self.robot_name = robot_name
        self.objects = None
        self.parts = {}
        self.jdict = {}
        self.ordered_joint_names = []
        self.self_collision = self_collision

    def addToScene(self, bodies):
        if np.isscalar(bodies):  # streamline the case where bodies is actually just one body
            bodies = [bodies]
        for i in range(len(bodies)):
            for j in range(self._p.getNumJoints(bodies[i])):
                joint_info = self._p.getJointInfo(bodies[i], j)
                joint_name = joint_info[1].decode("utf8")
                self.ordered_joint_names.append(joint_name)
                part_name = joint_info[12].decode("utf8")
                self.parts[part_name] = BodyPart(self._p, part_name, bodies, i, j)
                self.jdict[joint_name] = Joint(self._p, bodies, i, j, joint_info)


class URDFBasedRobot(XmlBasedRobot):
    """Base class for URDF .xml based robots."""

    def __init__(self, bullet_client, model_urdf, robot_name, base_position=None,
                 base_orientation=None, fixed_base=False, self_collision=False):
        XmlBasedRobot.__init__(self,
                               bullet_client=bullet_client,
                               robot_name=robot_name,
                               self_collision=self_collision)
        if base_position is None:
            base_position = [0, 0, 0]
        if base_orientation is None:
            base_orientation = [0, 0, 0, 1]
        self.model_urdf = model_urdf
        self.base_position = base_position
        self.base_orientation = base_orientation
        self.fixed_base = fixed_base
        self.robot_urdf_loaded = False
        self.target_keys = ['target_red', 'target_blue', 'target_green', 'target_purple']
        self.target_bodies = {
            'target_red': None,
            'target_blue': None,
            'target_green': None,
            'target_purple': None
        }
        self.target_initial_pos = {
            'target_red': [-0.54, 0.0, 0.035, 0.0, 0.0, 0.0, 1.0],
            'target_blue': [-0.54, 0.0, 0.035, 0.0, 0.0, 0.0, 1.0],
            'target_green': [-0.54, 0.0, 0.035, 0.0, 0.0, 0.0, 1.0],
            'target_purple': [-0.54, 0.0, 0.035, 0.0, 0.0, 0.0, 1.0]
        }

    def reset(self):
        # load urdf if it's the first time that reset() gets called
        if not self.robot_urdf_loaded:
            full_path = os.path.join(os.path.dirname(__file__), "..", "assets", self.model_urdf)
            self.robot_urdf_loaded = True
            if self.self_collision:
                self.addToScene(self._p.loadURDF(full_path,
                                                 basePosition=self.base_position,
                                                 baseOrientation=self.base_orientation,
                                                 useFixedBase=self.fixed_base,
                                                 flags=self._p.URDF_USE_SELF_COLLISION))
            else:
                self.addToScene(self._p.loadURDF(full_path,
                                                 basePosition=self.base_position,
                                                 baseOrientation=self.base_orientation,
                                                 useFixedBase=self.fixed_base))
            # for target_name in self.target_keys:
            #     self.target_bodies[target_name] = self._p.loadURDF(
            #         os.path.join(os.path.dirname(__file__), "..", "assets", "robots", target_name + ".urdf"),
            #         basePosition=self.target_initial_pos[target_name][:3],
            #         baseOrientation=self.target_initial_pos[target_name][3:])
        # reset robot-specific configuration
        self.robot_specific_reset()

    def robot_specific_reset(self):
        # method to override, purposed to reset robot-specific configuration
        raise NotImplementedError

    def calc_robot_state(self):
        # method to override, purposed to obtain robot-specific states
        raise NotImplementedError

    def apply_action(self, action):
        # method to override, purposed to apply robot-specific actions
        raise NotImplementedError


class BodyPart:
    def __init__(self, bullet_client, body_name, bodies, bodyIndex, bodyPartIndex):
        self.bodies = bodies
        self._p = bullet_client
        self.body_name = body_name
        self.bodyIndex = bodyIndex
        self.bodyPartIndex = bodyPartIndex
        self.initialPosition = self.get_position()
        self.initialOrientation = self.get_orientation()

    def get_pose(self):
        (x, y, z), (a, b, c, w), _, _, _, _ = self._p.getLinkState(self.bodies[self.bodyIndex], self.bodyPartIndex)
        return np.array([x, y, z, a, b, c, w])

    def get_orientation(self):
        # return orientation in quaternion
        return self.get_pose()[3:]

    def get_orientation_eular(self):
        return self._p.getEulerFromQuaternion(self.get_orientation())

    def get_position(self):
        return self.get_pose()[:3]

    def get_velocities(self):
        _, _, _, _, _, _, (vx, vy, vz), (vr, vp, vya) = self._p.getLinkState(self.bodies[self.bodyIndex],
                                                                             self.bodyPartIndex,
                                                                             computeLinkVelocity=1)
        return np.array([vx, vy, vz, vr, vp, vya])

    def get_linear_velocity(self):
        return self.get_velocities()[:3]

    def get_angular_velocity(self):
        return self.get_velocities()[3:]

    def reset_position(self, position):
        self._p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex],
                                                position,
                                                self.get_orientation())

    def reset_orientation(self, orientation):
        self._p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex],
                                                self.get_position(),
                                                orientation)

    def reset_velocity(self, linearVelocity=None, angularVelocity=None):
        if angularVelocity is None:
            angularVelocity = [0, 0, 0]
        if linearVelocity is None:
            linearVelocity = [0, 0, 0]
        self._p.resetBaseVelocity(self.bodies[self.bodyIndex], linearVelocity, angularVelocity)

    def reset_pose(self, position, orientation):
        self._p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex], position, orientation)

    def contact_list(self):
        return self._p.getContactPoints(self.bodies[self.bodyIndex], -1, self.bodyPartIndex, -1)


class Joint:
    def __init__(self, bullet_client, bodies, bodyIndex, jointIndex, joint_info):
        self.bodies = bodies
        self._p = bullet_client
        self.bodyIndex = bodyIndex
        self.jointIndex = jointIndex
        self.joint_name = joint_info[1].decode("utf8")
        self.jointType = joint_info[2]
        self.lowerLimit = joint_info[8]
        self.upperLimit = joint_info[9]
        self.jointHasLimits = self.lowerLimit < self.upperLimit
        self.jointMaxVelocity = joint_info[11]

    def set_state(self, x, vx):
        self._p.resetJointState(self.bodies[self.bodyIndex], self.jointIndex, x, vx)

    def current_relative_position(self):
        pos, vel, _ = self.get_state()
        if self.jointHasLimits:
            pos_mid = 0.5 * (self.lowerLimit + self.upperLimit)
            pos = 2 * (pos - pos_mid) / (self.upperLimit - self.lowerLimit)

        if self.jointMaxVelocity > 0:
            vel /= self.jointMaxVelocity
        elif self.jointType == 0:  # JOINT_REVOLUTE_TYPE
            vel *= 0.1
        else:
            vel *= 0.5
        return (
            pos,
            vel
        )

    def get_state(self):
        # getJointState output:
        #   jointPosition: The position value of this joint (as jonit angle/position or joint orientation quaternion)
        #   jointVelocity: The velocity value of this joint
        #   jointReactionForces
        #   appliedJointMotorTorque
        x, vx, fx, _ = self._p.getJointState(self.bodies[self.bodyIndex], self.jointIndex)
        return x, vx, fx

    def get_position(self):
        x, _, _ = self.get_state()
        return x

    def get_orientation(self):
        _, r, _ = self.get_state()
        return r

    def get_velocity(self):
        _, vx, _ = self.get_state()
        return vx

    def get_force(self):
        _, _, fx = self.get_state()
        return fx

    def set_position(self, position):
        self._p.setJointMotorControl2(self.bodies[self.bodyIndex], self.jointIndex, self._p.POSITION_CONTROL,
                                      targetPosition=position)

    def set_velocity(self, velocity):
        self._p.setJointMotorControl2(self.bodies[self.bodyIndex], self.jointIndex, self._p.VELOCITY_CONTROL,
                                      targetVelocity=velocity)

    def set_torque(self, torque):
        self._p.setJointMotorControl2(bodyIndex=self.bodies[self.bodyIndex], jointIndex=self.jointIndex,
                                      controlMode=self._p.TORQUE_CONTROL,
                                      force=torque)  # , positionGain=0.1, velocityGain=0.1)

    def reset_position(self, position, velocity):
        self._p.resetJointState(self.bodies[self.bodyIndex], self.jointIndex, targetValue=position,
                                targetVelocity=velocity)
        self.disable_motor()

    def disable_motor(self):
        self._p.setJointMotorControl2(self.bodies[self.bodyIndex], self.jointIndex,
                                      controlMode=self._p.POSITION_CONTROL, targetPosition=0, targetVelocity=0,
                                      positionGain=0.1, velocityGain=0.1, force=0)
