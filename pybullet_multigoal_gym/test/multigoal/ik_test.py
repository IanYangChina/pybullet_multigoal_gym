import os
import pybullet as p
import time
import math
from datetime import datetime
import pybullet_data
import numpy as np
cwd = os.getcwd()


def get_state(p, bodyIndex, kuka_joint_index):
    kuka_joint_pos = []
    kuka_joint_vel = []
    for i in kuka_joint_index:
        x, vx, _, _ = p.getJointState(bodyIndex, i)
        kuka_joint_pos.append(x)
        kuka_joint_vel.append(vx)

    return kuka_joint_pos, kuka_joint_vel


clid = p.connect(p.SHARED_MEMORY)
if clid < 0:
    p.connect(p.GUI)
    # p.connect(p.SHARED_MEMORY_GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
# p.resetDebugVisualizerCamera(0.9, -100, -30, [0, 0, 0.4])
p.resetDebugVisualizerCamera(0.9, -100, -10, [0, 0, 0.0])
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# model_path = '/home/xintong/Documents/PyProjects/pybullet_multigoal_gym/pybullet_multigoal_gym/assets/robots/kuka/iiwa14_robotiq85.urdf'
model_path = os.path.join(cwd, '..', '..', 'assets', 'robots', 'kuka', 'iiwa14_parallel_jaw.urdf')

kukaId = p.loadURDF(model_path, [0, 0, 0])
p.loadURDF(os.path.join(cwd, '..', '..', 'assets', 'objects', 'table.urdf'),
           [-0.4, 0.0, 0.08])
p.loadURDF(os.path.join(cwd, '..', '..', 'assets', 'objects', 'block.urdf'),
           [-0.415, 0.0, 0.18])
p.resetBasePositionAndOrientation(kukaId, [0, 0, 0], [0, 0, 0, 1])
jointIndex = [1, 2, 3, 4, 5, 6, 7]
kukaEndEffectorIndex = 8
numJoints = len(jointIndex)
# if (numJoints != 7):
#     exit()

for j in range(p.getNumJoints(kukaId)):
    joint_info = p.getJointInfo(kukaId, j)
    print(joint_info[1], joint_info[12])

# lower limits for null space
ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
# upper limits for null space
ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
# joint ranges for null space
jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
# restposes for null space
rp = [0, -0.52563, 0, 2.09435, 0, -0.495188, 0]
# joint damping coefficents
jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

for i in range(numJoints):
    p.resetJointState(kukaId, jointIndex[i], rp[i])

"""
12 iiwa_gripper_finger1_joint
14 iiwa_gripper_finger2_joint
16 iiwa_gripper_finger1_inner_knuckle_joint
17 iiwa_gripper_finger1_finger_tip_joint
19 iiwa_gripper_finger2_inner_knuckle_joint
20 iiwa_gripper_finger2_finger_tip_joint
"""
robotiq_gripper_joint_index = [12, 14, 16, 17, 19, 20]
robotiq_gripper_ctrl_multiplier = np.array([1.0, 1.0, 1.0, -1.0, 1.0, -1.0])

gripper_joint_index = [12, 14]
gripper_ctrl_multiplier = np.array([1, 1])

for i in range(len(gripper_joint_index)):
    p.resetJointState(kukaId, gripper_joint_index[i], 0.02 * gripper_ctrl_multiplier[i])

p.setGravity(0, 0, -9.8)
p.setRealTimeSimulation(0)
# one action/simulation_step takes 0.002*20=0.04 seconds in real time
p.setPhysicsEngineParameter(fixedTimeStep=0.002 * 20, numSolverIterations=5, numSubSteps=20)
# trailDuration is duration (in seconds) after debug lines will be removed automatically

i = 0
mp = 1
g = 0.02
start_time = time.process_time()
# 25 simulation_steps = 1 seconds
z = 0.17
while z < 0.8:
    time.sleep(0.05)
    p.stepSimulation()

    # grip_ctrl_bound = 1.0
    # normalized action
    # a = np.random.uniform(-grip_ctrl_bound, grip_ctrl_bound)
    # de-normalized gripper ctrl signal
    # g = (a+grip_ctrl_bound) * 0.4
    z += 0.001 * i
    pos = [-0.4, 0.0, z]
    state, _ = get_state(p, kukaId, [1, 2, 3, 4, 5, 6, 7, 8])
    (x, y, z), (a, b, c, d), _, _, _, _ = p.getLinkState(kukaId, 8)
    i += 1
    # end effector points down, not up (in case useOrientation==1)
    orn = p.getQuaternionFromEuler([0, -math.pi, 0])

    jointPoses = p.calculateInverseKinematics(kukaId, kukaEndEffectorIndex,
                                              pos, orn, ll, ul, jr, rp)

    p.setJointMotorControlArray(bodyUniqueId=kukaId,
                                jointIndices=[1, 2, 3, 4, 5, 6, 7],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=jointPoses[:7],
                                targetVelocities=[0, 0, 0, 0, 0, 0, 0],
                                forces=[500, 500, 500, 500, 500, 500, 500],
                                positionGains=[0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
                                velocityGains=[1, 1, 1, 1, 1, 1, 1])

    # for parallel jaw
    p.setJointMotorControlArray(bodyUniqueId=kukaId,
                                jointIndices=gripper_joint_index,
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=g*gripper_ctrl_multiplier,
                                targetVelocities=np.zeros((2,)),
                                forces=np.ones((2,)) * 500,
                                positionGains=np.ones((2,)) * 0.03,
                                velocityGains=np.ones((2,)))

    # for robotiq gripper
    # p.setJointMotorControlArray(bodyUniqueId=kukaId,
    #                             jointIndices=robotiq_gripper_joint_index,
    #                             controlMode=p.POSITION_CONTROL,
    #                             targetPositions=g*robotiq_gripper_ctrl_multiplier,
    #                             targetVelocities=np.zeros((6,)),
    #                             forces=np.ones((6,))*500,
    #                             positionGains=np.ones((6,))*0.03,
    #                             velocityGains=np.ones((6,)))
