import pybullet as p
import time
import math
from datetime import datetime
import pybullet_data
import numpy as np

clid = p.connect(p.SHARED_MEMORY)
if clid < 0:
    p.connect(p.GUI)
    # p.connect(p.SHARED_MEMORY_GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
# p.resetDebugVisualizerCamera(0.9, -100, -30, [0, 0, 0.4])
p.resetDebugVisualizerCamera(0.9, -100, -10, [0, 0, 0.0])
p.setAdditionalSearchPath(pybullet_data.getDataPath())

model_path = '/home/xintong/Documents/PyProjects/pybullet_multigoal_gym/pybullet_multigoal_gym/assets/robots/kuka/iiwa14_robotiq85.urdf'

kukaId = p.loadURDF(model_path, [0, 0, 0])
p.loadURDF("random_urdfs/001/001.urdf", [0, 0.3, 0.3])
p.resetBasePositionAndOrientation(kukaId, [0, 0, 0], [0, 0, 0, 1])
jointIndex = [1, 2, 3, 4, 5, 6, 7]
kukaEndEffectorIndex = 8
numJoints = len(jointIndex)
# if (numJoints != 7):
#     exit()

# lower limits for null space
ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
# upper limits for null space
ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
# joint ranges for null space
jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
# restposes for null space
rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
# joint damping coefficents
jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

for i in range(numJoints):
    p.resetJointState(kukaId, jointIndex[i], rp[i])

"""
10 iiwa_gripper_finger1_joint
12 iiwa_gripper_finger2_joint
14 iiwa_gripper_finger1_inner_knuckle_joint
15 iiwa_gripper_finger1_finger_tip_joint
16 iiwa_gripper_finger2_inner_knuckle_joint
17 iiwa_gripper_finger2_finger_tip_joint
"""

p.setGravity(0, 0, 0)
t = 0.
prevPose = [0, 0, 0]
prevPose1 = [0, 0, 0]
hasPrevPose = 0
useNullSpace = 1
useOrientation = 1
ikSolver = 0
p.setRealTimeSimulation(0)
# one action/simulation_step takes 0.002*20=0.04 seconds in real time
p.setPhysicsEngineParameter(fixedTimeStep=0.002 * 20, numSolverIterations=5, numSubSteps=20)
# trailDuration is duration (in seconds) after debug lines will be removed automatically
# use 0 for no-removal
trailDuration = 15

i = 0
mp = 1
g = 0
start_time = time.process_time()
# 25 simulation_steps = 1 seconds
while i < (25*50):
    i += 1
    t = t + 0.04
    p.stepSimulation()

    g += 0.08 * mp
    if i % 10 == 0:
        mp = -mp
    # grip_ctrl_bound = 1.0
    # normalized action
    # a = np.random.uniform(-grip_ctrl_bound, grip_ctrl_bound)
    # de-normalized gripper ctrl signal
    # g = (a+grip_ctrl_bound) * 0.4

    for _ in range(1):
        pos = [-0.4, 0.2 * math.cos(t), 0. + 0.2 * math.sin(t)]
        # end effector points down, not up (in case useOrientation==1)
        orn = p.getQuaternionFromEuler([0, -math.pi, 0])

        jointPoses = p.calculateInverseKinematics(kukaId, kukaEndEffectorIndex,
                                                  pos, orn, ll, ul, jr, rp)

        # p.setJointMotorControlArray(bodyUniqueId=kukaId,
        #                             jointIndices=[1, 2, 3, 4, 5, 6, 7],
        #                             controlMode=p.POSITION_CONTROL,
        #                             targetPositions=jointPoses[:7],
        #                             targetVelocities=[0, 0, 0, 0, 0, 0, 0],
        #                             forces=[500, 500, 500, 500, 500, 500, 500],
        #                             positionGains=[0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
        #                             velocityGains=[1, 1, 1, 1, 1, 1, 1])
        #
        # p.setJointMotorControlArray(bodyUniqueId=kukaId,
        #                             jointIndices=[10, 12, 14, 15, 17, 18],
        #                             controlMode=p.POSITION_CONTROL,
        #                             targetPositions=np.array([g, g, g, -g, g, -g]),
        #                             targetVelocities=np.zeros((6,)),
        #                             forces=np.ones((6,))*500,
        #                             positionGains=np.ones((6,))*0.03,
        #                             velocityGains=np.ones((6,)))

    ls = p.getLinkState(kukaId, kukaEndEffectorIndex)
    if (hasPrevPose):
        p.addUserDebugLine(prevPose, pos, [0, 0, 0.3], 1, trailDuration)
        p.addUserDebugLine(prevPose1, ls[4], [1, 0, 0], 1, trailDuration)
    prevPose = pos
    prevPose1 = ls[4]
    hasPrevPose = 1
print('%0.2f seconds' % (time.process_time() - start_time))
p.disconnect()
