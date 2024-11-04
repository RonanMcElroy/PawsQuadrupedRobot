import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt
import can
import can_utils
import os


class PawsRobot:
    """""
    This class...
    """""
    def __init__(self, L1=0.06, L2=0.15, L3=0.2):

        self.L1 = L1
        self.L2 = L2
        self.L3 = L3

        # Define Joint Indexes
        self.FR_HIP_ABDUCTOR_ADDUCTOR_JOINT = 0
        self.FR_HIP_FLEXOR_EXTENDOR_JOINT = 1
        self.FR_KNEE_JOINT = 2
        self.FR_FOOT_JOINT = 3
        self.FL_HIP_ABDUCTOR_ADDUCTOR_JOINT = 4
        self.FL_HIP_FLEXOR_EXTENDOR_JOINT = 5
        self.FL_KNEE_JOINT = 6
        self.FL_FOOT_JOINT = 7
        self.RR_HIP_ABDUCTOR_ADDUCTOR_JOINT = 8
        self.RR_HIP_FLEXOR_EXTENDOR_JOINT = 9
        self.RR_KNEE_JOINT = 10
        self.RR_FOOT_JOINT = 11
        self.RL_HIP_ABDUCTOR_ADDUCTOR_JOINT = 12
        self.RL_HIP_FLEXOR_EXTENDOR_JOINT = 13
        self.RL_KNEE_JOINT = 14
        self.RL_FOOT_JOINT = 15

        # Define Link Indexes
        self.FR_HIP_ABDUCTOR_ADDUCTOR_LINK = 0
        self.FR_HIP_FLEXOR_EXTENDOR_LINK = 1
        self.FR_KNEE_LINK = 2
        self.FR_FOOT_LINK = 3
        self.FL_HIP_ABDUCTOR_ADDUCTOR_LINK = 4
        self.FL_HIP_FLEXOR_EXTENDOR_LINK = 5
        self.FL_KNEE_LINK = 6
        self.FL_FOOT_LINK = 7
        self.RR_HIP_ABDUCTOR_ADDUCTOR_LINK = 8
        self.RR_HIP_FLEXOR_EXTENDOR_LINK = 9
        self.RR_KNEE_LINK = 10
        self.RR_FOOT_LINK = 11
        self.RL_HIP_ABDUCTOR_ADDUCTOR_LINK = 12
        self.RL_HIP_FLEXOR_EXTENDOR_LINK = 13
        self.RL_KNEE_LINK = 14
        self.RL_FOOT_LINK = 15


    def createTrotTrajectory(self, stepLength=0.1, stepHeight=0.05, T=2, t_0=0, sampleRate=500, dutyFactor = 0.5,
                               x_0=0, h_0=-0.25, backSwingCoef=0.2, L1=0.06):
        """""
        This function...
        Args:
            name (type): Description.
        Returns:
            type: Description.
        """""
        self.stepLength = stepLength
        self.stepHeight = stepHeight
        self.T = T

        T_swing = self.T * (1 - dutyFactor)
        T_stance = self.T * dutyFactor

        t_1 = t_0 + 0.25*T_swing
        t_2 = t_0 + 0.75*T_swing
        t_w = t_0 + 0.5*T_swing
        samplesPerSwing = int(sampleRate * T_swing)
        samplesPerStance = int(sampleRate * T_stance)

        t_backSwing = np.linspace(t_0, t_1, int(0.25 * samplesPerSwing))
        t_stride = np.linspace(t_1, t_2, int(0.5 * samplesPerSwing))
        t_forwardSwing = np.linspace(t_2, T_swing, int(0.25 * samplesPerSwing))
        t_lift = np.linspace(t_0, t_w, int(0.5 * samplesPerSwing))
        t_place = np.linspace(t_w, T_swing, int(0.5 * samplesPerSwing))
        t_stance = np.linspace(t_0, T_stance, samplesPerStance)

        x_1 = x_0 - backSwingCoef * self.stepLength
        x_2 = x_0 + (1 + backSwingCoef) * self.stepLength
        x_3 = x_0 + self.stepLength
        h = self.stepHeight

        T_x1 = t_1 - t_0
        T_x2 = t_2 - t_1
        T_x3 = T_swing - t_2
        T_h1 = t_w - t_0
        T_h2 = T_swing - t_w

        xBackSwing = x_0 + (20*x_1 - 20*x_0)/(2*T_x1**3)*np.power(t_backSwing - t_0, 3) \
                     + (30*x_0 - 30*x_1)/(2*T_x1**4)*np.power(t_backSwing - t_0, 4) \
                     + (12*x_1 - 12*x_0)/(2*T_x1**5)*np.power(t_backSwing - t_0, 5)

        xStride = (x_2-x_1)*((t_stride-t_1)/T_x2 - (1/(2*np.pi))*np.sin((2*np.pi*(t_stride-t_1))/T_x2)) + x_1

        xForwardSwing = x_2 + (20*x_3 - 20*x_2)/(2*T_x3**3)*np.power(t_forwardSwing - t_2, 3) \
                        + (30*x_2 - 30*x_3)/(2*T_x3**4)*np.power(t_forwardSwing - t_2, 4) \
                        + (12*x_3 - 12*x_2)/(2*T_x3**5)*np.power(t_forwardSwing - t_2, 5)

        xStance = x_3 + (20*x_0 - 20*x_3)/(2*(T_stance - t_0)**3)*np.power(t_stance - t_0, 3) \
                  + (30*x_3 - 30*x_0)/(2*(T_stance - t_0)**4)*np.power(t_stance - t_0, 4) \
                  + (12*x_0 - 12*x_3)/(2*(T_stance - t_0)**5)*np.power(t_stance - t_0, 5)

        zLift = h_0 + (20*h)/(2*T_h1**3)*np.power(t_lift - t_0, 3) \
                - (30*h)/(2*T_h1**4)*np.power(t_lift - t_0, 4) \
                + (12*h)/(2*T_h1**5)*np.power(t_lift - t_0, 5)

        zPlace = h_0 + h - (20*h)/(2*T_h2**3)*np.power(t_place - t_w, 3) \
                 + (30*h)/(2*T_h2**4)*np.power(t_place - t_w, 4) \
                 - (12*h)/(2*T_h2**5)*np.power(t_place - t_w, 5)

        zStance = h_0 + np.zeros(samplesPerStance)

        ySwing = self.L1 + np.zeros(samplesPerSwing)
        yStance = self.L1 + np.zeros(samplesPerStance)

        self.x_tragectory = np.concatenate((xBackSwing, xStride, xForwardSwing, xStance), axis=0)
        self.z_tragectory = np.concatenate((zLift, zPlace, zStance), axis=0)
        self.y_tragectory = np.concatenate((ySwing, yStance), axis=0)

        self.footPositionTrajectory = np.zeros([sampleRate*self.T, 3])
        for i in range(0, sampleRate*self.T):
            self.footPositionTrajectory[i] = np.array([self.x_tragectory[i],self.y_tragectory[i],self.z_tragectory[i]])


    def createWalkTrajectory(self, stepLength=0.1, stepHeight=0.05, T=4, t_0=0, sampleRate=500, dutyFactor=0.75,
                               x_0=0, h_0=-0.25, backSwingCoef=0.2, L1=0.06):
        """""
        This function...
        Args:
            name (type): Description.
        Returns:
            type: Description.
        """""
        self.stepLength = stepLength
        self.stepHeight = stepHeight
        self.T = T

        T_swing = self.T * (1 - dutyFactor)
        T_stance = self.T * dutyFactor

        t_1 = t_0 + 0.25 * T_swing
        t_2 = t_0 + 0.75 * T_swing
        t_w = t_0 + 0.5 * T_swing
        samplesPerSwing = int(sampleRate * T_swing)
        samplesPerStance = int(sampleRate * T_stance)

        t_backSwing = np.linspace(t_0, t_1, int(0.25 * samplesPerSwing))
        t_stride = np.linspace(t_1, t_2, int(0.5 * samplesPerSwing))
        t_forwardSwing = np.linspace(t_2, T_swing, int(0.25 * samplesPerSwing))
        t_lift = np.linspace(t_0, t_w, int(0.5 * samplesPerSwing))
        t_place = np.linspace(t_w, T_swing, int(0.5 * samplesPerSwing))
        t_stance = np.linspace(t_0, T_swing, samplesPerSwing)

        x_1 = x_0 - backSwingCoef * self.stepLength
        x_2 = x_0 + (1 + backSwingCoef) * self.stepLength
        x_3 = x_0 + self.stepLength
        h = self.stepHeight

        T_x1 = t_1 - t_0
        T_x2 = t_2 - t_1
        T_x3 = T_swing - t_2
        T_h1 = t_w - t_0
        T_h2 = T_swing - t_w

        xBackSwing = x_0 + (20*x_1 - 20*x_0)/(2*T_x1**3)*np.power(t_backSwing-t_0, 3) \
                     + (30*x_0 - 30*x_1)/(2*T_x1**4)*np.power(t_backSwing - t_0, 4) \
                     + (12*x_1 - 12*x_0)/(2*T_x1**5)*np.power(t_backSwing - t_0, 5)

        xStride = (x_2 - x_1)*((t_stride - t_1)/T_x2 - (1/(2*np.pi))*np.sin((2*np.pi*(t_stride - t_1))/T_x2)) + x_1

        xForwardSwing = x_2 + (20*x_3 - 20*x_2)/(2*T_x3**3)*np.power(t_forwardSwing - t_2, 3) \
                        + (30*x_2 - 30*x_3)/(2*T_x3**4)*np.power(t_forwardSwing - t_2, 4) \
                        + (12*x_3 - 12*x_2)/(2*T_x3**5)*np.power(t_forwardSwing - t_2, 5)

        xStance1 = x_3 + (20*(2*x_3/3) - 20*x_3)/(2*(T_swing - t_0)**3)*np.power(t_stance - t_0, 3) \
                  + (30*x_3 - 30*(2*x_3/3))/(2*(T_swing - t_0)**4)*np.power(t_stance - t_0, 4) \
                  + (12*(2*x_3/3) - 12*x_3)/(2*(T_swing - t_0)**5)*np.power(t_stance - t_0, 5)

        xStance2 = (2*x_3/3) + (20*(x_3/3) - 20*(2*x_3/3))/(2*(T_swing - t_0)**3)*np.power(t_stance - t_0, 3) \
                   + (30*(2*x_3/3) - 30*(x_3/3))/(2*(T_swing - t_0)**4)*np.power(t_stance - t_0, 4) \
                   + (12*(x_3/3) - 12*(2*x_3/3))/(2*(T_swing - t_0)**5)*np.power(t_stance - t_0, 5)

        xStance3 = (x_3/3) + (20*x_0 - 20*(x_3/3))/(2*(T_swing - t_0)**3)*np.power(t_stance - t_0, 3) \
                   + (30*(x_3/3) - 30*x_0)/(2*(T_swing - t_0)**4)*np.power(t_stance - t_0, 4) \
                   + (12*x_0 - 12*(x_3/3))/(2*(T_swing - t_0)**5)*np.power(t_stance - t_0, 5)

        zLift = h_0 + (20*h)/(2*T_h1**3)*np.power(t_lift - t_0, 3) \
                - (30*h)/(2*T_h1**4)*np.power(t_lift - t_0, 4) \
                + (12*h)/(2*T_h1**5)*np.power(t_lift - t_0, 5)

        zPlace = h_0 + h - (20*h)/(2*T_h2**3)*np.power(t_place - t_w, 3) \
                 + (30*h)/(2*T_h2**4)*np.power(t_place - t_w, 4) \
                 - (12*h)/(2*T_h2**5)*np.power(t_place - t_w, 5)

        zStance = h_0 + np.zeros(samplesPerStance)

        ySwing = self.L1 + np.zeros(samplesPerSwing)
        yStance = self.L1 + np.zeros(samplesPerStance)

        self.x_tragectory = np.concatenate((xBackSwing, xStride, xForwardSwing, xStance1, xStance2, xStance3), axis=0)
        self.z_tragectory = np.concatenate((zLift, zPlace, zStance), axis=0)
        self.y_tragectory = np.concatenate((ySwing, yStance), axis=0)

        self.footPositionTrajectory = np.zeros([sampleRate * self.T, 3])
        for i in range(0, sampleRate * self.T):
            self.footPositionTrajectory[i] = np.array([self.x_tragectory[i],self.y_tragectory[i],self.z_tragectory[i]])


    def plotTrajectory(self, axis):
        """""
        This function...
        Args:
            name (type): Description.
        Returns:
            type: Description.
        """""
        if axis == "xz":
            fig = plt.figure(1)
            plt.plot(self.x_tragectory, self.z_tragectory)
            plt.axis('equal')
            plt.xlabel('x (m)')
            plt.ylabel('z (m)')
            plt.title("Step Length {} m, Step Height {} m, Step Period {} s".format(self.stepLength,
                                                                                      self.stepHeight,
                                                                                      self.T))
            plt.grid()
            plt.show()

        elif axis == "xy":
            fig = plt.figure(1)
            plt.plot(self.x_tragectory, self.y_tragectory)
            plt.axis('equal')
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            plt.title("Step Length {} m, Step Height {} m, Step Period {} s".format(self.stepLength,
                                                                                      self.stepHeight,
                                                                                      self.T))
            plt.grid()
            plt.show()

        elif axis == "yz":
            fig = plt.figure(1)
            plt.plot(self.y_tragectory, self.z_tragectory)
            plt.axis('equal')
            plt.xlabel('y (m)')
            plt.ylabel('z (m)')
            plt.title("Step Length {} m, Step Height {} m, Step Period {} s".format(self.stepLength,
                                                                                      self.stepHeight,
                                                                                      self.T))
            plt.grid()
            plt.show()

        else:
            print("Cannot Plot Requested Trajectory.")


    def calculateIK(self, targetFootPosition, kneeLagging):

        x = targetFootPosition[0]
        y = targetFootPosition[1]
        z = targetFootPosition[2]
        z_dash = -np.sqrt(z ** 2 + y ** 2 - self.L1 ** 2)

        alpha = np.arccos(abs(y) / np.sqrt(z ** 2 + y ** 2))
        beta = np.arccos(self.L1 / np.sqrt(z ** 2 + y ** 2))
        phi = np.arccos(abs(x) / np.sqrt(x ** 2 + z_dash ** 2))
        psi = np.arccos((x ** 2 + z_dash ** 2 + self.L2 ** 2 - self.L3 ** 2) / (2 * self.L2 * np.sqrt(x ** 2 + z_dash ** 2)))

        if y >= 0:
            theta_1 = beta - alpha
        else:
            theta_1 = alpha + beta - np.pi

        if kneeLagging:
            if x >= 0:
                theta_2 = np.pi / 2 - psi - phi
            else:
                theta_2 = -np.pi / 2 - psi + phi

            theta_3 = np.pi - np.arccos((self.L2 ** 2 + self.L3 ** 2 - x ** 2 - z_dash ** 2) / (2 * self.L2 * self.L3))
        else:
            if x >= 0:
                theta_2 = np.pi / 2 + psi - phi
            else:
                theta_2 = -np.pi / 2 + psi + phi

            theta_3 = - np.pi + np.arccos((self.L2 ** 2 + self.L3 ** 2 - x ** 2 - z_dash ** 2) / (2 * self.L2 * self.L3))

        targetJointPositions = np.array([theta_1, theta_2, theta_3])

        return targetJointPositions


    def simulateTrotGate(self):

        p.connect(p.GUI)
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        startPos = [0, 0, 0.39]
        planeId = p.loadURDF("plane.urdf")
        simPaws = p.loadURDF(r'PawsURDF/urdf/Paws.urdf', startPos, useFixedBase=0)

        samplesPerPeriod = self.footPositionTrajectory.shape[0]
        FR_targetFootPosition = self.footPositionTrajectory[0]
        FR_targetJointPositions = paws.calculateIK(FR_targetFootPosition, kneeLagging=True)
        RL_targetJointPositions = paws.calculateIK(FR_targetFootPosition, kneeLagging=True)

        FL_targetFootPosition = self.footPositionTrajectory[int(samplesPerPeriod / 2)]
        FL_targetJointPositions = paws.calculateIK(FL_targetFootPosition, kneeLagging=True)
        RR_targetJointPositions = paws.calculateIK(FL_targetFootPosition, kneeLagging=True)

        p.setJointMotorControl2(simPaws, self.FL_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=-FL_targetJointPositions[0])
        p.setJointMotorControl2(simPaws, self.FL_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=-FL_targetJointPositions[1])
        p.setJointMotorControl2(simPaws, self.FL_KNEE_JOINT, p.POSITION_CONTROL,
                                targetPosition=FL_targetJointPositions[2])

        p.setJointMotorControl2(simPaws, self.FR_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=FR_targetJointPositions[0])
        p.setJointMotorControl2(simPaws, self.FR_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=FR_targetJointPositions[1])
        p.setJointMotorControl2(simPaws, self.FR_KNEE_JOINT, p.POSITION_CONTROL,
                                targetPosition=FR_targetJointPositions[2])

        p.setJointMotorControl2(simPaws, self.RL_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=RL_targetJointPositions[0])
        p.setJointMotorControl2(simPaws, self.RL_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=-RL_targetJointPositions[1])
        p.setJointMotorControl2(simPaws, self.RL_KNEE_JOINT, p.POSITION_CONTROL,
                                targetPosition=RL_targetJointPositions[2])

        p.setJointMotorControl2(simPaws, self.RR_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=-RR_targetJointPositions[0])
        p.setJointMotorControl2(simPaws, self.RR_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=RR_targetJointPositions[1])
        p.setJointMotorControl2(simPaws, self.RR_KNEE_JOINT, p.POSITION_CONTROL,
                                targetPosition=RR_targetJointPositions[2])

        for j in range(0, 100):
            p.stepSimulation()
            time.sleep(0.03)

        i = 0
        while True:

            FR_targetFootPosition = self.footPositionTrajectory[i]
            FR_targetJointPositions = paws.calculateIK(FR_targetFootPosition, kneeLagging=True)
            RL_targetJointPositions = paws.calculateIK(FR_targetFootPosition, kneeLagging=True)

            if i >= int(samplesPerPeriod / 2):
                j = i + int(samplesPerPeriod / 2) - samplesPerPeriod
            else:
                j = i + int(samplesPerPeriod / 2)

            FL_targetFootPosition = self.footPositionTrajectory[j]
            FL_targetJointPositions = paws.calculateIK(FL_targetFootPosition, kneeLagging=True)
            RR_targetJointPositions = paws.calculateIK(FL_targetFootPosition, kneeLagging=True)

            p.setJointMotorControl2(simPaws, self.FR_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=FR_targetJointPositions[0])
            p.setJointMotorControl2(simPaws, self.FR_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=FR_targetJointPositions[1])
            p.setJointMotorControl2(simPaws, self.FR_KNEE_JOINT, p.POSITION_CONTROL,
                                    targetPosition=FR_targetJointPositions[2])

            p.setJointMotorControl2(simPaws, self.RR_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=-RR_targetJointPositions[0])
            p.setJointMotorControl2(simPaws, self.RR_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=RR_targetJointPositions[1])
            p.setJointMotorControl2(simPaws, self.RR_KNEE_JOINT, p.POSITION_CONTROL,
                                    targetPosition=RR_targetJointPositions[2])

            p.setJointMotorControl2(simPaws, self.FL_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=-FL_targetJointPositions[0])
            p.setJointMotorControl2(simPaws, self.FL_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=-FL_targetJointPositions[1])
            p.setJointMotorControl2(simPaws, self.FL_KNEE_JOINT, p.POSITION_CONTROL,
                                    targetPosition=FL_targetJointPositions[2])

            p.setJointMotorControl2(simPaws, self.RL_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=RL_targetJointPositions[0])
            p.setJointMotorControl2(simPaws, self.RL_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=-RL_targetJointPositions[1])
            p.setJointMotorControl2(simPaws, self.RL_KNEE_JOINT, p.POSITION_CONTROL,
                                    targetPosition=RL_targetJointPositions[2])

            i = i+1
            if i == samplesPerPeriod:
                i = 0

            p.stepSimulation()
            # time.sleep(0.002)


    def simulateWalkGate(self):

        p.connect(p.GUI)
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        startPos = [0, 0, 0.39]
        planeId = p.loadURDF("plane.urdf")
        simPaws = p.loadURDF(r'PawsURDF/urdf/Paws.urdf', startPos, useFixedBase=0)

        samplesPerPeriod = self.footPositionTrajectory.shape[0]
        FR_targetFootPosition = self.footPositionTrajectory[0]
        FR_targetJointPositions = paws.calculateIK(FR_targetFootPosition, kneeLagging=True)
        RL_targetFootPosition = self.footPositionTrajectory[int(3*samplesPerPeriod/4)]
        RL_targetJointPositions = paws.calculateIK(RL_targetFootPosition, kneeLagging=True)
        FL_targetFootPosition = self.footPositionTrajectory[int(samplesPerPeriod/2)]
        FL_targetJointPositions = paws.calculateIK(FL_targetFootPosition, kneeLagging=True)
        RR_targetFootPosition = self.footPositionTrajectory[int(samplesPerPeriod/4)]
        RR_targetJointPositions = paws.calculateIK(RR_targetFootPosition, kneeLagging=True)

        p.setJointMotorControl2(simPaws, self.FL_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=-FL_targetJointPositions[0])
        p.setJointMotorControl2(simPaws, self.FL_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=-FL_targetJointPositions[1])
        p.setJointMotorControl2(simPaws, self.FL_KNEE_JOINT, p.POSITION_CONTROL,
                                targetPosition=FL_targetJointPositions[2])

        p.setJointMotorControl2(simPaws, self.FR_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=FR_targetJointPositions[0])
        p.setJointMotorControl2(simPaws, self.FR_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=FR_targetJointPositions[1])
        p.setJointMotorControl2(simPaws, self.FR_KNEE_JOINT, p.POSITION_CONTROL,
                                targetPosition=FR_targetJointPositions[2])

        p.setJointMotorControl2(simPaws, self.RL_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=RL_targetJointPositions[0])
        p.setJointMotorControl2(simPaws, self.RL_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=-RL_targetJointPositions[1])
        p.setJointMotorControl2(simPaws, self.RL_KNEE_JOINT, p.POSITION_CONTROL,
                                targetPosition=RL_targetJointPositions[2])

        p.setJointMotorControl2(simPaws, self.RR_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=-RR_targetJointPositions[0])
        p.setJointMotorControl2(simPaws, self.RR_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=RR_targetJointPositions[1])
        p.setJointMotorControl2(simPaws, self.RR_KNEE_JOINT, p.POSITION_CONTROL,
                                targetPosition=RR_targetJointPositions[2])

        for j in range(0, 100):
            p.stepSimulation()
            time.sleep(0.03)

        i = 0
        while True:

            FR_targetFootPosition = self.footPositionTrajectory[i]
            FR_targetJointPositions = paws.calculateIK(FR_targetFootPosition, kneeLagging=True)

            if i >= int(samplesPerPeriod / 4):
                j = i + int(3 * samplesPerPeriod / 4) - samplesPerPeriod
            else:
                j = i + int(3 * samplesPerPeriod / 4)

            RL_targetFootPosition = self.footPositionTrajectory[j]
            RL_targetJointPositions = paws.calculateIK(RL_targetFootPosition, kneeLagging=True)

            if i >= int(samplesPerPeriod / 2):
                j = i + int(samplesPerPeriod / 2) - samplesPerPeriod
            else:
                j = i + int(samplesPerPeriod / 2)

            FL_targetFootPosition = self.footPositionTrajectory[j]
            FL_targetJointPositions = paws.calculateIK(FL_targetFootPosition, kneeLagging=True)

            if i >= int(3 * samplesPerPeriod / 4):
                j = i + int(samplesPerPeriod / 4) - samplesPerPeriod
            else:
                j = i + int(samplesPerPeriod / 4)

            RR_targetFootPosition = self.footPositionTrajectory[j]
            RR_targetJointPositions = paws.calculateIK(RR_targetFootPosition, kneeLagging=True)


            p.setJointMotorControl2(simPaws, self.FR_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=FR_targetJointPositions[0])
            p.setJointMotorControl2(simPaws, self.FR_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=FR_targetJointPositions[1])
            p.setJointMotorControl2(simPaws, self.FR_KNEE_JOINT, p.POSITION_CONTROL,
                                    targetPosition=FR_targetJointPositions[2])

            p.setJointMotorControl2(simPaws, self.RR_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=-RR_targetJointPositions[0])
            p.setJointMotorControl2(simPaws, self.RR_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=RR_targetJointPositions[1])
            p.setJointMotorControl2(simPaws, self.RR_KNEE_JOINT, p.POSITION_CONTROL,
                                    targetPosition=RR_targetJointPositions[2])

            p.setJointMotorControl2(simPaws, self.FL_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=-FL_targetJointPositions[0])
            p.setJointMotorControl2(simPaws, self.FL_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=-FL_targetJointPositions[1])
            p.setJointMotorControl2(simPaws, self.FL_KNEE_JOINT, p.POSITION_CONTROL,
                                    targetPosition=FL_targetJointPositions[2])

            p.setJointMotorControl2(simPaws, self.RL_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=RL_targetJointPositions[0])
            p.setJointMotorControl2(simPaws, self.RL_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=-RL_targetJointPositions[1])
            p.setJointMotorControl2(simPaws, self.RL_KNEE_JOINT, p.POSITION_CONTROL,
                                    targetPosition=RL_targetJointPositions[2])

            i = i+1
            if i == samplesPerPeriod:
                i = 0

            p.stepSimulation()
            # time.sleep(0.002)


    def runTrotGate(self):

        os.system('sudo ifconfig can0 down')
        os.system('sudo ip link set can0 type can bitrate 1000000')
        os.system('sudo ifconfig can0 txqueuelen 100000')
        os.system('sudo ifconfig can0 up')

        bus = can.Bus(interface='socketcan', channel='can0', bitrate=1000000)

        samplesPerPeriod = self.footPositionTrajectory.shape[0]
        FR_targetFootPosition = self.footPositionTrajectory[0]
        FR_targetJointPositions = paws.calculateIK(FR_targetFootPosition, kneeLagging=True)
        RL_targetJointPositions = paws.calculateIK(FR_targetFootPosition, kneeLagging=True)

        FL_targetFootPosition = self.footPositionTrajectory[int(samplesPerPeriod / 2)]
        FL_targetJointPositions = paws.calculateIK(FL_targetFootPosition, kneeLagging=True)
        RR_targetJointPositions = paws.calculateIK(FL_targetFootPosition, kneeLagging=True)

        can_utils.genCanMsg(self.FL_HIP_ABDUCTOR_ADDUCTOR_JOINT, position=-FL_targetJointPositions[0])
        can_utils.genCanMsg(self.FL_HIP_FLEXOR_EXTENDOR_JOINT, position=-FL_targetJointPositions[1])
        can_utils.genCanMsg(self.FL_KNEE_JOINT, position=FL_targetJointPositions[2])

        can_utils.genCanMsg(self.FR_HIP_ABDUCTOR_ADDUCTOR_JOINT, position=FR_targetJointPositions[0])
        can_utils.genCanMsg(self.FR_HIP_FLEXOR_EXTENDOR_JOINT, position=FR_targetJointPositions[1])
        can_utils.genCanMsg(self.FR_KNEE_JOINT, position=FR_targetJointPositions[2])

        can_utils.genCanMsg(self.RL_HIP_ABDUCTOR_ADDUCTOR_JOINT, position=RL_targetJointPositions[0])
        can_utils.genCanMsg(self.RL_HIP_FLEXOR_EXTENDOR_JOINT, position=-RL_targetJointPositions[1])
        can_utils.genCanMsg(self.RL_KNEE_JOINT, position=RL_targetJointPositions[2])

        can_utils.genCanMsg(self.RR_HIP_ABDUCTOR_ADDUCTOR_JOINT, position=-RR_targetJointPositions[0])
        can_utils.genCanMsg(self.RR_HIP_FLEXOR_EXTENDOR_JOINT, position=RR_targetJointPositions[1])
        can_utils.genCanMsg(self.RR_KNEE_JOINT, position=RR_targetJointPositions[2])

        time.sleep(3)

        i = 0
        while True:

            FR_targetFootPosition = self.footPositionTrajectory[i]
            FR_targetJointPositions = paws.calculateIK(FR_targetFootPosition, kneeLagging=True)
            RL_targetJointPositions = paws.calculateIK(FR_targetFootPosition, kneeLagging=True)

            if i >= int(samplesPerPeriod / 2):
                j = i + int(samplesPerPeriod / 2) - samplesPerPeriod
            else:
                j = i + int(samplesPerPeriod / 2)

            FL_targetFootPosition = self.footPositionTrajectory[j]
            FL_targetJointPositions = paws.calculateIK(FL_targetFootPosition, kneeLagging=True)
            RR_targetJointPositions = paws.calculateIK(FL_targetFootPosition, kneeLagging=True)

            can_utils.genCanMsg(self.FR_HIP_ABDUCTOR_ADDUCTOR_JOINT, position=FR_targetJointPositions[0])
            can_utils.genCanMsg(self.FR_HIP_FLEXOR_EXTENDOR_JOINT, position=FR_targetJointPositions[1])
            can_utils.genCanMsg(self.FR_KNEE_JOINT, position=FR_targetJointPositions[2])

            can_utils.genCanMsg(self.RR_HIP_ABDUCTOR_ADDUCTOR_JOINT, position=-RR_targetJointPositions[0])
            can_utils.genCanMsg(self.RR_HIP_FLEXOR_EXTENDOR_JOINT, position=RR_targetJointPositions[1])
            can_utils.genCanMsg(self.RR_KNEE_JOINT, position=RR_targetJointPositions[2])

            can_utils.genCanMsg(self.FL_HIP_ABDUCTOR_ADDUCTOR_JOINT, position=-FL_targetJointPositions[0])
            can_utils.genCanMsg(self.FL_HIP_FLEXOR_EXTENDOR_JOINT, position=-FL_targetJointPositions[1])
            can_utils.genCanMsg(self.FL_KNEE_JOINT, position=FL_targetJointPositions[2])

            can_utils.genCanMsg(self.RL_HIP_ABDUCTOR_ADDUCTOR_JOINT, position=RL_targetJointPositions[0])
            can_utils.genCanMsg(self.RL_HIP_FLEXOR_EXTENDOR_JOINT, position=-RL_targetJointPositions[1])
            can_utils.genCanMsg(self.RL_KNEE_JOINT, position=RL_targetJointPositions[2])

            i = i+1
            if i == samplesPerPeriod:
                i = 0
            time.sleep(0.00001)


    def runWalkGate(self):

        os.system('sudo ifconfig can0 down')
        os.system('sudo ip link set can0 type can bitrate 1000000')
        os.system('sudo ifconfig can0 txqueuelen 100000')
        os.system('sudo ifconfig can0 up')

        bus = can.Bus(interface='socketcan', channel='can0', bitrate=1000000)

        samplesPerPeriod = self.footPositionTrajectory.shape[0]
        FR_targetFootPosition = self.footPositionTrajectory[0]
        FR_targetJointPositions = paws.calculateIK(FR_targetFootPosition, kneeLagging=True)
        RL_targetFootPosition = self.footPositionTrajectory[int(3*samplesPerPeriod/4)]
        RL_targetJointPositions = paws.calculateIK(RL_targetFootPosition, kneeLagging=True)
        FL_targetFootPosition = self.footPositionTrajectory[int(samplesPerPeriod/2)]
        FL_targetJointPositions = paws.calculateIK(FL_targetFootPosition, kneeLagging=True)
        RR_targetFootPosition = self.footPositionTrajectory[int(samplesPerPeriod/4)]
        RR_targetJointPositions = paws.calculateIK(RR_targetFootPosition, kneeLagging=True)

        can_utils.genCanMsg(self.FL_HIP_ABDUCTOR_ADDUCTOR_JOINT, position=-FL_targetJointPositions[0])
        can_utils.genCanMsg(self.FL_HIP_FLEXOR_EXTENDOR_JOINT, position=-FL_targetJointPositions[1])
        can_utils.genCanMsg(self.FL_KNEE_JOINT, position=FL_targetJointPositions[2])

        can_utils.genCanMsg(self.FR_HIP_ABDUCTOR_ADDUCTOR_JOINT, position=FR_targetJointPositions[0])
        can_utils.genCanMsg(self.FR_HIP_FLEXOR_EXTENDOR_JOINT, position=FR_targetJointPositions[1])
        can_utils.genCanMsg(self.FR_KNEE_JOINT, position=FR_targetJointPositions[2])

        can_utils.genCanMsg(self.RL_HIP_ABDUCTOR_ADDUCTOR_JOINT, position=RL_targetJointPositions[0])
        can_utils.genCanMsg(self.RL_HIP_FLEXOR_EXTENDOR_JOINT, position=-RL_targetJointPositions[1])
        can_utils.genCanMsg(self.RL_KNEE_JOINT, position=RL_targetJointPositions[2])

        can_utils.genCanMsg(self.RR_HIP_ABDUCTOR_ADDUCTOR_JOINT, position=-RR_targetJointPositions[0])
        can_utils.genCanMsg(self.RR_HIP_FLEXOR_EXTENDOR_JOINT, position=RR_targetJointPositions[1])
        can_utils.genCanMsg(self.RR_KNEE_JOINT, position=RR_targetJointPositions[2])

        time.sleep(3)

        i = 0
        while True:

            FR_targetFootPosition = self.footPositionTrajectory[i]
            FR_targetJointPositions = paws.calculateIK(FR_targetFootPosition, kneeLagging=True)

            if i >= int(samplesPerPeriod / 4):
                j = i + int(3 * samplesPerPeriod / 4) - samplesPerPeriod
            else:
                j = i + int(3 * samplesPerPeriod / 4)

            RL_targetFootPosition = self.footPositionTrajectory[j]
            RL_targetJointPositions = paws.calculateIK(RL_targetFootPosition, kneeLagging=True)

            if i >= int(samplesPerPeriod / 2):
                j = i + int(samplesPerPeriod / 2) - samplesPerPeriod
            else:
                j = i + int(samplesPerPeriod / 2)

            FL_targetFootPosition = self.footPositionTrajectory[j]
            FL_targetJointPositions = paws.calculateIK(FL_targetFootPosition, kneeLagging=True)

            if i >= int(3 * samplesPerPeriod / 4):
                j = i + int(samplesPerPeriod / 4) - samplesPerPeriod
            else:
                j = i + int(samplesPerPeriod / 4)

            RR_targetFootPosition = self.footPositionTrajectory[j]
            RR_targetJointPositions = paws.calculateIK(RR_targetFootPosition, kneeLagging=True)


            can_utils.genCanMsg(self.FR_HIP_ABDUCTOR_ADDUCTOR_JOINT, position=FR_targetJointPositions[0])
            can_utils.genCanMsg(self.FR_HIP_FLEXOR_EXTENDOR_JOINT, position=FR_targetJointPositions[1])
            can_utils.genCanMsg(self.FR_KNEE_JOINT, position=FR_targetJointPositions[2])

            can_utils.genCanMsg(self.RR_HIP_ABDUCTOR_ADDUCTOR_JOINT, position=-RR_targetJointPositions[0])
            can_utils.genCanMsg(self.RR_HIP_FLEXOR_EXTENDOR_JOINT, position=RR_targetJointPositions[1])
            can_utils.genCanMsg(self.RR_KNEE_JOINT, position=RR_targetJointPositions[2])

            can_utils.genCanMsg(self.FL_HIP_ABDUCTOR_ADDUCTOR_JOINT, position=-FL_targetJointPositions[0])
            can_utils.genCanMsg(self.FL_HIP_FLEXOR_EXTENDOR_JOINT, position=-FL_targetJointPositions[1])
            can_utils.genCanMsg(self.FL_KNEE_JOINT, position=FL_targetJointPositions[2])

            can_utils.genCanMsg(self.RL_HIP_ABDUCTOR_ADDUCTOR_JOINT, position=RL_targetJointPositions[0])
            can_utils.genCanMsg(self.RL_HIP_FLEXOR_EXTENDOR_JOINT, position=-RL_targetJointPositions[1])
            can_utils.genCanMsg(self.RL_KNEE_JOINT, position=RL_targetJointPositions[2])

            i = i+1
            if i == samplesPerPeriod:
                i = 0

            time.sleep(0.00001)


paws = PawsRobot()
#
# paws.createTrotTrajectory()
# # paws.runTrotGate()
# paws.simulateTrotGate()
# paws.plotTrajectory(axis="xz")

# paws.createWalkTrajectory()
# paws.simulateWalkGate()
# paws.plotTrajectory(axis="xz")