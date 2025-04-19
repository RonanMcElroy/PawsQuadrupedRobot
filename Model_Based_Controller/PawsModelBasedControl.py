import pybullet as p                # Physics engine
import pybullet_data                # Containes plane.urdf
import time                         # Used for delays and measuring elapsed time
import numpy as np                  # Used for efficient numerical computation
import matplotlib.pyplot as plt     # Used for visualisation
import can                          # Used for sending and receiving
import os                           # Access to operating system for setting up CAN bus communications
import random                       # Used for random number generation
import pandas as pd                 # Used for logging
import math                         # Used for trajectory generation


class Paws:
    """""
    This class contains all the functions to simulate the model-based locomotion controller in PyBullet and run it on 
    the physical PAWS robot.
    """""
    def __init__(self, L1=0.06, L2=0.15, L3=0.2):
        """""
        This method initialises the attributes corresponding to the geometry of the PAWS robot (link lengths) and the 
        coordinate frames attached to each joint (link and joints have the same index i.e. FR_KNEE_LNIK = FR_KNEE_JOINT)
        Args:
            L1 (float): Length of link1 in meters, measured as 0.06 on PAWS.
            L2 (float): Length of link2 (the upper leg limb) in meters, measured as 0.15 on PAWS.
            L3 (float): Length of link3 (the lower leg limb) in meters. measured as 0.235 m on the real PAWS robot but 
                        0.2 m in simulation.
        Returns:
            None
        """""

        # Define Link lengths
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3

        # Define Joint Indexes (Same as Link Indexes)
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

        # Define Motor Numbers
        self.FR_HIP_ABDUCTOR_ADDUCTOR_MOTOR = 1
        self.FR_HIP_FLEXOR_EXTENDOR_MOTOR = 2
        self.FR_KNEE_MOTOR = 3
        self.FL_HIP_ABDUCTOR_ADDUCTOR_MOTOR = 4
        self.FL_HIP_FLEXOR_EXTENDOR_MOTOR = 5
        self.FL_KNEE_MOTOR = 6
        self.RL_HIP_ABDUCTOR_ADDUCTOR_MOTOR = 7
        self.RL_HIP_FLEXOR_EXTENDOR_MOTOR = 8
        self.RL_KNEE_MOTOR = 9
        self.RR_HIP_ABDUCTOR_ADDUCTOR_MOTOR = 10
        self.RR_HIP_FLEXOR_EXTENDOR_MOTOR = 11
        self.RR_KNEE_MOTOR = 12


    def simulateTrotGate(self, forwardSwingLength=0.08, latSwingLength=0.0, swingHeight=0.04,
                         latStanceLength=0.0, T_swing=0.2, h_0=-0.28, sampleRate=240, backSwingCoef=0.2,
                         hipOffset=0.06, terrain="FLAT", hieght_mean=0.1, height_std=0.011, xy_scale=0.05,
                         stairStepLen=0.2, stairStepWidth=2, stairStepHeight=0.03, fixBase=False, logData=False):
        """""
        This method simulates the trot gait in PyBullet. Arguments determine the initial step parameters, which can 
        then be changed during the runtime of the simulation using sliders. Also accepts arguments for the terrain type.
        Args:
            forwardSwingLength (float): Length of swing in x-direction relative to hip (negative goes backwards).
            latSwingLength (float): Length of swing in y-direction relative to hip (negative goes inwards).
            swingHeight (float): Maximum height of swing phase relative to the stance height. 
            latStanceLength (float): Maximum y-lean during stance phase. Recommended 0 m.
            T_swing (float): Swing period.
            h_0 (float): z-position of feet relative to hip during the stance phase (i.e. - body height).
            sampleRate (float): Sample rate, recommended 240 Hz.
            backSwingCoef (float): back/forward swing coefficient.
            hipOffset (float): 0 m for feet directly below motors, L1 m for legs straight (recommended).
            terrain (str): type of terrain, is asserted in ["FLAT", "ROUGH", "STAIRS"].
            hieght_mean (float): mean height of terrain if terrain==ROUGH.
            height_std (float): standard deviation of terrain if terrain==ROUGH.
            xy_scale (float): how big each square is if terrain==ROUGH.
            stairStepLen (float): length in x-direction of each step if terrain==STAIRS.
            stairStepWidth (float): width in y-direction of each step if terrain==STAIRS.
            stairStepHeight(float): height in z-direction of each step if terrain==STAIRS.
            fixBase (bool): if True then suspend in air, else free base.
            logData(bool): if True then log simulation data to CSV file.
        Returns:
            None: Simulation continues on infinite loop until program is ended.
        """""

        assert terrain in ["FLAT", "RANDOM_ROUGH", "STAIRS"]  # Ensure valid terrain

        # Create terrain
        if terrain == "FLAT":
            self._createSimEnvironment(terrain)
        elif terrain == "RANDOM_ROUGH":
            self._createSimEnvironment(terrain, hieght_mean, height_std, xy_scale)
        elif terrain == "STAIRS":
            self._createSimEnvironment(terrain, stepLen=stairStepLen, stepWidth=stairStepWidth,
                                       stepHeight=stairStepHeight)

        filename = r"C:\Users\Ronan\PawsQuadrupedRobot\log.csv"  # File directory for logs

        startPos = [0, 0, 0.49]
        planeId = p.loadURDF("plane.urdf")
        self.simPaws = p.loadURDF(r'PawsURDF/urdf/Paws.urdf', startPos, useFixedBase=fixBase)

        swingX_0 = -forwardSwingLength / 2  # Initial position of swing feet in x-direction
        stanceX_0 = forwardSwingLength / 2  # Initial position of stance feet in x-direction
        swing1Y_0 = 0  # Initial position of swing foot 1 in y-direction
        swing2Y_0 = 0  # Initial position of swing foot 2 in y-direction
        stance1Y_0 = 0  # Initial position of stance foot 1 in y-direction
        stance2Y_0 = 0  # Initial position of stance foot 2 in y-direction

        # Create sliders to change step parameters during runtime
        forwardSwingLengthIn = p.addUserDebugParameter("Forward Step Length", -0.2, 0.2, forwardSwingLength)
        swingHeightIn = p.addUserDebugParameter("Step Height", 0.03, 0.15, swingHeight)
        latSwingLengthIn = p.addUserDebugParameter("Lateral Step Length", -0.1, 0.1, latSwingLength)
        latStanceLengthIn = p.addUserDebugParameter("Sideways Lean Length", 0, 0.1, latStanceLength)
        T_swing_In = p.addUserDebugParameter("Swing Period", 0.1, 5, T_swing)
        h_0_In = p.addUserDebugParameter("Body Height", -0.33, -0.13, h_0)

        # Define virtual PD controller gains
        kp = 0.15
        kd = 0.55

        # Generate foot trajectories for the first swing phase
        feetTrajectories = self._genTrotStep(forwardSwingLength, latSwingLength, swingHeight, latStanceLength,
                                                  T_swing, h_0, sampleRate, swingX_0, stanceX_0, swing1Y_0, swing2Y_0,
                                                  stance1Y_0, stance2Y_0, backSwingCoef, hipOffset)

        self.sampleRate = sampleRate  # Update class attribute
        samplesPerSwing = feetTrajectories[0].shape[0]  # Determine the number of samples per swing

        # Calculate initial joint positions for all legs using IK
        FR_targetFootPosition = feetTrajectories[0][0]  # Initially swing
        FR_targetJointPositions = self._calculateIK(FR_targetFootPosition, kneeLagging=True)
        RL_targetFootPosition = feetTrajectories[1][0]  # Initially swing
        RL_targetJointPositions = self._calculateIK(RL_targetFootPosition, kneeLagging=True)
        RR_targetFootPosition = feetTrajectories[2][0]  # Initially stance
        RR_targetJointPositions = self._calculateIK(RR_targetFootPosition, kneeLagging=True)
        FL_targetFootPosition = feetTrajectories[3][0]  # Initially stance
        FL_targetJointPositions = self._calculateIK(FL_targetFootPosition, kneeLagging=True)

        ### Send commands to all motors ###
        p.setJointMotorControl2(self.simPaws, self.FR_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=FR_targetJointPositions[0])
        p.setJointMotorControl2(self.simPaws, self.FR_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=FR_targetJointPositions[1])
        p.setJointMotorControl2(self.simPaws, self.FR_KNEE_JOINT, p.POSITION_CONTROL,
                                targetPosition=FR_targetJointPositions[2])

        p.setJointMotorControl2(self.simPaws, self.RL_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=RL_targetJointPositions[0])
        p.setJointMotorControl2(self.simPaws, self.RL_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=-RL_targetJointPositions[1])
        p.setJointMotorControl2(self.simPaws, self.RL_KNEE_JOINT, p.POSITION_CONTROL,
                                targetPosition=RL_targetJointPositions[2])

        p.setJointMotorControl2(self.simPaws, self.RR_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=-RR_targetJointPositions[0])
        p.setJointMotorControl2(self.simPaws, self.RR_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=RR_targetJointPositions[1])
        p.setJointMotorControl2(self.simPaws, self.RR_KNEE_JOINT, p.POSITION_CONTROL,
                                targetPosition=RR_targetJointPositions[2])

        p.setJointMotorControl2(self.simPaws, self.FL_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=-FL_targetJointPositions[0])
        p.setJointMotorControl2(self.simPaws, self.FL_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=-FL_targetJointPositions[1])
        p.setJointMotorControl2(self.simPaws, self.FL_KNEE_JOINT, p.POSITION_CONTROL,
                                targetPosition=FL_targetJointPositions[2])

        # Step the simulation 100 times (100/240 = 0.417 s) to move motors to initial positions
        for j in range(0, 100):
            p.stepSimulation()
        # time.sleep(10)  # Uncomment for debugging

        trotStage = 0  # Counter for current stage of trot cycle. Trot has two stages: FR-RL in swing or FL-RR in swing
        self.firstSwingLog = True  # The first log creates a dataframe, subsequent logs append data
        self.firstSimLog = True  # The first log creates a CSV file, subsequent logs append data
        startTime = time.perf_counter()  # Store start time

        camCounter = 0  # Counter to move debug visualiser camera as PAWS moves forwards
        base_Pos, baseOrient = p.getBasePositionAndOrientation(self.simPaws)
        base_Pos = np.array(base_Pos)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=27.2, cameraPitch=-20.4,
                                     cameraTargetPosition=base_Pos)
        self.stepCounter = 0  # Counter for number of simulation steps

        while True:

            base_Pos, baseOrient = p.getBasePositionAndOrientation(self.simPaws)
            base_Pos = np.array(base_Pos)

            # If the x-position of the base has travelled far enough forwards, step the debug camera
            if base_Pos[0] >= 2 * camCounter:
                camCounter += 1
                base_Pos = base_Pos + np.array([1, 0, 0])
                p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=27.2, cameraPitch=-20.4,
                                             cameraTargetPosition=base_Pos)

            for j in range(samplesPerSwing):  # For each sample in the current stage of the trot cycle
                if trotStage == 0:
                    k = 0  # FR in swing
                elif trotStage == 1:
                    k = 2  # FR in stance
                FR_targetFootPosition = feetTrajectories[k][j]  # Desired FR task-space position
                # Desired FR joint-space positions
                FR_targetJointPositions = self._calculateIK(FR_targetFootPosition, kneeLagging=True)

                if trotStage == 0:
                    k = 1  # RL in swing
                elif trotStage == 1:
                    k = 3  # RL in stance
                RL_targetFootPosition = feetTrajectories[k][j]  # Desired RL task-space position
                # Desired RL joint-space positions
                RL_targetJointPositions = self._calculateIK(RL_targetFootPosition, kneeLagging=True)

                if trotStage == 0:
                    k = 2  # RR in stance
                elif trotStage == 1:
                    k = 0  # RR in swing
                RR_targetFootPosition = feetTrajectories[k][j]  # Desired RR task-space position
                # Desired RR joint-space positions
                RR_targetJointPositions = self._calculateIK(RR_targetFootPosition, kneeLagging=True)

                if trotStage == 0:
                    k = 3  # FL in stance
                elif trotStage == 1:
                    k = 1  # FL in swing
                FL_targetFootPosition = feetTrajectories[k][j]  # Desired FL task-space position
                # Desired FL joint-space positions
                FL_targetJointPositions = self._calculateIK(FL_targetFootPosition, kneeLagging=True)

                ### Send commands to all motors ###
                p.setJointMotorControl2(self.simPaws, self.FR_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                        targetPosition=FR_targetJointPositions[0], positionGain=kp, velocityGain=kd)
                p.setJointMotorControl2(self.simPaws, self.FR_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                        targetPosition=FR_targetJointPositions[1], positionGain=kp, velocityGain=kd)
                p.setJointMotorControl2(self.simPaws, self.FR_KNEE_JOINT, p.POSITION_CONTROL,
                                        targetPosition=FR_targetJointPositions[2], positionGain=kp, velocityGain=kd)

                p.setJointMotorControl2(self.simPaws, self.RL_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                        targetPosition=RL_targetJointPositions[0], positionGain=kp, velocityGain=kd)
                p.setJointMotorControl2(self.simPaws, self.RL_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                        targetPosition=-RL_targetJointPositions[1], positionGain=kp, velocityGain=kd)
                p.setJointMotorControl2(self.simPaws, self.RL_KNEE_JOINT, p.POSITION_CONTROL,
                                        targetPosition=RL_targetJointPositions[2], positionGain=kp, velocityGain=kd)

                p.setJointMotorControl2(self.simPaws, self.RR_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                        targetPosition=-RR_targetJointPositions[0], positionGain=kp, velocityGain=kd)
                p.setJointMotorControl2(self.simPaws, self.RR_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                        targetPosition=RR_targetJointPositions[1], positionGain=kp, velocityGain=kd)
                p.setJointMotorControl2(self.simPaws, self.RR_KNEE_JOINT, p.POSITION_CONTROL,
                                        targetPosition=RR_targetJointPositions[2], positionGain=kp, velocityGain=kd)

                p.setJointMotorControl2(self.simPaws, self.FL_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                        targetPosition=-FL_targetJointPositions[0], positionGain=kp, velocityGain=kd)
                p.setJointMotorControl2(self.simPaws, self.FL_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                        targetPosition=-FL_targetJointPositions[1], positionGain=kp, velocityGain=kd)
                p.setJointMotorControl2(self.simPaws, self.FL_KNEE_JOINT, p.POSITION_CONTROL,
                                        targetPosition=FL_targetJointPositions[2], positionGain=kp, velocityGain=kd)

                elapsedTime = time.perf_counter() - startTime

                while elapsedTime < T_swing / self.sampleRate:  # Delay until next sample time
                    elapsedTime = time.perf_counter() - startTime

                p.stepSimulation()  # Step simulation
                self.stepCounter += 1  # Increment step count
                startTime = time.perf_counter()  # Reset timestep start time

                if logData and j % 1 == 0:  # Log every sample. Change to j%2 to log every other sample,
                    # j%3 for every third etc.

                    self._collectSimData()  # Accumulate simulation data for current swing phase in Pandas dataframe

            ### Update the start position of the feet for the next stage of the trot cycle ###
            swingX_0 = feetTrajectories[2][samplesPerSwing - 1][0]
            stanceX_0 = feetTrajectories[0][samplesPerSwing - 1][0]
            swing1Y_0 = feetTrajectories[2][samplesPerSwing - 1][1] - hipOffset
            swing2Y_0 = feetTrajectories[3][samplesPerSwing - 1][1] - hipOffset
            stance1Y_0 = feetTrajectories[0][samplesPerSwing - 1][1] - hipOffset
            stance2Y_0 = feetTrajectories[1][samplesPerSwing - 1][1] - hipOffset

            if logData and self.firstSimLog:  # Create a CSV file if this is the first log of the simulation
                self.cog_df.to_csv(filename, index=False)
                self.cog_df.drop(self.cog_df.index, inplace=True)
                self.firstSimLog = False
            elif logData:  # Append to the CSV file if this is not the first log of the simulation
                self.cog_df.to_csv(filename, mode='a', index=False, header=False)
                self.cog_df.drop(self.cog_df.index, inplace=True)

            self.firstSwingLog = True  # Recreate the dataframe of sim data on the next loop

            trotStage = trotStage + 1  # Move to the next stage of trot
            if trotStage == 2:
                trotStage = 0

            ### Sign corrections for each stage ###
            if trotStage == 0:
                latStanceSign = 1
            elif trotStage == 1:
                latStanceSign = -1

            ### Read data from sliders to obtain step parameters for next stage of the trot gait ###
            forwardSwingLength = p.readUserDebugParameter(forwardSwingLengthIn)
            latSwingLength = p.readUserDebugParameter(latSwingLengthIn)
            swingHeight = p.readUserDebugParameter(swingHeightIn)
            latStanceLength = latStanceSign * p.readUserDebugParameter(latStanceLengthIn)
            T_swing = p.readUserDebugParameter(T_swing_In)
            h_0 = p.readUserDebugParameter(h_0_In)

            ### Create foot trajectories for next stage of the trot gait ###
            feetTrajectories = self._genTrotStep(forwardSwingLength, latSwingLength, swingHeight, latStanceLength,
                                                      T_swing, h_0, self.sampleRate, swingX_0, stanceX_0, swing1Y_0,
                                                      swing2Y_0, stance1Y_0, stance2Y_0, hipOffset=hipOffset)

            samplesPerSwing = feetTrajectories[0].shape[0]


    def simulateWalkGate(self, forwardSwingLength = 0.08, latSwingLength = 0, swingHeight = 0.04, leanLength = 0.05,
                          h_0 = -0.28, T_swing = 0.5, T_lean=0.5, sampleRate=240, backSwingCoef=0.2, hipOffset=0.06,
                          terrain="FLAT", hieght_mean=0.1, height_std=0.01, xy_scale=0.05, stairStepLen=0.2,
                          stairStepWidth=2, stairStepHeight=0.02, fixBase=False, logData=False):
        """""
        This method simulates the walk gait in PyBullet, with two additional lean periods each cycle. Arguments 
        determine the initial step parameters, which can then be altered during the runtime of the simulation using 
        sliders. Also accepts arguments for the terrain type.
        Args:
            forwardSwingLength (float): Length of swing in x-direction relative to hip (negative goes backwards).
            latSwingLength (float): Length of swing in y-direction relative to hip (negative goes inwards).
            swingHeight (float): Maximum height of swing phase relative to the stance height. 
            leanLength (float): Maximum y-lean during lean phases.
            h_0 (float): z-position of feet relative to hip during the stance phase (i.e. - body height).
            T_swing (float): Swing period.
            T_lean (float): Lean period.
            sampleRate (float): Sample rate, recommended 240 Hz.
            backSwingCoef (float): back/forward swing coefficient.
            hipOffset (float): 0 m for feet directly below motors, L1 m for legs straight (recommended).
            terrain (str): type of terrain, is asserted in ["FLAT", "ROUGH", "STAIRS"].
            hieght_mean (float): mean height of terrain if terrain==ROUGH.
            height_std (float): standard deviation of terrain if terrain==ROUGH.
            xy_scale (float): how big each square is if terrain==ROUGH.
            stairStepLen (float): length in x-direction of each step if terrain==STAIRS.
            stairStepWidth (float): width in y-direction of each step if terrain==STAIRS.
            stairStepHeight(float): height in z-direction of each step if terrain==STAIRS.
            fixBase (bool): if True then suspend in air, else free base.
            logData(bool): if True then log simulation data to CSV file.
        Returns:
            None: Simulation continues on infinite loop until program is ended.
        """""

        assert terrain in ["FLAT", "RANDOM_ROUGH", "STAIRS"]  # Ensure valid terrain

        # Create terrain
        if terrain == "FLAT":
            self._createSimEnvironment(terrain)
        elif terrain == "RANDOM_ROUGH":
            self._createSimEnvironment(terrain, hieght_mean, height_std, xy_scale)
        elif terrain == "STAIRS":
            self._createSimEnvironment(terrain, stepLen=stairStepLen, stepWidth=stairStepWidth, stepHeight=stairStepHeight)

        filename = r"C:\Users\Ronan\PawsQuadrupedRobot\log.csv"  # File directory for logs

        startPos = [0, 0, 0.49]
        planeId = p.loadURDF("plane.urdf")
        self.simPaws = p.loadURDF(r'PawsURDF/urdf/Paws.urdf', startPos, useFixedBase=fixBase)

        # Create sliders to change step parameters during runtime
        forwardSwingLengthIn = p.addUserDebugParameter("Forward Step Length", 0, 0.2, forwardSwingLength)
        swingHeightIn = p.addUserDebugParameter("Step Height", 0, 0.15, swingHeight)
        latSwingLengthIn = p.addUserDebugParameter("Lateral Step Length", -0.1, 0.1, latSwingLength)
        leanLengthIn = p.addUserDebugParameter("Lean Length", 0, 0.1, leanLength)
        T_swing_In = p.addUserDebugParameter("Swing Period", 0.1, 5, T_swing)
        T_lean_In = p.addUserDebugParameter("Lean Period", 0.1, 5, T_lean)
        h_0_In = p.addUserDebugParameter("Body Height", -0.33, -0.13, h_0)

        leanDirection = False  # Define boolean to change lean direction every lean phase

        # Generate foot trajectories for the first swing phase
        leanTrajectory = self._genLeanOutTrajectory(leanLength, T_lean, h_0, sampleRate, hipOffset, leanDirection,
                                                         -forwardSwingLength/2, 0, -forwardSwingLength/6, 0,
                                                         forwardSwingLength/6, 0, forwardSwingLength/2, 0)
        self.sampleRate = sampleRate  # Update class attribute
        samplesPerLean = leanTrajectory[0].shape[0]  # Determine the number of samples per swing

        # Calculate initial joint positions for all legs using IK
        FR_targetFootPosition = leanTrajectory[0][0]  # Initially swing
        FR_targetJointPositions = self._calculateIK(FR_targetFootPosition, kneeLagging=True)
        RL_targetFootPosition = leanTrajectory[1][0]  # Initially stance 1
        RL_targetJointPositions = self._calculateIK(RL_targetFootPosition, kneeLagging=True)
        FL_targetFootPosition = leanTrajectory[2][0]  # Initially stance 2
        FL_targetJointPositions = self._calculateIK(FL_targetFootPosition, kneeLagging=True)
        RR_targetFootPosition = leanTrajectory[3][0]  # Initially stance 3
        RR_targetJointPositions = self._calculateIK(RR_targetFootPosition, kneeLagging=True)
        startTime = time.perf_counter()

        ### Send commands to all motors ###
        p.setJointMotorControl2(self.simPaws, self.FR_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=FR_targetJointPositions[0])
        p.setJointMotorControl2(self.simPaws, self.FR_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=FR_targetJointPositions[1])
        p.setJointMotorControl2(self.simPaws, self.FR_KNEE_JOINT, p.POSITION_CONTROL,
                                targetPosition=FR_targetJointPositions[2])

        p.setJointMotorControl2(self.simPaws, self.RL_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=RL_targetJointPositions[0])
        p.setJointMotorControl2(self.simPaws, self.RL_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=-RL_targetJointPositions[1])
        p.setJointMotorControl2(self.simPaws, self.RL_KNEE_JOINT, p.POSITION_CONTROL,
                                targetPosition=RL_targetJointPositions[2])

        p.setJointMotorControl2(self.simPaws, self.FL_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=-FL_targetJointPositions[0])
        p.setJointMotorControl2(self.simPaws, self.FL_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=-FL_targetJointPositions[1])
        p.setJointMotorControl2(self.simPaws, self.FL_KNEE_JOINT, p.POSITION_CONTROL,
                                targetPosition=FL_targetJointPositions[2])

        p.setJointMotorControl2(self.simPaws, self.RR_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=-RR_targetJointPositions[0])
        p.setJointMotorControl2(self.simPaws, self.RR_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=RR_targetJointPositions[1])
        p.setJointMotorControl2(self.simPaws, self.RR_KNEE_JOINT, p.POSITION_CONTROL,
                                targetPosition=RR_targetJointPositions[2])

        # Step the simulation 100 times (100/240 = 0.417 s) to move motors to initial positions
        for j in range(0, 100):
            p.stepSimulation()
        # time.sleep(10)  # Uncomment for debugging


        ### Perform initial lean phase ###
        startTime = time.perf_counter()  # Store start time

        for j in range(samplesPerLean):

            FR_targetFootPosition = leanTrajectory[0][j]  # Desired FR task-space position
            # Desired FR joint-space positions
            FR_targetJointPositions = self._calculateIK(FR_targetFootPosition, kneeLagging=True)

            RL_targetFootPosition = leanTrajectory[1][j]  # Desired RL task-space position
            # Desired RL joint-space positions
            RL_targetJointPositions = self._calculateIK(RL_targetFootPosition, kneeLagging=True)

            FL_targetFootPosition = leanTrajectory[2][j]  # Desired FL task-space position
            # Desired FL joint-space positions
            FL_targetJointPositions = self._calculateIK(FL_targetFootPosition, kneeLagging=True)

            RR_targetFootPosition = leanTrajectory[3][j]  # Desired RR task-space position
            # Desired RR joint-space positions
            RR_targetJointPositions = self._calculateIK(RR_targetFootPosition, kneeLagging=True)

            ### Send commands to all motors ###
            p.setJointMotorControl2(self.simPaws, self.FR_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=FR_targetJointPositions[0])
            p.setJointMotorControl2(self.simPaws, self.FR_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=FR_targetJointPositions[1])
            p.setJointMotorControl2(self.simPaws, self.FR_KNEE_JOINT, p.POSITION_CONTROL,
                                    targetPosition=FR_targetJointPositions[2])

            p.setJointMotorControl2(self.simPaws, self.RL_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=RL_targetJointPositions[0])
            p.setJointMotorControl2(self.simPaws, self.RL_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=-RL_targetJointPositions[1])
            p.setJointMotorControl2(self.simPaws, self.RL_KNEE_JOINT, p.POSITION_CONTROL,
                                    targetPosition=RL_targetJointPositions[2])

            p.setJointMotorControl2(self.simPaws, self.FL_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=-FL_targetJointPositions[0])
            p.setJointMotorControl2(self.simPaws, self.FL_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=-FL_targetJointPositions[1])
            p.setJointMotorControl2(self.simPaws, self.FL_KNEE_JOINT, p.POSITION_CONTROL,
                                    targetPosition=FL_targetJointPositions[2])

            p.setJointMotorControl2(self.simPaws, self.RR_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=-RR_targetJointPositions[0])
            p.setJointMotorControl2(self.simPaws, self.RR_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=RR_targetJointPositions[1])
            p.setJointMotorControl2(self.simPaws, self.RR_KNEE_JOINT, p.POSITION_CONTROL,
                                    targetPosition=RR_targetJointPositions[2])

            elapsedTime = time.perf_counter() - startTime

            while elapsedTime < T_lean / sampleRate:  # Delay until next sample time
                elapsedTime = time.perf_counter() - startTime

            p.stepSimulation()  # Step simulation
            startTime = time.perf_counter()  # Reset timestep start time

        ### Update the start position of the feet for the next stage of the walking cycle ###
        swingX_0 = leanTrajectory[0][samplesPerLean - 1][0]
        swingY_0 = leanTrajectory[0][samplesPerLean - 1][1] - hipOffset
        stance1X_0 = leanTrajectory[3][samplesPerLean - 1][0]
        stance1Y_0 = leanTrajectory[3][samplesPerLean - 1][1] - hipOffset
        stance2X_0 = leanTrajectory[2][samplesPerLean - 1][0]
        stance2Y_0 = leanTrajectory[2][samplesPerLean - 1][1] - hipOffset
        stance3X_0 = leanTrajectory[1][samplesPerLean - 1][0]
        stance3Y_0 = leanTrajectory[1][samplesPerLean - 1][1] - hipOffset

        ### Create foot trajectories for next stage of the trot gait ###
        feetTrajectories = self._genWalkStep(forwardSwingLength, latSwingLength, swingHeight,
                                                  T_swing, h_0, sampleRate, swingX_0, swingY_0, stance1X_0, stance1Y_0,
                                                  stance2X_0, stance2Y_0, stance3X_0, stance3Y_0, backSwingCoef,
                                                  hipOffset, stage=0)
        samplesPerSwing = feetTrajectories[0].shape[0]

        walkStage = 0
        self.firstSwingLog = True  # The first log creates a dataframe, subsequent logs append data
        self.firstSimLog = True    # The first log creates a CSV file, subsequent logs append data
        camCounter=1               # Counter to move debug visualiser camera as PAWS moves forwards
        startTime = time.perf_counter()  # Store start time
        self.stepCounter = 0       # Counter for number of simulation steps

        # Initial debug camera position
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=27.2, cameraPitch=-20.4,
                                     cameraTargetPosition=[1.006656527519226, 0.016066139563918114, 0.2826265096664429])


        while True:

            base_Pos, baseOrient = p.getBasePositionAndOrientation(self.simPaws)
            base_Pos = np.array(base_Pos)

            # If the x-position of the base has travelled far enough forwards, step the debug camera
            if base_Pos[0] >= camCounter:
                camCounter += 1
                p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=27.2, cameraPitch=-20.4,
                                             cameraTargetPosition=base_Pos)

            # If walking stage 1 or 3 (i.e. RL about to swing or RR about to swing) a lean period is inserted first
            if walkStage==1 or walkStage==3:

                for j in range(samplesPerLean):  # For each sample in the lean trajectory

                    FR_targetFootPosition = leanTrajectory[0][j]  # Desired FR task-space position
                    # Desired FR joint-space positions
                    FR_targetJointPositions = self._calculateIK(FR_targetFootPosition, kneeLagging=True)

                    RL_targetFootPosition = leanTrajectory[1][j]  # Desired RL task-space position
                    # Desired RL joint-space positions
                    RL_targetJointPositions = self._calculateIK(RL_targetFootPosition, kneeLagging=True)

                    FL_targetFootPosition = leanTrajectory[2][j]   # Desired FL task-space position
                    # Desired FL joint-space positions
                    FL_targetJointPositions = self._calculateIK(FL_targetFootPosition, kneeLagging=True)

                    RR_targetFootPosition = leanTrajectory[3][j]  # Desired RR task-space position
                    # Desired RR joint-space positions
                    RR_targetJointPositions = self._calculateIK(RR_targetFootPosition, kneeLagging=True)

                    ### Send commands to all motors ###
                    p.setJointMotorControl2(self.simPaws, self.FR_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                            targetPosition=FR_targetJointPositions[0])
                    p.setJointMotorControl2(self.simPaws, self.FR_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                            targetPosition=FR_targetJointPositions[1])
                    p.setJointMotorControl2(self.simPaws, self.FR_KNEE_JOINT, p.POSITION_CONTROL,
                                            targetPosition=FR_targetJointPositions[2])

                    p.setJointMotorControl2(self.simPaws, self.RL_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                            targetPosition=RL_targetJointPositions[0])
                    p.setJointMotorControl2(self.simPaws, self.RL_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                            targetPosition=-RL_targetJointPositions[1])
                    p.setJointMotorControl2(self.simPaws, self.RL_KNEE_JOINT, p.POSITION_CONTROL,
                                            targetPosition=RL_targetJointPositions[2])

                    p.setJointMotorControl2(self.simPaws, self.FL_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                            targetPosition=-FL_targetJointPositions[0])
                    p.setJointMotorControl2(self.simPaws, self.FL_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                            targetPosition=-FL_targetJointPositions[1])
                    p.setJointMotorControl2(self.simPaws, self.FL_KNEE_JOINT, p.POSITION_CONTROL,
                                            targetPosition=FL_targetJointPositions[2])

                    p.setJointMotorControl2(self.simPaws, self.RR_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                            targetPosition=-RR_targetJointPositions[0])
                    p.setJointMotorControl2(self.simPaws, self.RR_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                            targetPosition=RR_targetJointPositions[1])
                    p.setJointMotorControl2(self.simPaws, self.RR_KNEE_JOINT, p.POSITION_CONTROL,
                                            targetPosition=RR_targetJointPositions[2])

                    elapsedTime = time.perf_counter() - startTime

                    while elapsedTime < T_lean / self.sampleRate:  # Delay until next sample time
                        elapsedTime = time.perf_counter() - startTime

                    p.stepSimulation()  # Step simulation
                    self.stepCounter += 1  # Increment step count
                    startTime = time.perf_counter()  # Reset timestep start time

                    if logData and j%1 == 0:    # Log every sample. Change to j%2 to log every other sample,
                                                # j%3 for every third etc.

                        self._collectSimData()  # Accumulate simulation data for current swing phase in Pandas dataframe


            for j in range(samplesPerSwing):   # For each sample in the current stage of the walking cycle

                FR_targetFootPosition = feetTrajectories[walkStage][j]  # Desired FR task-space position
                # Desired FR joint-space positions
                FR_targetJointPositions = self._calculateIK(FR_targetFootPosition, kneeLagging=True)

                # Determine stage of RL leg (swing, stance 1, stance 2 or stance 3)
                if walkStage+3 <= 3:
                    k = walkStage+3
                else:
                    k = walkStage+3-4
                RL_targetFootPosition = feetTrajectories[k][j]  # Desired RL task-space position
                # Desired RL joint-space positions
                RL_targetJointPositions = self._calculateIK(RL_targetFootPosition, kneeLagging=True)

                # Determine stage of FL leg (swing, stance 1, stance 2 or stance 3)
                if walkStage+2 <= 3:
                    k = walkStage+2
                else:
                    k= walkStage+2-4
                FL_targetFootPosition = feetTrajectories[k][j]  # Desired FL task-space position
                # Desired FL joint-space positions
                FL_targetJointPositions = self._calculateIK(FL_targetFootPosition, kneeLagging=True)

                # Determine stage of RR leg (swing, stance 1, stance 2 or stance 3)
                if walkStage+1 <= 3:
                    k = walkStage+1
                else:
                    k= walkStage+1-4
                RR_targetFootPosition = feetTrajectories[k][j]  # Desired RR task-space position
                # Desired RR joint-space positions
                RR_targetJointPositions = self._calculateIK(RR_targetFootPosition, kneeLagging=True)

                ### Send commands to all motors ###
                p.setJointMotorControl2(self.simPaws, self.FR_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                        targetPosition=FR_targetJointPositions[0])
                p.setJointMotorControl2(self.simPaws, self.FR_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                        targetPosition=FR_targetJointPositions[1])
                p.setJointMotorControl2(self.simPaws, self.FR_KNEE_JOINT, p.POSITION_CONTROL,
                                        targetPosition=FR_targetJointPositions[2])

                p.setJointMotorControl2(self.simPaws, self.RL_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                        targetPosition=RL_targetJointPositions[0])
                p.setJointMotorControl2(self.simPaws, self.RL_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                        targetPosition=-RL_targetJointPositions[1])
                p.setJointMotorControl2(self.simPaws, self.RL_KNEE_JOINT, p.POSITION_CONTROL,
                                        targetPosition=RL_targetJointPositions[2])

                p.setJointMotorControl2(self.simPaws, self.FL_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                        targetPosition=-FL_targetJointPositions[0])
                p.setJointMotorControl2(self.simPaws, self.FL_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                        targetPosition=-FL_targetJointPositions[1])
                p.setJointMotorControl2(self.simPaws, self.FL_KNEE_JOINT, p.POSITION_CONTROL,
                                        targetPosition=FL_targetJointPositions[2])

                p.setJointMotorControl2(self.simPaws, self.RR_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                        targetPosition=-RR_targetJointPositions[0])
                p.setJointMotorControl2(self.simPaws, self.RR_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                        targetPosition=RR_targetJointPositions[1])
                p.setJointMotorControl2(self.simPaws, self.RR_KNEE_JOINT, p.POSITION_CONTROL,
                                        targetPosition=RR_targetJointPositions[2])

                elapsedTime = time.perf_counter() - startTime

                while elapsedTime < T_swing / sampleRate:  # Delay until next sample time
                    elapsedTime = time.perf_counter() - startTime

                p.stepSimulation()  # Step simulation
                self.stepCounter += 1  # Increment step count
                startTime = time.perf_counter()  # Reset timestep start time

                if logData and j % 1 == 0:   # Log every sample. Change to j%2 to log every other sample,
                                             # j%3 for every third etc.

                    self._collectSimData()  # Accumulate simulation data for current swing phase in Pandas dataframe


            if logData and self.firstSimLog:  # Create a CSV file if this is the first log of the simulation
                self.cog_df.to_csv(filename, index=False)
                self.cog_df.drop(self.cog_df.index, inplace=True)
                self.firstSimLog = False
            elif logData:  # Append to the CSV file if this is not the first log of the simulation
                self.cog_df.to_csv(filename, mode='a', index=False, header=False)
                self.cog_df.drop(self.cog_df.index, inplace=True)

            self.firstSwingLog = True  # Recreate the dataframe of sim data on the next loop

            ### Update the start position of the feet for the next stage of the walking cycle ###
            swingX_0 = feetTrajectories[3][samplesPerSwing-1][0]
            swingY_0 = feetTrajectories[3][samplesPerSwing-1][1] - hipOffset
            stance1X_0 = feetTrajectories[0][samplesPerSwing-1][0]
            stance1Y_0 = feetTrajectories[0][samplesPerSwing-1][1] - hipOffset
            stance2X_0 = feetTrajectories[1][samplesPerSwing-1][0]
            stance2Y_0 = feetTrajectories[1][samplesPerSwing-1][1] - hipOffset
            stance3X_0 = feetTrajectories[2][samplesPerSwing-1][0]
            stance3Y_0 = feetTrajectories[2][samplesPerSwing-1][1] - hipOffset

            walkStage = walkStage + 1  # Move to the next stage of the walking gait
            if walkStage == 4:
                walkStage = 0

            ### Sign corrections for each stage
            if walkStage == 0 or walkStage==3:
                latSwingSignCorrection = 1
            elif walkStage == 1 or walkStage==2:
                latSwingSignCorrection = -1

            ### Read data from sliders to obtain step parameters for next stage of the walking gait ###
            forwardSwingLength = p.readUserDebugParameter(forwardSwingLengthIn)
            latSwingLength = latSwingSignCorrection*p.readUserDebugParameter(latSwingLengthIn)
            swingHeight = p.readUserDebugParameter(swingHeightIn)
            leanLength = p.readUserDebugParameter(leanLengthIn)
            T_swing = p.readUserDebugParameter(T_swing_In)
            T_lean = p.readUserDebugParameter(T_lean_In)
            h_0 = p.readUserDebugParameter(h_0_In)

            if walkStage == 1: # If the next swing foot will be RL, insert a lean in phase

                leanDirection = True
                ### Create foot trajectories for next lean stage of the walking gait ###
                leanTrajectory = self._genLeanInTrajectory(2*leanLength, T_lean, h_0, sampleRate, hipOffset,
                                                                leanDirection, stance1X_0, stance1Y_0, swingX_0,
                                                                swingY_0, stance3X_0, stance3Y_0, stance2X_0,
                                                                stance2Y_0)
                samplesPerLean = leanTrajectory[0].shape[0]

                ### Update the start position of the feet for the next swing stage of the walking cycle ###
                swingX_0 = leanTrajectory[1][samplesPerLean - 1][0]
                swingY_0 = leanTrajectory[1][samplesPerLean - 1][1] - hipOffset
                stance1X_0 = leanTrajectory[0][samplesPerLean - 1][0]
                stance1Y_0 = leanTrajectory[0][samplesPerLean - 1][1] - hipOffset
                stance2X_0 = leanTrajectory[3][samplesPerLean - 1][0]
                stance2Y_0 = leanTrajectory[3][samplesPerLean - 1][1] - hipOffset
                stance3X_0 = leanTrajectory[2][samplesPerLean - 1][0]
                stance3Y_0 = leanTrajectory[2][samplesPerLean - 1][1] - hipOffset

            elif walkStage == 3: # If the next swing foot will be RR, insert a lean out phase

                leanDirection = False
                ### Create foot trajectories for next lean stage of the walking gait ###
                leanTrajectory = self._genLeanOutTrajectory(2*leanLength, T_lean, h_0, sampleRate, hipOffset,
                                                                 leanDirection, stance3X_0, stance3Y_0, stance2X_0,
                                                                 stance2Y_0, stance1X_0, stance1Y_0, swingX_0,
                                                                 swingY_0)
                samplesPerLean = leanTrajectory[0].shape[0]

                ### Update the start position of the feet for the next swing stage of the walking cycle ###
                swingX_0 = leanTrajectory[3][samplesPerLean - 1][0]
                swingY_0 = leanTrajectory[3][samplesPerLean - 1][1] - hipOffset
                stance1X_0 = leanTrajectory[2][samplesPerLean - 1][0]
                stance1Y_0 = leanTrajectory[2][samplesPerLean - 1][1] - hipOffset
                stance2X_0 = leanTrajectory[1][samplesPerLean - 1][0]
                stance2Y_0 = leanTrajectory[1][samplesPerLean - 1][1] - hipOffset
                stance3X_0 = leanTrajectory[0][samplesPerLean - 1][0]
                stance3Y_0 = leanTrajectory[0][samplesPerLean - 1][1] - hipOffset

            ### Create foot trajectories for next swing stage of the walking gait ###
            feetTrajectories = self._genWalkStep(forwardSwingLength, latSwingLength, swingHeight, T_swing, h_0,
                                                       sampleRate, swingX_0, swingY_0, stance1X_0, stance1Y_0,
                                                       stance2X_0, stance2Y_0, stance3X_0, stance3Y_0,
                                                       hipOffset=hipOffset, stage=walkStage)

            samplesPerSwing = feetTrajectories[0].shape[0]


    def simulateWalkGateNoAdditionalLean(self, forwardSwingLength=0.08, latSwingLength=0.0, swingHeight=0.04,
                                         latStanceLength=0.0, T_swing=0.2, h_0=-0.28, sampleRate=240, backSwingCoef=0.2,
                                         hipOffset=0.06, terrain="FLAT", hieght_mean=0.1, height_std=0.011,
                                         xy_scale=0.05, stairStepLen=0.2, stairStepWidth=2, stairStepHeight=0.03,
                                         fixBase=False, logData=False):
        """""
        This method simulates the walk gait in PyBullet, with no additional lean periods but instead leaning while one 
        foot is swinging. Arguments determine the initial step parameters, which can then be changed during the runtime 
        of the simulation using sliders. Also accepts arguments for the terrain type.
        Args:
            forwardSwingLength (float): Length of swing in x-direction relative to hip (negative goes backwards).
            latSwingLength (float): Length of swing in y-direction relative to hip (negative goes inwards).
            swingHeight (float): Maximum height of swing phase relative to the stance height. 
            latStanceLength (float): Maximum y-lean during stance phase.
            T_swing (float): Swing period.
            h_0 (float): z-position of feet relative to hip during the stance phase (i.e. - body height).
            sampleRate (float): Sample rate, recommended 240 Hz.
            backSwingCoef (float): back/forward swing coefficient.
            hipOffset (float): 0 m for feet directly below motors, L1 m for legs straight (recommended).
            terrain (str): type of terrain, is asserted in ["FLAT", "ROUGH", "STAIRS"].
            hieght_mean (float): mean height of terrain if terrain==ROUGH.
            height_std (float): standard deviation of terrain if terrain==ROUGH.
            xy_scale (float): how big each square is if terrain==ROUGH.
            stairStepLen (float): length in x-direction of each step if terrain==STAIRS.
            stairStepWidth (float): width in y-direction of each step if terrain==STAIRS.
            stairStepHeight(float): height in z-direction of each step if terrain==STAIRS.
            fixBase (bool): if True then suspend in air, else free base.
            logData(bool): if True then log simulation data to CSV file.
        Returns:
            None: Simulation continues on infinite loop until program is ended.
        """""
        assert terrain in ["FLAT", "RANDOM_ROUGH", "STAIRS"]  # Ensure valid terrain

        # Create terrain
        if terrain == "FLAT":
            self._createSimEnvironment(terrain)
        elif terrain == "RANDOM_ROUGH":
            self._createSimEnvironment(terrain, hieght_mean, height_std, xy_scale)
        elif terrain == "STAIRS":
            self._createSimEnvironment(terrain, stepLen=stairStepLen, stepWidth=stairStepWidth,
                                       stepHeight=stairStepHeight)

        filename = r"C:\Users\Ronan\PawsQuadrupedRobot\log.csv"  # File directory for logs

        startPos = [0, 0, 0.49]
        planeId = p.loadURDF("plane.urdf")
        self.simPaws = p.loadURDF(r'PawsURDF/urdf/Paws.urdf', startPos, useFixedBase=fixBase)

        ### Initial position of feet ###
        swingX_0 = -forwardSwingLength / 2
        swingY_0 = 0
        stance1X_0 = -forwardSwingLength / 6
        stance1Y_0 = 0
        stance2X_0 = forwardSwingLength / 6
        stance2Y_0 = 0
        stance3X_0 = forwardSwingLength / 2
        stance3Y_0 = 0

        # Create sliders to change step parameters during runtime
        forwardSwingLengthIn = p.addUserDebugParameter("Forward Step Length", -0.2, 0.2, forwardSwingLength)
        swingHeightIn = p.addUserDebugParameter("Step Height", 0.03, 0.15, swingHeight)
        latSwingLengthIn = p.addUserDebugParameter("Lateral Step Length", -0.1, 0.1, latSwingLength)
        latStanceLengthIn = p.addUserDebugParameter("Lateral Stance Length", 0, 0.1, latStanceLength)
        T_swing_In = p.addUserDebugParameter("Swing Period", 0.1, 5, T_swing)
        h_0_In = p.addUserDebugParameter("Body Height", -0.33, -0.13, h_0)

        feetTrajectories = self._genWalkStepNoAdditionalLean(forwardSwingLength, latSwingLength, swingHeight,
                                                             latStanceLength, T_swing, h_0, sampleRate, swingX_0,
                                                             swingY_0, stance1X_0, stance1Y_0, stance2X_0, stance2Y_0,
                                                             stance3X_0, stance3Y_0, backSwingCoef, hipOffset, stage=0)

        self.sampleRate = sampleRate  # Update class attribute
        samplesPerSwing = feetTrajectories[0].shape[0]

        ### Calculate initial joint positions for all legs using IK ###
        FR_targetFootPosition = feetTrajectories[0][0]
        FR_targetJointPositions = self._calculateIK(FR_targetFootPosition, kneeLagging=True)
        RL_targetFootPosition = feetTrajectories[3][0]
        RL_targetJointPositions = self._calculateIK(RL_targetFootPosition, kneeLagging=True)
        FL_targetFootPosition = feetTrajectories[2][0]
        FL_targetJointPositions = self._calculateIK(FL_targetFootPosition, kneeLagging=True)
        RR_targetFootPosition = feetTrajectories[1][0]
        RR_targetJointPositions = self._calculateIK(RR_targetFootPosition, kneeLagging=True)

        ### Send commands to all motors ###
        p.setJointMotorControl2(self.simPaws, self.FR_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=FR_targetJointPositions[0])
        p.setJointMotorControl2(self.simPaws, self.FR_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=FR_targetJointPositions[1])
        p.setJointMotorControl2(self.simPaws, self.FR_KNEE_JOINT, p.POSITION_CONTROL,
                                targetPosition=FR_targetJointPositions[2])

        p.setJointMotorControl2(self.simPaws, self.RL_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=RL_targetJointPositions[0]) # Changed to -
        p.setJointMotorControl2(self.simPaws, self.RL_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=-RL_targetJointPositions[1])
        p.setJointMotorControl2(self.simPaws, self.RL_KNEE_JOINT, p.POSITION_CONTROL,
                                targetPosition=RL_targetJointPositions[2])

        p.setJointMotorControl2(self.simPaws, self.FL_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=-FL_targetJointPositions[0]) # Changed to +
        p.setJointMotorControl2(self.simPaws, self.FL_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=-FL_targetJointPositions[1])
        p.setJointMotorControl2(self.simPaws, self.FL_KNEE_JOINT, p.POSITION_CONTROL,
                                targetPosition=FL_targetJointPositions[2])

        p.setJointMotorControl2(self.simPaws, self.RR_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=-RR_targetJointPositions[0])
        p.setJointMotorControl2(self.simPaws, self.RR_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=RR_targetJointPositions[1])
        p.setJointMotorControl2(self.simPaws, self.RR_KNEE_JOINT, p.POSITION_CONTROL,
                                targetPosition=RR_targetJointPositions[2])

        # Step the simulation 100 times (100/240 = 0.417 s) to move motors to initial positions
        for j in range(0, 100):
            p.stepSimulation()
        # time.sleep(10)  # Uncomment for debugging

        walkStage = 0  # Counter for current stage of trot cycle. Trot has two stages: FR-RL in swing or FL-RR in swing
        self.firstSwingLog = True  # The first log creates a dataframe, subsequent logs append data
        self.firstSimLog = True  # The first log creates a CSV file, subsequent logs append data
        startTime = time.perf_counter()  # Store start time

        camCounter = 0  # Counter to move debug visualiser camera as PAWS moves forwards
        base_Pos, baseOrient = p.getBasePositionAndOrientation(self.simPaws)
        base_Pos = np.array(base_Pos)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=27.2, cameraPitch=-20.4,
                                     cameraTargetPosition=base_Pos)
        self.stepCounter = 0  # Counter for number of simulation steps

        while True:

            base_Pos, baseOrient = p.getBasePositionAndOrientation(self.simPaws)
            base_Pos = np.array(base_Pos)

            # If the x-position of the base has travelled far enough forwards, step the debug camera
            if base_Pos[0] >= 2 * camCounter:
                camCounter += 1
                base_Pos = base_Pos + np.array([1, 0, 0])
                p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=27.2, cameraPitch=-20.4,
                                             cameraTargetPosition=base_Pos)

            for j in range(samplesPerSwing):

                FR_targetFootPosition = feetTrajectories[walkStage][j]
                FR_targetJointPositions = self._calculateIK(FR_targetFootPosition, kneeLagging=True)

                if walkStage+3 <= 3:
                    k = walkStage+3
                else:
                    k= walkStage+3-4
                RL_targetFootPosition = feetTrajectories[k][j]
                RL_targetJointPositions = self._calculateIK(RL_targetFootPosition, kneeLagging=True)

                if walkStage+2 <= 3:
                    k = walkStage+2
                else:
                    k= walkStage+2-4
                FL_targetFootPosition = feetTrajectories[k][j]
                FL_targetJointPositions = self._calculateIK(FL_targetFootPosition, kneeLagging=True)

                if walkStage+1 <= 3:
                    k = walkStage+1
                else:
                    k= walkStage+1-4
                RR_targetFootPosition = feetTrajectories[k][j]
                RR_targetJointPositions = self._calculateIK(RR_targetFootPosition, kneeLagging=True)

                ### Send commands to all motors ###
                p.setJointMotorControl2(self.simPaws, self.FR_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                        targetPosition=FR_targetJointPositions[0])
                p.setJointMotorControl2(self.simPaws, self.FR_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                        targetPosition=FR_targetJointPositions[1])
                p.setJointMotorControl2(self.simPaws, self.FR_KNEE_JOINT, p.POSITION_CONTROL,
                                        targetPosition=FR_targetJointPositions[2])

                p.setJointMotorControl2(self.simPaws, self.RL_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                        targetPosition=RL_targetJointPositions[0])
                p.setJointMotorControl2(self.simPaws, self.RL_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                        targetPosition=-RL_targetJointPositions[1])
                p.setJointMotorControl2(self.simPaws, self.RL_KNEE_JOINT, p.POSITION_CONTROL,
                                        targetPosition=RL_targetJointPositions[2])

                p.setJointMotorControl2(self.simPaws, self.FL_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                        targetPosition=-FL_targetJointPositions[0])
                p.setJointMotorControl2(self.simPaws, self.FL_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                        targetPosition=-FL_targetJointPositions[1])
                p.setJointMotorControl2(self.simPaws, self.FL_KNEE_JOINT, p.POSITION_CONTROL,
                                        targetPosition=FL_targetJointPositions[2])

                p.setJointMotorControl2(self.simPaws, self.RR_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                        targetPosition=-RR_targetJointPositions[0])
                p.setJointMotorControl2(self.simPaws, self.RR_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                        targetPosition=RR_targetJointPositions[1])
                p.setJointMotorControl2(self.simPaws, self.RR_KNEE_JOINT, p.POSITION_CONTROL,
                                        targetPosition=RR_targetJointPositions[2])

                elapsedTime = time.perf_counter() - startTime

                while elapsedTime < T_swing / self.sampleRate:  # Delay until next sample time
                    elapsedTime = time.perf_counter() - startTime

                p.stepSimulation()  # Step simulation
                self.stepCounter += 1  # Increment step count
                startTime = time.perf_counter()  # Reset timestep start time

                if logData and j % 1 == 0:  # Log every sample. Change to j%2 to log every other sample,
                    # j%3 for every third etc.

                    self._collectSimData()  # Accumulate simulation data for current swing phase in Pandas dataframe

            ### Update the start position of the feet for the next stage of the walking cycle ###
            swingX_0 = feetTrajectories[3][samplesPerSwing-1][0]
            swingY_0 = feetTrajectories[3][samplesPerSwing-1][1] - hipOffset
            stance1X_0 = feetTrajectories[0][samplesPerSwing-1][0]
            stance1Y_0 = feetTrajectories[0][samplesPerSwing-1][1] - hipOffset
            stance2X_0 = feetTrajectories[1][samplesPerSwing-1][0]
            stance2Y_0 = feetTrajectories[1][samplesPerSwing-1][1] - hipOffset
            stance3X_0 = feetTrajectories[2][samplesPerSwing-1][0]
            stance3Y_0 = feetTrajectories[2][samplesPerSwing-1][1] - hipOffset

            if logData and self.firstSimLog:  # Create a CSV file if this is the first log of the simulation
                self.cog_df.to_csv(filename, index=False)
                self.cog_df.drop(self.cog_df.index, inplace=True)
                self.firstSimLog = False
            elif logData:  # Append to the CSV file if this is not the first log of the simulation
                self.cog_df.to_csv(filename, mode='a', index=False, header=False)
                self.cog_df.drop(self.cog_df.index, inplace=True)

            self.firstSwingLog = True  # Recreate the dataframe of sim data on the next loop

            walkStage = walkStage + 1  # Move to the next stage of walking
            if walkStage == 4:
                walkStage = 0

            ### Sign corrections for each stage ###
            if walkStage == 0 or walkStage==3:
                latSwingSignCorrection = 1
                latStanceSignCorrection = 1
            elif walkStage == 1 or walkStage==2:
                latSwingSignCorrection = -1
                latStanceSignCorrection = -1

            ### Read data from sliders to obtain step parameters for next stage of the walking gait ###
            forwardSwingLength = p.readUserDebugParameter(forwardSwingLengthIn)
            latSwingLength = latSwingSignCorrection*p.readUserDebugParameter(latSwingLengthIn)
            swingHeight = p.readUserDebugParameter(swingHeightIn)
            latStanceLength = latStanceSignCorrection*p.readUserDebugParameter(latStanceLengthIn)
            T_swing = p.readUserDebugParameter(T_swing_In)
            h_0 = p.readUserDebugParameter(h_0_In)

            ### Create foot trajectories for next stage of the walking gait ###
            feetTrajectories = self._genWalkStepNoAdditionalLean(forwardSwingLength, latSwingLength, swingHeight,
                                                                 latStanceLength, T_swing, h_0, sampleRate, swingX_0,
                                                                 swingY_0, stance1X_0, stance1Y_0, stance2X_0,
                                                                 stance2Y_0, stance3X_0, stance3Y_0,
                                                                 hipOffset=hipOffset, stage=walkStage)

            samplesPerSwing = feetTrajectories[0].shape[0]


    def runTrotGate(self, forwardSwingLength=0.08, latSwingLength=0.0, swingHeight=0.04,
                     latStanceLength=0.0, T_swing=0.2, h_0=-0.28, sampleRate=240, backSwingCoef=0.2,
                     hipOffset=0.06):
        """""
        This method runs the trot gait on the physical PAWS robot. Arguments determine the step parameters, which are 
        then kept constant.
        Args:
            forwardSwingLength (float): Length of swing in x-direction relative to hip (negative goes backwards).
            latSwingLength (float): Length of swing in y-direction relative to hip (negative goes inwards).
            swingHeight (float): Maximum height of swing phase relative to the stance height. 
            latStanceLength (float): Maximum y-lean during stance phase. Recommended 0 m.
            T_swing (float): Swing period.
            h_0 (float): z-position of feet relative to hip during the stance phase (i.e. - body height).
            sampleRate (float): Sample rate, recommended 240 Hz.
            backSwingCoef (float): back/forward swing coefficient.
            hipOffset (float): 0 m for feet directly below motors, L1 m for legs straight (recommended).
        Returns:
            None: Program continues in an infinite loop until is ended.
        """""

        ### Configure CAN bus settings ###
        os.system('sudo ifconfig can0 down')
        os.system('sudo ip link set can0 type can bitrate 1000000')
        os.system('sudo ifconfig can0 txqueuelen 100000')
        os.system('sudo ifconfig can0 up')

        # Create CAN bus object
        bus = can.Bus(interface='socketcan', channel='can0', bitrate=1000000)

        swingX_0 = -forwardSwingLength / 2
        stanceX_0 = forwardSwingLength / 2
        swing1Y_0 = 0
        swing2Y_0 = 0
        stance1Y_0 = 0
        stance2Y_0 = 0

        # Generate foot trajectories for the first swing phase
        feetTrajectories = self._genTrotStep(forwardSwingLength, latSwingLength, swingHeight, latStanceLength,
                                                  T_swing, h_0, sampleRate, swingX_0, stanceX_0, swing1Y_0, swing2Y_0,
                                                  stance1Y_0, stance2Y_0, backSwingCoef, hipOffset)

        # Calculate initial joint positions for all legs using IK
        FR_targetFootPosition = feetTrajectories[0][0]
        FR_targetJointPositions = self._calculateIK(FR_targetFootPosition, kneeLagging=True)
        RL_targetFootPosition = feetTrajectories[1][0]
        RL_targetJointPositions = self._calculateIK(RL_targetFootPosition, kneeLagging=False)
        RR_targetFootPosition = feetTrajectories[2][0]
        RR_targetJointPositions = self._calculateIK(RR_targetFootPosition, kneeLagging=False)
        FL_targetFootPosition = feetTrajectories[3][0]
        FL_targetJointPositions = self._calculateIK(FL_targetFootPosition, kneeLagging=True)

        ### Send commands to all motors over CAN bus ###
        self._sendMotorPosCanMsg(self.FR_HIP_ABDUCTOR_ADDUCTOR_MOTOR, position=FR_targetJointPositions[0])
        self._sendMotorPosCanMsg(self.FR_HIP_FLEXOR_EXTENDOR_MOTOR, position=FR_targetJointPositions[1])
        self._sendMotorPosCanMsg(self.FR_KNEE_MOTOR, position=2*FR_targetJointPositions[2]) # 2:1 pulley ratio

        self._sendMotorPosCanMsg(self.RL_HIP_ABDUCTOR_ADDUCTOR_MOTOR, position=RL_targetJointPositions[0])
        self._sendMotorPosCanMsg(self.RL_HIP_FLEXOR_EXTENDOR_MOTOR, position=-RL_targetJointPositions[1])
        self._sendMotorPosCanMsg(self.RL_KNEE_JOINT, position=2 * RL_targetJointPositions[2]) # 2:1 pulley ratio

        self._sendMotorPosCanMsg(self.RR_HIP_ABDUCTOR_ADDUCTOR_MOTOR, position=-RR_targetJointPositions[0])
        self._sendMotorPosCanMsg(self.RR_HIP_FLEXOR_EXTENDOR_MOTOR, position=RR_targetJointPositions[1])
        self._sendMotorPosCanMsg(self.RR_KNEE_MOTOR, position=2*RR_targetJointPositions[2]) # 2:1 pulley ratio

        self._sendMotorPosCanMsg(self.FL_HIP_ABDUCTOR_ADDUCTOR_MOTOR, position=-FL_targetJointPositions[0])
        self._sendMotorPosCanMsg(self.FL_HIP_FLEXOR_EXTENDOR_MOTOR, position=-FL_targetJointPositions[1])
        self._sendMotorPosCanMsg(self.FL_KNEE_MOTOR, position=2*FL_targetJointPositions[2]) # 2:1 pulley ratio

        time.sleep(3)

        trotStage = 0
        while True:

            if trotStage==0:
                latStanceSign = 1
            elif trotStage==1:
                latStanceSign = -1

            latStanceLength = latStanceSign*latStanceLength

            feetTrajectories = self._genTrotStep(forwardSwingLength, latSwingLength, swingHeight, latStanceLength,
                                                      T_swing, h_0, sampleRate, swingX_0, stanceX_0, swing1Y_0,
                                                      swing2Y_0, stance1Y_0, stance2Y_0, hipOffset=hipOffset)

            samplesPerSwing = feetTrajectories[0].shape[0]

            for j in range(samplesPerSwing):

                startTime = time.perf_counter()

                if trotStage==0:
                    k = 0
                elif trotStage==1:
                    k = 2
                FR_targetFootPosition = feetTrajectories[k][j]
                FR_targetJointPositions = self._calculateIK(FR_targetFootPosition, kneeLagging=True)

                if trotStage == 0:
                    k = 1
                elif trotStage == 1:
                    k = 3
                RL_targetFootPosition = feetTrajectories[k][j]
                RL_targetJointPositions = self._calculateIK(RL_targetFootPosition, kneeLagging=False)

                if trotStage == 0:
                    k = 2
                elif trotStage == 1:
                    k = 0
                RR_targetFootPosition = feetTrajectories[k][j]
                RR_targetJointPositions = self._calculateIK(RR_targetFootPosition, kneeLagging=False)

                if trotStage == 0:
                    k = 3
                elif trotStage == 1:
                    k = 1
                FL_targetFootPosition = feetTrajectories[k][j]
                FL_targetJointPositions = self._calculateIK(FL_targetFootPosition, kneeLagging=True)

                self._sendMotorPosCanMsg(self.FR_HIP_ABDUCTOR_ADDUCTOR_MOTOR, position=FR_targetJointPositions[0])
                self._sendMotorPosCanMsg(self.FR_HIP_FLEXOR_EXTENDOR_MOTOR, position=FR_targetJointPositions[1])
                self._sendMotorPosCanMsg(self.FR_KNEE_MOTOR, position=2*FR_targetJointPositions[2])

                self._sendMotorPosCanMsg(self.RL_HIP_ABDUCTOR_ADDUCTOR_MOTOR, position=RL_targetJointPositions[0])
                self._sendMotorPosCanMsg(self.RL_HIP_FLEXOR_EXTENDOR__MOTOR, position=-RL_targetJointPositions[1])
                self._sendMotorPosCanMsg(self.RL_KNEE_MOTOR, position=2*RL_targetJointPositions[2])

                self._sendMotorPosCanMsg(self.RR_HIP_ABDUCTOR_ADDUCTOR_MOTOR, position=-RR_targetJointPositions[0])
                self._sendMotorPosCanMsg(self.RR_HIP_FLEXOR_EXTENDOR_MOTOR, position=RR_targetJointPositions[1])
                self._sendMotorPosCanMsg(self.RR_KNEE_MOTOR, position=2*RR_targetJointPositions[2])

                self._sendMotorPosCanMsg(self.FL_HIP_ABDUCTOR_ADDUCTOR_MOTOR, position=-FL_targetJointPositions[0])
                self._sendMotorPosCanMsg(self.FL_HIP_FLEXOR_EXTENDOR_MOTOR, position=-FL_targetJointPositions[1])
                self._sendMotorPosCanMsg(self.FL_KNEE_MOTOR, position=2*FL_targetJointPositions[2])

                elapsedTime = time.perf_counter() - startTime

                while elapsedTime < T_swing/sampleRate:
                    elapsedTime = time.perf_counter() - startTime

            swingX_0 = feetTrajectories[2][samplesPerSwing-1][0]
            stanceX_0 = feetTrajectories[0][samplesPerSwing-1][0]
            swing1Y_0 = feetTrajectories[2][samplesPerSwing-1][1]- hipOffset
            swing2Y_0 = feetTrajectories[3][samplesPerSwing-1][1] - hipOffset
            stance1Y_0 = feetTrajectories[0][samplesPerSwing-1][1] - hipOffset
            stance2Y_0 = feetTrajectories[1][samplesPerSwing-1][1] - hipOffset

            trotStage = trotStage + 1
            if trotStage == 2:
                trotStage = 0


    def runWalkGate(self, forwardSwingLength=0.08, latSwingLength=0, swingHeight=0.04, leanLength=0.05,
                    h_0=-0.28, T_swing=0.5, T_lean=0.5, sampleRate=500, backSwingCoef=0.2, hipOffset=0.06):
        """""
        This method runs the walk gait on the physical PAWS robot, with two additional lean periods each cycle. 
        Arguments determine the step parameters, which are then kept constant.
        Args:
            forwardSwingLength (float): Length of swing in x-direction relative to hip (negative goes backwards).
            latSwingLength (float): Length of swing in y-direction relative to hip (negative goes inwards).
            swingHeight (float): Maximum height of swing phase relative to the stance height. 
           leanLength (float): Maximum y-lean during lean phases.
            h_0 (float): z-position of feet relative to hip during the stance phase (i.e. - body height).
            T_swing (float): Swing period.
            T_lean (float): Lean period.
            sampleRate (float): Sample rate, recommended 240 Hz.
            backSwingCoef (float): back/forward swing coefficient.
            hipOffset (float): 0 m for feet directly below motors, L1 m for legs straight (recommended).
        Returns:
            None: Program continues in an infinite loop until is ended.
        """""

        ### Configure CAN bus settings ###
        os.system('sudo ifconfig can0 down')
        os.system('sudo ip link set can0 type can bitrate 1000000')
        os.system('sudo ifconfig can0 txqueuelen 100000')
        os.system('sudo ifconfig can0 up')

        # Create CAN bus object
        bus = can.Bus(interface='socketcan', channel='can0', bitrate=1000000)

        ### Define initial foot positions ###
        swingX = -forwardSwingLength / 2
        swingY = 0
        stance1X = -forwardSwingLength / 6
        stance1Y = 0
        stance2X = forwardSwingLength / 6
        stance2Y = 0
        stance3X = forwardSwingLength / 2
        stance3Y = 0

        leanDirection = False  # Define boolean to change lean direction every lean phase

        # Generate foot trajectories for the first swing phase
        leanTrajectory = self._genLeanOutTrajectory(leanLength, T_lean, h_0, sampleRate, hipOffset, leanDirection,
                                                    swingX, swingY, stance1X, stance1Y, stance2X, stance2Y, stance3X,
                                                    stance3Y)
        samplesPerLean = leanTrajectory[0].shape[0]  # Determine the number of samples per swing

        ### Calculate initial joint positions for all legs using IK ###
        FR_targetFootPosition = leanTrajectory[0][0]
        FR_targetJointPositions = self._calculateIK(FR_targetFootPosition, kneeLagging=True, L3=0.23)

        RL_targetFootPosition = leanTrajectory[1][0]
        RL_targetJointPositions = self._calculateIK(RL_targetFootPosition, kneeLagging=True, L3=0.23)

        FL_targetFootPosition = leanTrajectory[2][0]
        FL_targetJointPositions = self._calculateIK(FL_targetFootPosition, kneeLagging=True, L3=0.23)

        RR_targetFootPosition = leanTrajectory[3][0]
        RR_targetJointPositions = self._calculateIK(RR_targetFootPosition, kneeLagging=True, L3=0.23)

        ### Send commands to all motors ###
        self._sendMotorPosCanMsg(self.FR_HIP_ABDUCTOR_ADDUCTOR_MOTOR, position=FR_targetJointPositions[0])
        self._sendMotorPosCanMsg(self.FR_HIP_FLEXOR_EXTENDOR_MOTOR, position=FR_targetJointPositions[1])
        self._sendMotorPosCanMsg(self.FR_KNEE_MOTOR, position=2 * FR_targetJointPositions[2])

        self._sendMotorPosCanMsg(self.RL_HIP_ABDUCTOR_ADDUCTOR_MOTOR, position=-RL_targetJointPositions[0])
        self._sendMotorPosCanMsg(self.RL_HIP_FLEXOR_EXTENDOR_MOTOR, position=RL_targetJointPositions[1])
        self._sendMotorPosCanMsg(self.RL_KNEE_JOINT, position=2 * RL_targetJointPositions[2])

        self._sendMotorPosCanMsg(self.FL_HIP_ABDUCTOR_ADDUCTOR_MOTOR, position=-FL_targetJointPositions[0])
        self._sendMotorPosCanMsg(self.FL_HIP_FLEXOR_EXTENDOR_MOTOR, position=-FL_targetJointPositions[1])
        self._sendMotorPosCanMsg(self.FL_KNEE_MOTOR, position=2 * FL_targetJointPositions[2])

        self._sendMotorPosCanMsg(self.RR_HIP_ABDUCTOR_ADDUCTOR_MOTOR, position=-RR_targetJointPositions[0])
        self._sendMotorPosCanMsg(self.RR_HIP_FLEXOR_EXTENDOR_MOTOR, position=RR_targetJointPositions[1])
        self._sendMotorPosCanMsg(self.RR_KNEE_MOTOR, position=2 * RR_targetJointPositions[2])

        time.sleep(5)

        startTime = time.perf_counter()
        for j in range(samplesPerLean):

            FR_targetFootPosition = leanTrajectory[0][j]
            FR_targetJointPositions = self._calculateIK(FR_targetFootPosition, kneeLagging=True, L3=0.23)

            RL_targetFootPosition = leanTrajectory[1][j]
            RL_targetJointPositions = self._calculateIK(RL_targetFootPosition, kneeLagging=True, L3=0.23)

            FL_targetFootPosition = leanTrajectory[2][j]
            FL_targetJointPositions = self._calculateIK(FL_targetFootPosition, kneeLagging=True, L3=0.23)

            RR_targetFootPosition = leanTrajectory[3][j]
            RR_targetJointPositions = self._calculateIK(RR_targetFootPosition, kneeLagging=True, L3=0.23)

            self._sendMotorPosCanMsg(self.FR_HIP_ABDUCTOR_ADDUCTOR_MOTOR, position=FR_targetJointPositions[0])
            self._sendMotorPosCanMsg(self.FR_HIP_FLEXOR_EXTENDOR_MOTOR, position=FR_targetJointPositions[1])
            self._sendMotorPosCanMsg(self.FR_KNEE_MOTOR, position=2 * FR_targetJointPositions[2])  # 2:1 pulley ratio

            self._sendMotorPosCanMsg(self.RL_HIP_ABDUCTOR_ADDUCTOR_MOTOR, position=-RL_targetJointPositions[0])
            self._sendMotorPosCanMsg(self.RL_HIP_FLEXOR_EXTENDOR_MOTOR, position=RL_targetJointPositions[1])
            self._sendMotorPosCanMsg(self.RL_KNEE_MOTOR, position=2 * RL_targetJointPositions[2])  # 2:1 pulley ratio

            self._sendMotorPosCanMsg(self.FL_HIP_ABDUCTOR_ADDUCTOR_MOTOR, position=-FL_targetJointPositions[0])
            self._sendMotorPosCanMsg(self.FL_HIP_FLEXOR_EXTENDOR_MOTOR, position=-FL_targetJointPositions[1])
            self._sendMotorPosCanMsg(self.FL_KNEE_MOTOR, position=2 * FL_targetJointPositions[2])  # 2:1 pulley ratio

            self._sendMotorPosCanMsg(self.RR_HIP_ABDUCTOR_ADDUCTOR_MOTOR, position=-RR_targetJointPositions[0])
            self._sendMotorPosCanMsg(self.RR_HIP_FLEXOR_EXTENDOR_MOTOR, position=RR_targetJointPositions[1])
            self._sendMotorPosCanMsg(self.RR_KNEE_MOTOR, position=2 * RR_targetJointPositions[2])  # 2:1 pulley ratio

            elapsedTime = time.perf_counter() - startTime

            while elapsedTime < T_lean / sampleRate:
                elapsedTime = time.perf_counter() - startTime

            startTime = time.perf_counter()

        swingX_0 = leanTrajectory[0][samplesPerLean - 1][0]
        swingY_0 = leanTrajectory[0][samplesPerLean - 1][1] - hipOffset
        stance1X_0 = leanTrajectory[3][samplesPerLean - 1][0]
        stance1Y_0 = leanTrajectory[3][samplesPerLean - 1][1] - hipOffset
        stance2X_0 = leanTrajectory[2][samplesPerLean - 1][0]
        stance2Y_0 = leanTrajectory[2][samplesPerLean - 1][1] - hipOffset
        stance3X_0 = leanTrajectory[1][samplesPerLean - 1][0]
        stance3Y_0 = leanTrajectory[1][samplesPerLean - 1][1] - hipOffset

        feetTrajectories = self._genWalkStep(forwardSwingLength, latSwingLength, swingHeight, T_swing, h_0, sampleRate,
                                             swingX_0, swingY_0, stance1X_0, stance1Y_0, stance2X_0, stance2Y_0,
                                             stance3X_0, stance3Y_0, backSwingCoef, hipOffset, phase=0)
        samplesPerSwing = feetTrajectories[0].shape[0]

        i = 0
        l = 0
        startTime = time.perf_counter()
        while True:

            if i == 1 or i == 3:

                for j in range(samplesPerLean):

                    FR_targetFootPosition = leanTrajectory[0][j]
                    FR_targetJointPositions = self._calculateIK(FR_targetFootPosition, kneeLagging=True, L3=0.23)

                    RL_targetFootPosition = leanTrajectory[1][j]
                    RL_targetJointPositions = self._calculateIK(RL_targetFootPosition, kneeLagging=True, L3=0.23)

                    FL_targetFootPosition = leanTrajectory[2][j]
                    FL_targetJointPositions = self._calculateIK(FL_targetFootPosition, kneeLagging=True, L3=0.23)

                    RR_targetFootPosition = leanTrajectory[3][j]
                    RR_targetJointPositions = self._calculateIK(RR_targetFootPosition, kneeLagging=True, L3=0.23)

                    self._sendMotorPosCanMsg(self.FR_HIP_ABDUCTOR_ADDUCTOR_MOTOR, position=FR_targetJointPositions[0])
                    self._sendMotorPosCanMsg(self.FR_HIP_FLEXOR_EXTENDOR_MOTOR, position=FR_targetJointPositions[1])
                    self._sendMotorPosCanMsg(self.FR_KNEE_MOTOR, position=2 * FR_targetJointPositions[2])  # 2:1 pulley ratio

                    self._sendMotorPosCanMsg(self.RL_HIP_ABDUCTOR_ADDUCTOR_MOTOR, position=-RL_targetJointPositions[0])
                    self._sendMotorPosCanMsg(self.RL_HIP_FLEXOR_EXTENDOR_MOTOR, position=RL_targetJointPositions[1])
                    self._sendMotorPosCanMsg(self.RL_KNEE_MOTOR, position=2 * RL_targetJointPositions[2])  # 2:1 pulley ratio

                    self._sendMotorPosCanMsg(self.FL_HIP_ABDUCTOR_ADDUCTOR_MOTOR, position=-FL_targetJointPositions[0])
                    self._sendMotorPosCanMsg(self.FL_HIP_FLEXOR_EXTENDOR_MOTOR, position=-FL_targetJointPositions[1])
                    self._sendMotorPosCanMsg(self.FL_KNEE_MOTOR, position=2 * FL_targetJointPositions[2])  # 2:1 pulley ratio

                    self._sendMotorPosCanMsg(self.RR_HIP_ABDUCTOR_ADDUCTOR_MOTOR, position=-RR_targetJointPositions[0])
                    self._sendMotorPosCanMsg(self.RR_HIP_FLEXOR_EXTENDOR_MOTOR, position=RR_targetJointPositions[1])
                    self._sendMotorPosCanMsg(self.RR_KNEE_MOTOR, position=2 * RR_targetJointPositions[2])  # 2:1 pulley ratio

                    elapsedTime = time.perf_counter() - startTime

                    while elapsedTime < T_lean / sampleRate:
                        elapsedTime = time.perf_counter() - startTime

                    startTime = time.perf_counter()

            for j in range(samplesPerSwing):

                FR_targetFootPosition = feetTrajectories[i][j]
                FR_targetJointPositions = self._calculateIK(FR_targetFootPosition, kneeLagging=True, L3=0.23)

                if i + 3 <= 3:
                    k = i + 3
                else:
                    k = i + 3 - 4
                RL_targetFootPosition = feetTrajectories[k][j]
                RL_targetJointPositions = self._calculateIK(RL_targetFootPosition, kneeLagging=True, L3=0.23)

                if i + 2 <= 3:
                    k = i + 2
                else:
                    k = i + 2 - 4
                FL_targetFootPosition = feetTrajectories[k][j]
                FL_targetJointPositions = self._calculateIK(FL_targetFootPosition, kneeLagging=True, L3=0.23)

                if i + 1 <= 3:
                    k = i + 1
                else:
                    k = i + 1 - 4
                RR_targetFootPosition = feetTrajectories[k][j]
                RR_targetJointPositions = self._calculateIK(RR_targetFootPosition, kneeLagging=True, L3=0.23)

                self._sendMotorPosCanMsg(self.FR_HIP_ABDUCTOR_ADDUCTOR_MOTOR, position=FR_targetJointPositions[0])
                self._sendMotorPosCanMsg(self.FR_HIP_FLEXOR_EXTENDOR_MOTOR, position=FR_targetJointPositions[1])
                self._sendMotorPosCanMsg(self.FR_KNEE_MOTOR, position=2 * FR_targetJointPositions[2]) # 2:1 pulley ratio

                self._sendMotorPosCanMsg(self.RL_HIP_ABDUCTOR_ADDUCTOR_MOTOR, position=-RL_targetJointPositions[0])
                self._sendMotorPosCanMsg(self.RL_HIP_FLEXOR_EXTENDOR_MOTOR, position=RL_targetJointPositions[1])
                self._sendMotorPosCanMsg(self.RL_KNEE_MOTOR, position=2 * RL_targetJointPositions[2]) # 2:1 pulley ratio

                self._sendMotorPosCanMsg(self.FL_HIP_ABDUCTOR_ADDUCTOR_MOTOR, position=-FL_targetJointPositions[0])
                self._sendMotorPosCanMsg(self.FL_HIP_FLEXOR_EXTENDOR_MOTOR, position=-FL_targetJointPositions[1])
                self._sendMotorPosCanMsg(self.FL_KNEE_MOTOR, position=2 * FL_targetJointPositions[2]) # 2:1 pulley ratio

                self._sendMotorPosCanMsg(self.RR_HIP_ABDUCTOR_ADDUCTOR_MOTOR, position=-RR_targetJointPositions[0])
                self._sendMotorPosCanMsg(self.RR_HIP_FLEXOR_EXTENDOR_MOTOR, position=RR_targetJointPositions[1])
                self._sendMotorPosCanMsg(self.RR_KNEE_MOTOR, position=2 * RR_targetJointPositions[2]) # 2:1 pulley ratio

                elapsedTime = time.perf_counter() - startTime

                while elapsedTime < T_swing / sampleRate:
                    elapsedTime = time.perf_counter() - startTime

                startTime = time.perf_counter()

            swingX_0 = feetTrajectories[3][samplesPerSwing - 1][0]
            swingY_0 = feetTrajectories[3][samplesPerSwing - 1][1] - hipOffset
            stance1X_0 = feetTrajectories[0][samplesPerSwing - 1][0]
            stance1Y_0 = feetTrajectories[0][samplesPerSwing - 1][1] - hipOffset
            stance2X_0 = feetTrajectories[1][samplesPerSwing - 1][0]
            stance2Y_0 = feetTrajectories[1][samplesPerSwing - 1][1] - hipOffset
            stance3X_0 = feetTrajectories[2][samplesPerSwing - 1][0]
            stance3Y_0 = feetTrajectories[2][samplesPerSwing - 1][1] - hipOffset

            l = l + 1
            if l > 4:
                l = 4

            i = i + 1
            if i == 4:
                i = 0

            if i == 0 or i == 3:
                latSwingSignCorrection = 1
            elif i == 1 or i == 2:
                latSwingSignCorrection = -1

            if l == 1:
                swingLenCorrection = 1  # 0.3
            elif l == 2:
                swingLenCorrection = 1  # 0.55
            elif l == 3:
                swingLenCorrection = 1  # 0.8
            else:
                swingLenCorrection = 1

            if i == 1:

                leanDirection = True
                leanTrajectory = self._genLeanInTrajectory(2 * leanLength, T_lean, h_0, sampleRate, hipOffset,
                                                                leanDirection, stance1X_0, stance1Y_0, swingX_0,
                                                                swingY_0, stance3X_0, stance3Y_0, stance2X_0,
                                                                stance2Y_0)
                samplesPerLean = leanTrajectory[0].shape[0]

                swingX_0 = leanTrajectory[1][samplesPerLean - 1][0]
                swingY_0 = leanTrajectory[1][samplesPerLean - 1][1] - hipOffset
                stance1X_0 = leanTrajectory[0][samplesPerLean - 1][0]
                stance1Y_0 = leanTrajectory[0][samplesPerLean - 1][1] - hipOffset
                stance2X_0 = leanTrajectory[3][samplesPerLean - 1][0]
                stance2Y_0 = leanTrajectory[3][samplesPerLean - 1][1] - hipOffset
                stance3X_0 = leanTrajectory[2][samplesPerLean - 1][0]
                stance3Y_0 = leanTrajectory[2][samplesPerLean - 1][1] - hipOffset

            elif i == 3:

                leanDirection = False
                leanTrajectory = self._genLeanOutTrajectory(2 * leanLength, T_lean, h_0, sampleRate, hipOffset,
                                                                 leanDirection, stance3X_0, stance3Y_0, stance2X_0,
                                                                 stance2Y_0, stance1X_0, stance1Y_0, swingX_0,
                                                                 swingY_0)
                samplesPerLean = leanTrajectory[0].shape[0]

                swingX_0 = leanTrajectory[3][samplesPerLean - 1][0]
                swingY_0 = leanTrajectory[3][samplesPerLean - 1][1] - hipOffset
                stance1X_0 = leanTrajectory[2][samplesPerLean - 1][0]
                stance1Y_0 = leanTrajectory[2][samplesPerLean - 1][1] - hipOffset
                stance2X_0 = leanTrajectory[1][samplesPerLean - 1][0]
                stance2Y_0 = leanTrajectory[1][samplesPerLean - 1][1] - hipOffset
                stance3X_0 = leanTrajectory[0][samplesPerLean - 1][0]
                stance3Y_0 = leanTrajectory[0][samplesPerLean - 1][1] - hipOffset

            feetTrajectories = self._genWalkStep(forwardSwingLength, latSwingLength, swingHeight, T_swing, h_0,
                                                       sampleRate, swingX_0, swingY_0, stance1X_0, stance1Y_0,
                                                       stance2X_0, stance2Y_0, stance3X_0, stance3Y_0,
                                                       hipOffset=hipOffset, phase=i)

            samplesPerSwing = feetTrajectories[0].shape[0]


    def runWalkGateNoAdditionalLean(self, forwardSwingLength, latSwingLength, swingHeight, latStanceLength, T_swing,
                                    h_0, sampleRate=500, backSwingCoef=0.2, hipOffset=0.06):
        """""
        This method runs the walk gait on the physical PAWS robot, with no additional lean periods but instead leaning 
        while one foot is swinging. Arguments determine the step parameters, which are then kept constant.
        Args:
            forwardSwingLength (float): Length of swing in x-direction relative to hip (negative goes backwards).
            latSwingLength (float): Length of swing in y-direction relative to hip (negative goes inwards).
            swingHeight (float): Maximum height of swing phase relative to the stance height. 
            latStanceLength (float): Maximum y-lean during stance phase.
            T_swing (float): Swing period.
            h_0 (float): z-position of feet relative to hip during the stance phase (i.e. - body height).
            sampleRate (float): Sample rate, recommended 240 Hz.
            backSwingCoef (float): back/forward swing coefficient.
            hipOffset (float): 0 m for feet directly below motors, L1 m for legs straight (recommended).
        Returns:
            None: Program continues in an infinite loop until is ended.
        """""

        ### Configure CAN bus settings ###
        os.system('sudo ifconfig can0 down')
        os.system('sudo ip link set can0 type can bitrate 1000000')
        os.system('sudo ifconfig can0 txqueuelen 100000')
        os.system('sudo ifconfig can0 up')

        # Create CAN bus object
        bus = can.Bus(interface='socketcan', channel='can0', bitrate=1000000)

        ### Define initial foot positions ###
        swingX = -forwardSwingLength / 2
        swingY = 0
        stance1X = -forwardSwingLength / 6
        stance1Y = 0
        stance2X = forwardSwingLength / 6
        stance2Y = 0
        stance3X = forwardSwingLength / 2
        stance3Y = 0

        feetTrajectories = self._genWalkStepNoAdditionalLean(forwardSwingLength, latSwingLength, swingHeight, T_swing, h_0, sampleRate,
                                             swingX, swingY, stance1X, stance1Y, stance2X, stance2Y, stance3X,
                                             stance3Y, backSwingCoef, hipOffset=hipOffset, stage=0)

        FR_targetFootPosition = feetTrajectories[0][0]
        FR_targetJointPositions = self._calculateIK(FR_targetFootPosition, kneeLagging=True)
        RL_targetFootPosition = feetTrajectories[3][0]
        RL_targetJointPositions = self._calculateIK(RL_targetFootPosition, kneeLagging=False)
        FL_targetFootPosition = feetTrajectories[2][0]
        FL_targetJointPositions = self._calculateIK(FL_targetFootPosition, kneeLagging=True)
        RR_targetFootPosition = feetTrajectories[1][0]
        RR_targetJointPositions = self._calculateIK(RR_targetFootPosition, kneeLagging=False)

        self._sendMotorPosCanMsg(self.FR_HIP_ABDUCTOR_ADDUCTOR_MOTOR, position=FR_targetJointPositions[0])
        self._sendMotorPosCanMsg(self.FR_HIP_FLEXOR_EXTENDOR_MOTOR, position=FR_targetJointPositions[1])
        self._sendMotorPosCanMsg(self.FR_KNEE_MOTOR, position=2 * FR_targetJointPositions[2])

        self._sendMotorPosCanMsg(self.RL_HIP_ABDUCTOR_ADDUCTOR_MOTOR, position=RL_targetJointPositions[0])
        self._sendMotorPosCanMsg(self.RL_HIP_FLEXOR_EXTENDOR_MOTOR, position=-RL_targetJointPositions[1])
        self._sendMotorPosCanMsg(self.RL_KNEE_JOINT, position=2 * RL_targetJointPositions[2])

        self._sendMotorPosCanMsg(self.FL_HIP_ABDUCTOR_ADDUCTOR_MOTOR, position=-FL_targetJointPositions[0])
        self._sendMotorPosCanMsg(self.FL_HIP_FLEXOR_EXTENDOR_MOTOR, position=-FL_targetJointPositions[1])
        self._sendMotorPosCanMsg(self.FL_KNEE_MOTOR, position=2 * FL_targetJointPositions[2])

        self._sendMotorPosCanMsg(self.RR_HIP_ABDUCTOR_ADDUCTOR_MOTOR, position=-RR_targetJointPositions[0])
        self._sendMotorPosCanMsg(self.RR_HIP_FLEXOR_EXTENDOR_MOTOR, position=RR_targetJointPositions[1])
        self._sendMotorPosCanMsg(self.RR_KNEE_MOTOR, position=2 * RR_targetJointPositions[2])

        time.sleep(3)

        i = 0
        while True:

            if i == 0 or i == 3:
                latSwingSignCorrection = 1
                latStanceSignCorrection = 1
            elif i == 1 or i == 2:
                latSwingSignCorrection = -1
                latStanceSignCorrection = -1

            forwardSwingLength = forwardSwingLengthIn
            latSwingLength = latSwingSignCorrection * latSwingLengthIn
            swingHeight = swingHeightIn
            latStanceLength = latStanceSignCorrection * latStanceLengthIn
            T_swing = T_swing_In
            h_0 = h_0_In

            feetTrajectories = self._genWalkStepNoAdditionalLean(forwardSwingLength, latSwingLength, swingHeight, latStanceLength,
                                                      T_swing, h_0, sampleRate, swingX_0, swingY_0, stance1X_0,
                                                      stance1Y_0, stance2X_0, stance2Y_0, stance3X_0, stance3Y_0,
                                                      hipOffset=hipOffset, stage=i)

            samplesPerSwing = feetTrajectories[0].shape[0]

            for j in range(samplesPerSwing):

                startTime = time.perf_counter()

                FR_targetFootPosition = feetTrajectories[i][j]
                FR_targetJointPositions = self._calculateIK(FR_targetFootPosition, kneeLagging=True)

                if i + 3 <= 3:
                    k = i + 3
                else:
                    k = i + 3 - 4
                RL_targetFootPosition = feetTrajectories[k][j]
                RL_targetJointPositions = self._calculateIK(RL_targetFootPosition, kneeLagging=False)

                if i + 2 <= 3:
                    k = i + 2
                else:
                    k = i + 2 - 4
                FL_targetFootPosition = feetTrajectories[k][j]
                FL_targetJointPositions = self._calculateIK(FL_targetFootPosition, kneeLagging=True)

                if i + 1 <= 3:
                    k = i + 1
                else:
                    k = i + 1 - 4
                RR_targetFootPosition = feetTrajectories[k][j]
                RR_targetJointPositions = self._calculateIK(RR_targetFootPosition, kneeLagging=False)

                self._sendMotorPosCanMsg(self.FR_HIP_ABDUCTOR_ADDUCTOR_MOTOR, position=FR_targetJointPositions[0])
                self._sendMotorPosCanMsg(self.FR_HIP_FLEXOR_EXTENDOR_MOTOR, position=FR_targetJointPositions[1])
                self._sendMotorPosCanMsg(self.FR_KNEE_MOTOR, position=2*FR_targetJointPositions[2])

                self._sendMotorPosCanMsg(self.RL_HIP_ABDUCTOR_ADDUCTOR_MOTOR, position=RL_targetJointPositions[0])
                self._sendMotorPosCanMsg(self.RL_HIP_FLEXOR_EXTENDOR__MOTOR, position=-RL_targetJointPositions[1])
                self._sendMotorPosCanMsg(self.RL_KNEE_MOTOR, position=2*RL_targetJointPositions[2])

                self._sendMotorPosCanMsg(self.FL_HIP_ABDUCTOR_ADDUCTOR_MOTOR, position=-FL_targetJointPositions[0])
                self._sendMotorPosCanMsg(self.FL_HIP_FLEXOR_EXTENDOR_MOTOR, position=-FL_targetJointPositions[1])
                self._sendMotorPosCanMsg(self.FL_KNEE_MOTOR, position=2*FL_targetJointPositions[2])

                self._sendMotorPosCanMsg(self.RR_HIP_ABDUCTOR_ADDUCTOR_MOTOR, position=-RR_targetJointPositions[0])
                self._sendMotorPosCanMsg(self.RR_HIP_FLEXOR_EXTENDOR_MOTOR, position=RR_targetJointPositions[1])
                self._sendMotorPosCanMsg(self.RR_KNEE_MOTOR, position=2*RR_targetJointPositions[2])

                elapsedTime = time.perf_counter() - startTime

                while elapsedTime < T_swing / sampleRate:
                    elapsedTime = time.perf_counter() - startTime

            swingX_0 = feetTrajectories[3][samplesPerSwing - 1][0]
            swingY_0 = feetTrajectories[3][samplesPerSwing - 1][1] - hipOffset
            stance1X_0 = feetTrajectories[0][samplesPerSwing - 1][0]
            stance1Y_0 = feetTrajectories[0][samplesPerSwing - 1][1] - hipOffset
            stance2X_0 = feetTrajectories[1][samplesPerSwing - 1][0]
            stance2Y_0 = feetTrajectories[1][samplesPerSwing - 1][1] - hipOffset
            stance3X_0 = feetTrajectories[2][samplesPerSwing - 1][0]
            stance3Y_0 = feetTrajectories[2][samplesPerSwing - 1][1] - hipOffset

            i = i + 1
            if i == 4:
                i = 0


    def _createSimEnvironment(self, type="FLAT", height_mean=0.1, height_std_dev=0.01, xy_scale=0.05, stepLen=0.2,
                             stepWidth=2, stepHeight=0.02):
        """""
        This method creates a simulation environment with the specified terrain type and parameters
        Args:
            type (str): terrain type, which is asserted in ["FLAT", "ROUGH", "STAIRS"]
            height_mean (float): mean height of terrain if terrain==ROUGH.
            height_std_dev (float): standard deviation of terrain if terrain==ROUGH.
            xy_scale (float): how big each square is if terrain==ROUGH.
            stepLen (float): length in x-direction of each step if terrain==STAIRS.
            stepWidth (float): width in y-direction of each step if terrain==STAIRS.
            stepHeight(float): height in z-direction of each step if terrain==STAIRS.
        Returns:
            None
        """""

        fig = plt.figure(1)   # Calling this first makes the GUI higher resolution
        p.connect(p.GUI)      # Connect to PyBullet in render mode
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        if type == "RANDOM_ROUGH":

            random.seed(10)  # Set random seed so that the rough terrain is the same each time to allow fair comparison
                             # between gaits
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # Configure debug visualiser

            ### Set the size of the heightfield ###
            numHeightfieldRows = 256
            numHeightfieldColumns = 256
            heightfieldData = [0] * numHeightfieldRows * numHeightfieldColumns

            ### Create Gaussian distributed heightfield ###
            for j in range(int(numHeightfieldColumns / 2)):
                for i in range(int(numHeightfieldRows / 2)):
                    height = np.random.normal(height_mean, height_std_dev)
                    heightfieldData[2 * i + 2 * j * numHeightfieldRows] = height
                    heightfieldData[2 * i + 1 + 2 * j * numHeightfieldRows] = height
                    heightfieldData[2 * i + (2 * j + 1) * numHeightfieldRows] = height
                    heightfieldData[2 * i + 1 + (2 * j + 1) * numHeightfieldRows] = height

            ### Create collision object and render it ###
            terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD, meshScale=[xy_scale, xy_scale, 1],
                                                  heightfieldTextureScaling=(numHeightfieldRows - 1) / 2,
                                                  heightfieldData=heightfieldData,
                                                  numHeightfieldRows=numHeightfieldRows,
                                                  numHeightfieldColumns=numHeightfieldColumns)

            terrain = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=terrainShape,
                                        basePosition=[0, 0, height_mean])
            p.changeVisualShape(terrain, -1, rgbaColor=[1, 1, 1, 1])
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        elif type == "STAIRS":

            boxHalfLength = 0.5 * stepLen     # Half length of step
            boxHalfWidth = 0.5 * stepWidth    # Half width of step
            boxHalfHeight = 0.5 * stepHeight  # Half height of step

            ### Create 5 ascending steps and 4 descending steps ###
            for i in range(5):
                colShape = p.createCollisionShape(p.GEOM_BOX,
                                                  halfExtents=[boxHalfLength, boxHalfWidth, (1 + i) * boxHalfHeight])
                visShape = p.createVisualShape(p.GEOM_BOX,
                                               halfExtents=[boxHalfLength, boxHalfWidth, (1 + i) * boxHalfHeight])
                step = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colShape,
                                         basePosition=[0.5 + 2 * i * boxHalfLength, 0, (1 + i) * boxHalfHeight])
            for i in range(5):
                colShape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[boxHalfLength, boxHalfWidth,
                                                                           (1 + 4 - i) * boxHalfHeight])
                visShape = p.createVisualShape(p.GEOM_BOX,
                                               halfExtents=[boxHalfLength, boxHalfWidth, (1 + 4 - i) * boxHalfHeight])
                step = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colShape,
                                         basePosition=[0.5 + 2 * (i + 4) * boxHalfLength, 0,
                                                       (1 + 4 - i) * boxHalfHeight])

        elif type == "FLAT":
            pass  # Flat is just the default PyBullet plane


    def _calculateIK(self, targetFootPosition, kneeLagging, L1=0.06, L2=0.15, L3=0.21):
        """""
        This method returns the joint angles required to place a single foot at a desired position relative to the 
        corresponding hip. It implements the IK equations derived in the report and includes error checking for 
        positions that violate the workspace constraints.
        Args:
            targetFootPosition (list of float): Desired Cartesian position of foot relative to corresponding hip.
            kneeLagging (bool): There are two possible knee configurations for each desired foot position, which are 
                                knee leading the foot (bending in >‾‾>) or knee lagging the foot (bending out <‾‾<).
                                If kneeLagging is true then the knee will lag the foot, else it will lead it.
            L1 (float): Length of link1 in meters.
            L2 (float): Length of link2 (the upper leg limb) in meters.
            L3 (float): Length of link3 (the lower leg limb) in meters.
        Returns:
            list of float: the joint angles of the leg which are the hip abduction/adduction angle (theta_1), the hip 
                           flexion/extension angle (theta_2) and the knee flexion/extension angle (theta_3) 
        """""
        x = targetFootPosition[0]
        y = targetFootPosition[1]
        z = targetFootPosition[2]

        z_dash = -np.sqrt(z ** 2 + y ** 2 - L1 ** 2)

        # Check for workspace violation
        if np.any((abs(y) / np.sqrt(z ** 2 + y ** 2) < -1) | (abs(y) / np.sqrt(z ** 2 + y ** 2) > 1)):
            print("Workspace Violation!")
        alpha = np.arccos(np.clip(abs(y) / np.sqrt(z ** 2 + y ** 2), -1, 1)) # Clip to enforce workspace constraints

        # Check for workspace violation
        if np.any((L1 / np.sqrt(z ** 2 + y ** 2) < -1) | (L1 / np.sqrt(z ** 2 + y ** 2) > 1)):
            print("Workspace Violation!")
        beta = np.arccos(np.clip(L1 / np.sqrt(z ** 2 + y ** 2), -1, 1)) # Clip to enforce workspace constraints

        # Check for workspace violation
        if np.any((abs(x) / np.sqrt(x ** 2 + z_dash ** 2) < -1) | (abs(x) / np.sqrt(x ** 2 + z_dash ** 2) > 1)):
            print("Workspace Violation!")
        phi = np.arccos(np.clip(abs(x) / np.sqrt(x ** 2 + z_dash ** 2), -1, 1)) # Clip to enforce workspace constraints

        # Check for workspace violation
        if np.any(((x ** 2 + z_dash ** 2 + L2 ** 2 - L3 ** 2) / (2 * L2 * np.sqrt(x ** 2 + z_dash ** 2)) < -1) | (
                (x ** 2 + z_dash ** 2 + L2 ** 2 - L3 ** 2) / (2 * L2 * np.sqrt(x ** 2 + z_dash ** 2)) > 1)):
            print("Workspace Violation!")
        # Clip to enforce workspace constraints
        psi = np.arccos(
            np.clip((x ** 2 + z_dash ** 2 + L2 ** 2 - L3 ** 2) / (2 * L2 * np.sqrt(x ** 2 + z_dash ** 2)), -1, 1))

        if y >= 0:
            theta_1 = beta - alpha
        else:
            theta_1 = alpha + beta - np.pi

        if kneeLagging:
            if x >= 0:
                theta_2 = np.pi / 2 - psi - phi
            else:
                theta_2 = -np.pi / 2 - psi + phi

            theta_3 = np.pi - np.arccos(np.clip((L2 ** 2 + L3 ** 2 - x ** 2 - z_dash ** 2) / (2 * L2 * L3), -1, 1))
        else:
            if x >= 0:
                theta_2 = np.pi / 2 + psi - phi
            else:
                theta_2 = -np.pi / 2 + psi + phi

            theta_3 = - np.pi + np.arccos(
                np.clip((L2 ** 2 + L3 ** 2 - x ** 2 - z_dash ** 2) / (2 * L2 * L3), -1, 1))

        targetJointPositions = np.array([theta_1, theta_2, theta_3])

        return targetJointPositions


    def _genTrotStep(self, forwardSwingLength=0.1, latSwingLength=0, swingHeight=0.08, latStanceLength=0, T_swing=1,
                    h_0=-0.25, sampleRate=240, swingX_0=-0.05, stanceX_0=0.05, swing1Y_0=0, swing2Y_0=0, stance1Y_0=0,
                    stance2Y_0=0, backSwingCoef=0.2, hipOffset=0.06):
        """""
        This method returns the trajectories of all four legs for one stage of the trot cycle, that is the swing 
        trajectory for 2 legs and the stance trajectory for the other 2 (which is proportional to the swing trajectory). 
        The trajectories are constructed using splines of quintic polynomials and a cycloid curve, as derived in the 
        report.
        Args:
            forwardSwingLength (float): Length of swing in x-direction relative to hip (negative goes backwards).
            latSwingLength (float): Length of swing in y-direction relative to hip (negative goes inwards).
            swingHeight (float): Maximum height of swing phase relative to the stance height. 
            latStanceLength (float): Maximum y-lean during stance phase. Recommended 0 m.
            T_swing (float): Swing period.
            h_0 (float): z-position of feet relative to hip during the stance phase (i.e. - body height).
            sampleRate (float): Sample rate, recommended 240 Hz.
            swingX_0 (float): Initial position of the swing feet in the x-direction (forwards/backwards)
            stanceX_0 (float): Initial position of the stance feet in the x-direction
            swing1Y_0 (float): Initial position of the swing foot 1 in the y-direction (sideways)
            swing2Y_0 (float): Initial position of the swing foot 2 in the y-direction
            stance1Y_0 (float): Initial position of the stance foot 1 in the y-direction
            stance2Y_0 (float): Initial position of the stance foot 2 in the y-direction
            backSwingCoef (float): back/forward swing coefficient.
            hipOffset (float): 0 m for feet directly below motors, L1 m for legs straight (recommended).
        Returns:
            np.ndarray: Foot trajectories for all feet as a 3D array of floats. The shape is (num_feet, num_samples, 3), 
                        where the last dimension is the Cartesian (x, y, z) position of the foot relative to the hip for 
                        that sample. The foot trajectories are returned in the order (swing foot 1, swing foot 2, stance 
                        foot 1, stance foot 2). Depending on the stage of the trot cycle, this may correspond to 
                        (FR, RL, FL, RR) or (FL, RR, FR, RL).
        """""
        ### Define boundary times as in the report ###
        t_1 = 0.25 * T_swing
        t_2 = 0.75 * T_swing
        t_m = 0.5 * T_swing
        samplesPerSwing = int(sampleRate * T_swing)

        ### Sample times for each stage of the x-trajectory, y-trajectory and z-trajectory ###
        t_backSwing = np.linspace(0, t_1, math.ceil(0.25 * samplesPerSwing))
        t_stride = np.linspace(t_1, t_2, math.ceil(0.5 * samplesPerSwing))
        t_forwardSwing = np.linspace(t_2, T_swing, math.ceil(0.25 * samplesPerSwing))
        t_lift = np.linspace(0, t_m, math.ceil(0.5 * samplesPerSwing))
        t_place = np.linspace(t_m, T_swing, math.ceil(0.5 * samplesPerSwing))
        t_stance = np.linspace(0, T_swing, samplesPerSwing)

        swingX_1 = swingX_0 - backSwingCoef * forwardSwingLength        # x-position after the backswing
        swingX_2 = swingX_0 + (1 + backSwingCoef) * forwardSwingLength  # x-position after the stride
        swingX_3 = swingX_0 + forwardSwingLength                        # x-position after the forward swing
        stanceX_2 = stanceX_0 - forwardSwingLength                      # x-position after the stance
        h = swingHeight

        T_x1 = t_1
        T_x2 = t_2 - t_1
        T_x3 = T_swing - t_2
        T_h1 = t_m
        T_h2 = T_swing - t_m

        ######################## Define x-axis trajectories for swing legs and stance legs ########################

        # Quintic polynomial, defined in report
        xBackSwing = swingX_0 + (20*swingX_1 - 20*swingX_0)/(2*T_x1**3) * np.power(t_backSwing, 3) \
                     + (30*swingX_0 - 30*swingX_1)/(2*T_x1**4) * np.power(t_backSwing, 4) \
                     + (12*swingX_1 - 12*swingX_0)/(2*T_x1**5) * np.power(t_backSwing, 5)

        # Cycloid curve , defined in report
        xStride = (swingX_2 - swingX_1)*((t_stride-t_1)/T_x2 - (1/(2*np.pi))*np.sin((2*np.pi*(t_stride-t_1)) / T_x2)) \
                  + swingX_1

        # Quintic polynomial, defined in report
        xForwardSwing = swingX_2 + (20*swingX_3 - 20*swingX_2)/(2*T_x3**3) * np.power(t_forwardSwing - t_2, 3) \
                        + (30*swingX_2 - 30*swingX_3)/(2*T_x3**4) * np.power(t_forwardSwing - t_2, 4) \
                        + (12*swingX_3 - 12*swingX_2)/(2*T_x3**5) * np.power(t_forwardSwing - t_2, 5)

        # Full x-direction swing trajectory
        x_swingTragectory = np.concatenate((xBackSwing, xStride, xForwardSwing), axis=0)

        # Full x-direction stance trajectory which is a quintic polynomial, defined in report
        x_stanceTragectory = stanceX_0 + (20*stanceX_2 - 20*stanceX_0)/(2*(T_swing)**3) * np.power(t_stance, 3) \
                             + (30*stanceX_0 - 30*stanceX_2)/(2*(T_swing)**4)*np.power(t_stance, 4) \
                             + (12*stanceX_2 - 12*stanceX_0)/(2*(T_swing)**5)*np.power(t_stance, 5)

        ######################## Define y-axis trajectories for swing leg and all 3 stance legs ########################

        # y-direction swing trajectory for swing 1 foot
        y_swing1Tragectory = swing1Y_0 + hipOffset + (20*latSwingLength)/(2*T_swing**3) * np.power(t_stance, 3) \
                             - (30*latSwingLength)/(2*T_swing**4) * np.power(t_stance, 4) \
                             + (12*latSwingLength)/(2*T_swing**5) * np.power(t_stance, 5)

        # y-direction swing trajectory for swing 2 foot, with sign correction to ensure same direction as swing 1
        y_swing2Tragectory = swing2Y_0 + hipOffset + (20*(-1)*latSwingLength)/(2*T_swing**3)*np.power(t_stance, 3) \
                             - (30*(-1)*latSwingLength)/(2*T_swing**4)*np.power(t_stance, 4) \
                             + (12*(-1)*latSwingLength)/(2*T_swing**5)*np.power(t_stance, 5)

        ### y-direction stance trajectory for stance 1 foot ###
        yStance1Out = stance1Y_0 + hipOffset + (20*latStanceLength)/(2*T_h1**3) * np.power(t_lift, 3) \
                      - (30*latStanceLength)/(2*T_h1**4) * np.power(t_lift, 4) \
                      + (12*latStanceLength)/(2*T_h1**5) * np.power(t_lift, 5)

        yStance1In = stance1Y_0 + hipOffset + latStanceLength \
                     - (20*latStanceLength)/(2*T_h2**3) * np.power(t_place - t_m, 3) \
                     + (30*latStanceLength)/(2*T_h2**4) * np.power(t_place - t_m, 4) \
                     - (12*latStanceLength)/(2*T_h2**5) * np.power(t_place - t_m, 5)

        y_stance1Tragectory = np.concatenate((yStance1Out, yStance1In), axis=0)

        ### y-direction stance trajectory for stance 2 foot, with sign correction to ensure same direction as stance 1 ###
        yStance2Out = stance2Y_0 + hipOffset + (20 * (-1) * latStanceLength) / (2 * T_h1 ** 3) * np.power(t_lift, 3) \
                      - (30 * (-1) * latStanceLength) / (2 * T_h1 ** 4) * np.power(t_lift, 4) \
                      + (12 * (-1) * latStanceLength) / (2 * T_h1 ** 5) * np.power(t_lift, 5)
        yStance2In = stance2Y_0 + hipOffset + (-1) * latStanceLength \
                     - (20 * (-1) * latStanceLength) / (2 * T_h2 ** 3) * np.power(t_place - t_m, 3) \
                     + (30 * (-1) * latStanceLength) / (2 * T_h2 ** 4) * np.power(t_place - t_m, 4) \
                     - (12 * (-1) * latStanceLength) / (2 * T_h2 ** 5) * np.power(t_place - t_m, 5)

        y_stance2Tragectory = np.concatenate((yStance2Out, yStance2In), axis=0)

        ######################## Define z-axis trajectories for swing leg and all 3 stance legs ########################

        # Quintic polynomial, defined in report
        zLift = h_0 + (20 * h) / (2 * T_h1 ** 3) * np.power(t_lift, 3) \
                - (30 * h) / (2 * T_h1 ** 4) * np.power(t_lift, 4) \
                + (12 * h) / (2 * T_h1 ** 5) * np.power(t_lift, 5)

        # Quintic polynomial, defined in report
        zPlace = h_0 + h - (20 * h) / (2 * T_h2 ** 3) * np.power(t_place - t_m, 3) \
                 + (30 * h) / (2 * T_h2 ** 4) * np.power(t_place - t_m, 4) \
                 - (12 * h) / (2 * T_h2 ** 5) * np.power(t_place - t_m, 5)

        z_swingTragectory = np.concatenate((zLift, zPlace), axis=0)

        z_stanceTragectory = h_0 + np.zeros(samplesPerSwing)  # constant, defined in report

        ######################## Define complete trajectory for swing leg and all 3 stance legs ########################

        ### Preallocate memory ###
        swing1FootPositionTrajectory = np.zeros([samplesPerSwing, 3])
        swing2FootPositionTrajectory = np.zeros([samplesPerSwing, 3])
        stance1FootPositionTrajectory = np.zeros([samplesPerSwing, 3])
        stance2FootPositionTrajectory = np.zeros([samplesPerSwing, 3])

        # For each sample in the current phase of the trot cycle
        for i in range(0, samplesPerSwing):

            ### Construct 3D return array ###
            swing1FootPositionTrajectory[i] = np.array(
                [x_swingTragectory[i], y_swing1Tragectory[i], z_swingTragectory[i]])

            swing2FootPositionTrajectory[i] = np.array(
                [x_swingTragectory[i], y_swing2Tragectory[i], z_swingTragectory[i]])

            stance1FootPositionTrajectory[i] = np.array(
                [x_stanceTragectory[i], y_stance1Tragectory[i], z_stanceTragectory[i]])

            stance2FootPositionTrajectory[i] = np.array(
                [x_stanceTragectory[i], y_stance2Tragectory[i], z_stanceTragectory[i]])

        feetTrajectories = np.array([swing1FootPositionTrajectory, swing2FootPositionTrajectory,
                                     stance1FootPositionTrajectory, stance2FootPositionTrajectory])

        ######### Uncomment the following for plots #########

        # fig = plt.figure(figsize=(8.42, 3))
        # plt.plot(xBackSwing, z_swingTragectory[:len(xBackSwing)], color='red', label="Back Swing")
        # plt.plot(xStride, z_swingTragectory[len(xBackSwing):len(xStride)+len(xBackSwing)], color='blue', label="Stride")
        # plt.plot(xForwardSwing, z_swingTragectory[len(xStride)+len(xBackSwing):], color='magenta', label="Forward Swing")
        # plt.plot(x_stanceTragectory, z_stanceTragectory, color='green', label="Stance")
        # plt.axis('equal')
        # plt.xlabel('x Position Relative to Hip (m)')
        # plt.ylabel('z Position Relative to Hip (m)')
        # plt.legend(loc='best')
        # # plt.title("xz Trajectory")
        # # plt.title(
        # #     "xz Trajectory for a Step Length of {} m, Step Height of {} m and Step Period of {} s".format(
        # #         forwardSwingLength,
        # #         swingHeight,
        # #         T_swing))
        # plt.grid()
        # plt.tight_layout()
        # # plt.show()
        # plt.show(block=False)
        #
        # fig = plt.figure(figsize=(8.42, 3))
        # plt.plot(t_backSwing, xBackSwing, color='red', label="Back Swing")
        # plt.plot(t_stride, xStride, color='blue', label="Stride")
        # plt.plot(t_forwardSwing, xForwardSwing, color='magenta', label="Forward Swing")
        # plt.plot(t_stance2, x_stanceTragectory, color='green', label="Stance")
        # plt.xlabel('t (s)')
        # plt.ylabel('x Position Relative to Hip (m)')
        # plt.legend(loc='best')
        # # plt.title("x Position Over Time")
        # # plt.title(
        # #     "x Position Over Time for a Step Length of {} m, Step Height of {} m and Step Period of {} s".format(
        # #         forwardSwingLength,
        # #         swingHeight,
        # #         T_swing))
        # plt.grid()
        # plt.tight_layout()
        # plt.show(block=False)
        #
        # fig = plt.figure(figsize=(8.42, 3))
        # plt.plot(t_lift, zLift, color='red', label="Lift")
        # plt.plot(t_place, zPlace, color='blue', label="Place")
        # plt.plot(t_stance2, z_stanceTragectory, color='magenta', label="Stance")
        # plt.xlabel('t (s)')
        # plt.ylabel('z Position Relative to Hip (m)')
        # plt.legend(loc='best')
        # # plt.title("z Position Over Time")
        # # plt.title(
        # #     "z Position Over Time for a Step Length of {} m, Step Height of {} m and Step Period of {} s".format(
        # #         forwardSwingLength,
        # #         swingHeight,
        # #         T_swing))
        # plt.grid()
        # plt.tight_layout()
        # plt.show(block=False)
        #
        # fig = plt.figure(figsize=(8.42, 3))
        # plt.plot(t_stance, y_swing1Tragectory, color='red', label="Sidestep")
        # plt.plot(t_stance2, y_stance1Tragectory, color='blue', label="Lean")
        # plt.xlabel('t (s)')
        # plt.ylabel('y Position Relative to Hip (m)')
        # plt.legend(loc='best')
        # # plt.title("y Position Over Time")
        # # plt.title(
        # #     "y Position Over Time for a Step Length of {} m, Step Height of {} m and Step Period of {} s".format(
        # #         forwardSwingLength,
        # #         swingHeight,
        # #         T_swing))
        # plt.grid()
        # plt.tight_layout()
        # plt.show(block=False)
        #
        # fig = plt.figure(figsize=(8.42, 3))
        # plt.plot(x_swingTragectory, y_swing1Tragectory, color='red', label="Swing")
        # plt.plot(x_stanceTragectory, y_stance1Tragectory, color='blue', label="Stance")
        # plt.axis('equal')
        # plt.xlabel('x Position Relative to Hip (m)')
        # plt.ylabel('y Position Relative to Hip (m)')
        # plt.legend(loc='best')
        # # plt.title("xy Trajectory")
        # # plt.title("xy Trajectory for a Step Length of {} m, Step Height of {} m and Step Period of {} s".format(
        # #         forwardSwingLength,
        # #         swingHeight,
        # #         T_swing))
        # plt.grid()
        # plt.tight_layout()
        # plt.show()

        return feetTrajectories


    def _genWalkStep(self, forwardSwingLength=0.1, latSwingLength=0, swingHeight=0.08, T_swing=1, h_0=-0.25,
                     sampleRate=240, swingX_0=-0.05, swingY_0=0, stance1X_0=0.05, stance1Y_0=0, stance2X_0=1 / 60,
                     stance2Y_0=0, stance3X_0=-1 / 60, stance3Y_0=0, backSwingCoef=0.2, hipOffset=0.06, stage=0):
        """""
        This method returns the trajectories of all four legs for one stage of the walking cycle, to be used with 
        additional lean periods. That is the swing trajectory for 1 leg and the stance trajectory for the other 3 
        (which are proportional to the swing trajectory). The trajectories are constructed using splines of quintic 
        polynomials and a cycloid curve, as derived in the report.
        Args:
            forwardSwingLength (float): Length of swing in x-direction relative to hip (negative goes backwards).
            latSwingLength (float): Length of swing in y-direction relative to hip (negative goes inwards).
            swingHeight (float): Maximum height of swing phase relative to the stance height. 
            T_swing (float): Swing period.
            h_0 (float): z-position of feet relative to hip during the stance phase (i.e. - body height).
            sampleRate (float): Sample rate, recommended 240 Hz.
            swingX_0 (float): Initial position of the swing foot in the x-direction (forwards/backwards)
            swingY_0 (float): Initial position of the swing foot in the y-direction (sideways)
            stance1X_0 (float): Initial position of stance foot 1 in the x-direction
            stance1Y_0 (float): Initial position of stance foot 1 in the y-direction
            stance2X_0 (float): Initial position of stance foot 2 in the x-direction
            stance2Y_0 (float): Initial position of stance foot 2 in the y-direction
            stance3X_0 (float): Initial position of stance foot 3 in the x-direction
            stance3Y_0 (float): Initial position of stance foot 3 in the y-direction
            backSwingCoef (float): back/forward swing coefficient.
            hipOffset (float): 0 m for feet directly below motors, L1 m for legs straight (recommended).
            stage (int): Current stage of the walking cycle.            
        Returns:
            np.ndarray: Foot trajectories for all feet as a 3D array of floats. The shape is (num_feet, num_samples, 3), 
                        where the last dimension is the Cartesian (x, y, z) position of the foot relative to the hip for 
                        that sample. The foot trajectories are returned in the order (swing foot 1, swing foot 2, stance 
                        foot 1, stance foot 2). Depending on the stage of the trot cycle, this may correspond to 
                        (FR, RL, FL, RR) or (FL, RR, FR, RL).
        """""
        t_1 = 0.25 * T_swing
        t_2 = 0.75 * T_swing
        t_w = 0.5 * T_swing
        samplesPerSwing = int(sampleRate * T_swing)

        t_backSwing = np.linspace(0, t_1, math.ceil(0.25 * samplesPerSwing))
        t_stride = np.linspace(t_1, t_2, math.ceil(0.5 * samplesPerSwing))
        t_forwardSwing = np.linspace(t_2, T_swing, math.ceil(0.25 * samplesPerSwing))
        t_lift = np.linspace(0, t_w, math.ceil(0.5 * samplesPerSwing))
        t_place = np.linspace(t_w, T_swing, math.ceil(0.5 * samplesPerSwing))
        t_stance = np.linspace(0, T_swing, samplesPerSwing)

        swingX_1 = swingX_0 - backSwingCoef * forwardSwingLength
        swingX_2 = swingX_0 + (1 + backSwingCoef) * forwardSwingLength
        swingX_3 = swingX_0 + forwardSwingLength

        stance1X_2 = stance1X_0 - forwardSwingLength / 3
        stance2X_2 = stance2X_0 - forwardSwingLength / 3
        stance3X_2 = stance3X_0 - forwardSwingLength / 3

        h = swingHeight

        T_x1 = t_1
        T_x2 = t_2 - t_1
        T_x3 = T_swing - t_2
        T_h1 = t_w
        T_h2 = T_swing - t_w

        if stage == 0 or stage == 3:
            swingLatLenSign = 1
        elif stage == 1 or stage == 2:
            swingLatLenSign = -1

        ######################## Define x-axis trajectories for swing leg and all 3 stance legs ########################

        xBackSwing = swingX_0 + 20 * (swingX_1 - swingX_0) / (2 * T_x1 ** 3) * np.power(t_backSwing, 3) \
                     + 30 * (swingX_0 - swingX_1) / (2 * T_x1 ** 4) * np.power(t_backSwing, 4) \
                     + 12 * (swingX_1 - swingX_0) / (2 * T_x1 ** 5) * np.power(t_backSwing, 5)

        xStride = (swingX_2 - swingX_1) * (
                    (t_stride - t_1) / T_x2 - (1 / (2 * np.pi)) * np.sin((2 * np.pi * (t_stride - t_1)) / T_x2)) \
                  + swingX_1

        xForwardSwing = swingX_2 + 20 * (swingX_3 - swingX_2) / (2 * T_x3 ** 3) * np.power(t_forwardSwing - t_2, 3) \
                        + 30 * (swingX_2 - swingX_3) / (2 * T_x3 ** 4) * np.power(t_forwardSwing - t_2, 4) \
                        + 12 * (swingX_3 - swingX_2) / (2 * T_x3 ** 5) * np.power(t_forwardSwing - t_2, 5)

        x_swingTragectory = np.concatenate((xBackSwing, xStride, xForwardSwing), axis=0)

        x_stance1Tragectory = stance1X_0 + 20 * (stance1X_2 - stance1X_0) / (2 * T_swing ** 3) * np.power(t_stance, 3) \
                              + 30 * (stance1X_0 - stance1X_2) / (2 * T_swing ** 4) * np.power(t_stance, 4) \
                              + 12 * (stance1X_2 - stance1X_0) / (2 * T_swing ** 5) * np.power(t_stance, 5)

        x_stance2Tragectory = stance2X_0 + 20 * (stance2X_2 - stance2X_0) / (2 * T_swing ** 3) * np.power(t_stance, 3) \
                              + 30 * (stance2X_0 - stance2X_2) / (2 * T_swing ** 4) * np.power(t_stance, 4) \
                              + 12 * (stance2X_2 - stance2X_0) / (2 * T_swing ** 5) * np.power(t_stance, 5)

        x_stance3Tragectory = stance3X_0 + 20 * (stance3X_2 - stance3X_0) / (2 * T_swing ** 3) * np.power(t_stance, 3) \
                              + 30 * (stance3X_0 - stance3X_2) / (2 * T_swing ** 4) * np.power(t_stance, 4) \
                              + 12 * (stance3X_2 - stance3X_0) / (2 * T_swing ** 5) * np.power(t_stance, 5)

        ######################## Define y-axis trajectories for swing leg and all 3 stance legs ########################

        y_swingTragectory = swingY_0 + hipOffset + (20 * swingLatLenSign * latSwingLength) / (
                    2 * T_swing ** 3) * np.power(t_stance, 3) \
                            - (30 * swingLatLenSign * latSwingLength) / (2 * T_swing ** 4) * np.power(t_stance, 4) \
                            + (12 * swingLatLenSign * latSwingLength) / (2 * T_swing ** 5) * np.power(t_stance, 5)

        y_stance1Tragectory = stance1Y_0 + hipOffset + np.zeros(samplesPerSwing)
        y_stance2Tragectory = stance2Y_0 + hipOffset + np.zeros(samplesPerSwing)
        y_stance3Tragectory = stance3Y_0 + hipOffset + np.zeros(samplesPerSwing)

        ######################## Define z-axis trajectories for swing leg and all 3 stance legs ########################

        zLift = h_0 + (20 * h) / (2 * T_h1 ** 3) * np.power(t_lift, 3) \
                - (30 * h) / (2 * T_h1 ** 4) * np.power(t_lift, 4) \
                + (12 * h) / (2 * T_h1 ** 5) * np.power(t_lift, 5)

        zPlace = h_0 + h - (20 * h) / (2 * T_h2 ** 3) * np.power(t_place - t_w, 3) \
                 + (30 * h) / (2 * T_h2 ** 4) * np.power(t_place - t_w, 4) \
                 - (12 * h) / (2 * T_h2 ** 5) * np.power(t_place - t_w, 5)

        z_swingTragectory = np.concatenate((zLift, zPlace), axis=0)

        z_stanceTragectory = h_0 + np.zeros(samplesPerSwing)

        ######################## Define complete trajectory for swing leg and all 3 stance legs ########################

        swingFootPositionTrajectory = np.zeros([samplesPerSwing, 3])
        stance1FootPositionTrajectory = np.zeros([samplesPerSwing, 3])
        stance2FootPositionTrajectory = np.zeros([samplesPerSwing, 3])
        stance3FootPositionTrajectory = np.zeros([samplesPerSwing, 3])

        for i in range(0, samplesPerSwing):
            swingFootPositionTrajectory[i] = np.array([x_swingTragectory[i],
                                                       y_swingTragectory[i],
                                                       z_swingTragectory[i]])

            stance1FootPositionTrajectory[i] = np.array([x_stance1Tragectory[i],
                                                         y_stance1Tragectory[i],
                                                         z_stanceTragectory[i]])

            stance2FootPositionTrajectory[i] = np.array([x_stance2Tragectory[i],
                                                         y_stance2Tragectory[i],
                                                         z_stanceTragectory[i]])

            stance3FootPositionTrajectory[i] = np.array([x_stance3Tragectory[i],
                                                         y_stance3Tragectory[i],
                                                         z_stanceTragectory[i]])

        feetTrajectories = np.array([swingFootPositionTrajectory, stance1FootPositionTrajectory,
                                     stance2FootPositionTrajectory, stance3FootPositionTrajectory])

        return feetTrajectories


    def _genLeanOutTrajectory(self, leanLength=0.02, T_lean=0.25, h_0=-0.28, sampleRate=240, hipOffset=0.06,
                              direction=False, FR_x0=0, FR_y0=0, RL_x0=0, RL_y0=0, FL_x0=0, FL_y0=0, RR_x0=0, RR_y0=0):
        """""
        This method returns the trajectories of all four legs for an additional lean out phase, to be used with 
        walking. The trajectories are constructed using splines of quintiic polynomials. 
        Args:
            leanLength (float): Maximum y-lean during lean phases.
            T_lean (float): Lean period.
            h_0 (float): z-position of feet relative to hip during the stance phase (i.e. - body height).
            sampleRate (float): Sample rate, recommended 240 Hz.
            hipOffset (float): 0 m for feet directly below motors, L1 m for legs straight (recommended). 
            direction (bool): Flag to decide which way to lean out. True leans to the right, false to the left.
            FR_x0 (float): Initial x-position of the FR foot.
            FR_y0 (float): Initial y-position of the FR foot.
            RL_x0 (float): Initial x-position of the RL foot.
            RL_y0 (float): Initial y-position of the RL foot.
            FL_x0 (float): Initial x-position of the FL foot.
            FL_y0 (float): Initial y-position of the FL foot.
            RR_x0 (float): Initial x-position of the RR foot.
            RR_y0 (float): Initial y-position of the RR foot.
        Returns:
            np.ndarray: Foot trajectories for all feet as a 3D array of floats. The shape is (num_feet, num_samples, 3), 
                        where the last dimension is the Cartesian (x, y, z) position of the foot relative to the hip for 
                        that sample. The foot trajectories are returned in the order (FR, RL, FL, RR).
        """""
        samplesPerLean = int(sampleRate * T_lean)
        t_lean = np.linspace(0, T_lean, samplesPerLean)

        if direction:
            leanLenSign = 1
        else:
            leanLenSign = -1

        ######################## Define x-axis trajectories for all 3 legs ########################

        FR_x = FR_x0 + np.zeros(samplesPerLean)
        RL_x = RL_x0 + np.zeros(samplesPerLean)
        FL_x = FL_x0 + np.zeros(samplesPerLean)
        RR_x = RR_x0 + np.zeros(samplesPerLean)

        ######################## Define y-axis trajectories for swing leg and all 3 stance legs ########################

        FR_y = FR_y0 + hipOffset + (20 * -leanLenSign * leanLength) / (2 * T_lean ** 3) * np.power(t_lean, 3) \
               - (30 * -leanLenSign * leanLength) / (2 * T_lean ** 4) * np.power(t_lean, 4) \
               + (12 * -leanLenSign * leanLength) / (2 * T_lean ** 5) * np.power(t_lean, 5)

        RL_y = RL_y0 + hipOffset + (20 * leanLenSign * leanLength) / (2 * T_lean ** 3) * np.power(t_lean, 3) \
               - (30 * leanLenSign * leanLength) / (2 * T_lean ** 4) * np.power(t_lean, 4) \
               + (12 * leanLenSign * leanLength) / (2 * T_lean ** 5) * np.power(t_lean, 5)

        FL_y = FL_y0 + hipOffset + (20 * leanLenSign * leanLength) / (2 * T_lean ** 3) * np.power(t_lean, 3) \
               - (30 * leanLenSign * leanLength) / (2 * T_lean ** 4) * np.power(t_lean, 4) \
               + (12 * leanLenSign * leanLength) / (2 * T_lean ** 5) * np.power(t_lean, 5)

        RR_y = RR_y0 + hipOffset + (20 * -leanLenSign * leanLength) / (2 * T_lean ** 3) * np.power(t_lean, 3) \
               - (30 * -leanLenSign * leanLength) / (2 * T_lean ** 4) * np.power(t_lean, 4) \
               + (12 * -leanLenSign * leanLength) / (2 * T_lean ** 5) * np.power(t_lean, 5)

        ######################## Define z-axis trajectories for swing leg and all 3 stance legs ########################

        z = h_0 + np.zeros(samplesPerLean)

        ######################## Define complete trajectory for all legs ########################

        footPositionTrajectoryFR = np.zeros([samplesPerLean, 3])
        footPositionTrajectoryRL = np.zeros([samplesPerLean, 3])
        footPositionTrajectoryFL = np.zeros([samplesPerLean, 3])
        footPositionTrajectoryRR = np.zeros([samplesPerLean, 3])

        for i in range(0, samplesPerLean):
            footPositionTrajectoryFR[i] = np.array([FR_x[i], FR_y[i], z[i]])
            footPositionTrajectoryRL[i] = np.array([RL_x[i], RL_y[i], z[i]])
            footPositionTrajectoryFL[i] = np.array([FL_x[i], FL_y[i], z[i]])
            footPositionTrajectoryRR[i] = np.array([RR_x[i], RR_y[i], z[i]])

        feetTrajectories = np.array([footPositionTrajectoryFR, footPositionTrajectoryRL,
                                     footPositionTrajectoryFL, footPositionTrajectoryRR])

        return feetTrajectories


    def _genLeanInTrajectory(self, leanLength=0.02, T_lean=0.25, h_0=-0.28, sampleRate=240, hipOffset=0.06,
                             direction=False, FR_x0=0, FR_y0=0, RL_x0=0, RL_y0=0, FL_x0=0, FL_y0=0, RR_x0=0, RR_y0=0):
        """""
        This method returns the trajectories of all four legs for an additional lean in phase, to be used with 
        walking. The trajectories are constructed using splines of quintiic polynomials. 
        Args:
            leanLength (float): Maximum y-lean during lean phases.
            T_lean (float): Lean period.
            h_0 (float): z-position of feet relative to hip during the stance phase (i.e. - body height).
            sampleRate (float): Sample rate, recommended 240 Hz.
            hipOffset (float): 0 m for feet directly below motors, L1 m for legs straight (recommended). 
            direction (bool): Flag to decide which way to lean in. True leans to the right, false to the left.
            FR_x0 (float): Initial x-position of the FR foot.
            FR_y0 (float): Initial y-position of the FR foot.
            RL_x0 (float): Initial x-position of the RL foot.
            RL_y0 (float): Initial y-position of the RL foot.
            FL_x0 (float): Initial x-position of the FL foot.
            FL_y0 (float): Initial y-position of the FL foot.
            RR_x0 (float): Initial x-position of the RR foot.
            RR_y0 (float): Initial y-position of the RR foot.
        Returns:
            np.ndarray: Foot trajectories for all feet as a 3D array of floats. The shape is (num_feet, num_samples, 3), 
                        where the last dimension is the Cartesian (x, y, z) position of the foot relative to the hip for 
                        that sample. The foot trajectories are returned in the order (FR, RL, FL, RR).
        """""
        samplesPerLean = int(sampleRate * T_lean)
        t_lean = np.linspace(0, T_lean, samplesPerLean)

        if direction:
            leanLenSign = 1
        else:
            leanLenSign = -1

        ######################## Define x-axis trajectories for all 3 legs ########################

        FR_x = FR_x0 + np.zeros(samplesPerLean)
        RL_x = RL_x0 + np.zeros(samplesPerLean)
        FL_x = FL_x0 + np.zeros(samplesPerLean)
        RR_x = RR_x0 + np.zeros(samplesPerLean)

        ######################## Define y-axis trajectories for swing leg and all 3 stance legs ########################

        FR_y = FR_y0 + hipOffset \
               - (20 * leanLenSign * leanLength) / (2 * T_lean ** 3) * np.power(t_lean, 3) \
               + (30 * leanLenSign * leanLength) / (2 * T_lean ** 4) * np.power(t_lean, 4) \
               - (12 * leanLenSign * leanLength) / (2 * T_lean ** 5) * np.power(t_lean, 5)

        RL_y = RL_y0 + hipOffset \
               - (20 * -leanLenSign * leanLength) / (2 * T_lean ** 3) * np.power(t_lean, 3) \
               + (30 * -leanLenSign * leanLength) / (2 * T_lean ** 4) * np.power(t_lean, 4) \
               - (12 * -leanLenSign * leanLength) / (2 * T_lean ** 5) * np.power(t_lean, 5)

        FL_y = FL_y0 + hipOffset \
               - (20 * -leanLenSign * leanLength) / (2 * T_lean ** 3) * np.power(t_lean, 3) \
               + (30 * -leanLenSign * leanLength) / (2 * T_lean ** 4) * np.power(t_lean, 4) \
               - (12 * -leanLenSign * leanLength) / (2 * T_lean ** 5) * np.power(t_lean, 5)

        RR_y = RR_y0 + hipOffset \
               - (20 * leanLenSign * leanLength) / (2 * T_lean ** 3) * np.power(t_lean, 3) \
               + (30 * leanLenSign * leanLength) / (2 * T_lean ** 4) * np.power(t_lean, 4) \
               - (12 * leanLenSign * leanLength) / (2 * T_lean ** 5) * np.power(t_lean, 5)

        ######################## Define z-axis trajectories for swing leg and all 3 stance legs ########################

        z = h_0 + np.zeros(samplesPerLean)

        ######################## Define complete trajectory for all legs ########################

        footPositionTrajectoryFR = np.zeros([samplesPerLean, 3])
        footPositionTrajectoryRL = np.zeros([samplesPerLean, 3])
        footPositionTrajectoryFL = np.zeros([samplesPerLean, 3])
        footPositionTrajectoryRR = np.zeros([samplesPerLean, 3])

        for i in range(0, samplesPerLean):
            footPositionTrajectoryFR[i] = np.array([FR_x[i], FR_y[i], z[i]])
            footPositionTrajectoryRL[i] = np.array([RL_x[i], RL_y[i], z[i]])
            footPositionTrajectoryFL[i] = np.array([FL_x[i], FL_y[i], z[i]])
            footPositionTrajectoryRR[i] = np.array([RR_x[i], RR_y[i], z[i]])

        feetTrajectories = np.array([footPositionTrajectoryFR, footPositionTrajectoryRL,
                                     footPositionTrajectoryFL, footPositionTrajectoryRR])

        return feetTrajectories


    def _genWalkStepNoAdditionalLean(self, forwardSwingLength=0.1, latSwingLength=0, swingHeight=0.08,
                                    latStanceLength=0.025, T_swing=1, h_0=-0.25, sampleRate=240, swingX_0=-0.05,
                                    swingY_0=0, stance1X_0=0.05, stance1Y_0=0, stance2X_0=1/60, stance2Y_0=0,
                                    stance3X_0=-1/60, stance3Y_0=0, backSwingCoef=0.2, hipOffset=0.06, stage=0):
        """""
        This method returns the trajectories of all four legs for one stage of the walking cycle, with no
        additional lean periods but instead leaning while one foot is swinging. Includes the swing trajectory for 1 leg 
        and the stance trajectory for the other 3 (which are proportional to the swing trajectory). The trajectories are 
        constructed using splines of quintic polynomials and a cycloid curve, as derived in the report.
        Args:
            forwardSwingLength (float): Length of swing in x-direction relative to hip (negative goes backwards).
            latSwingLength (float): Length of swing in y-direction relative to hip (negative goes inwards).
            swingHeight (float): Maximum height of swing phase relative to the stance height. 
            latStanceLength (float): Maximum y-lean during stance phase.
            T_swing (float): Swing period.
            h_0 (float): z-position of feet relative to hip during the stance phase (i.e. - body height).
            sampleRate (float): Sample rate, recommended 240 Hz.
            swingX_0 (float): Initial position of the swing foot in the x-direction (forwards/backwards)
            swingY_0 (float): Initial position of the swing foot in the y-direction (sideways)
            stance1X_0 (float): Initial position of stance foot 1 in the x-direction
            stance1Y_0 (float): Initial position of stance foot 1 in the y-direction
            stance2X_0 (float): Initial position of stance foot 2 in the x-direction
            stance2Y_0 (float): Initial position of stance foot 2 in the y-direction
            stance3X_0 (float): Initial position of stance foot 3 in the x-direction
            stance3Y_0 (float): Initial position of stance foot 3 in the y-direction
            backSwingCoef (float): back/forward swing coefficient.
            hipOffset (float): 0 m for feet directly below motors, L1 m for legs straight (recommended).
            stage (int): Current stage of the walking cycle.            
        Returns:
            np.ndarray: Foot trajectories for all feet as a 3D array of floats. The shape is (num_feet, num_samples, 3), 
                        where the last dimension is the Cartesian (x, y, z) position of the foot relative to the hip for 
                        that sample. The foot trajectories are returned in the order (swing foot 1, swing foot 2, stance 
                        foot 1, stance foot 2). Depending on the stage of the trot cycle, this may correspond to 
                        (FR, RL, FL, RR) or (FL, RR, FR, RL).
        """""
        t_1 = 0.25 * T_swing
        t_2 = 0.75 * T_swing
        t_w = 0.5 * T_swing
        samplesPerSwing = int(sampleRate * T_swing)

        t_backSwing = np.linspace(0, t_1, math.ceil(0.25 * samplesPerSwing))
        t_stride = np.linspace(t_1, t_2, math.ceil(0.5 * samplesPerSwing))
        t_forwardSwing = np.linspace(t_2, T_swing, math.ceil(0.25 * samplesPerSwing))
        t_lift = np.linspace(0, t_w, math.ceil(0.5 * samplesPerSwing))
        t_place = np.linspace(t_w, T_swing, math.ceil(0.5 * samplesPerSwing))
        t_stance = np.linspace(0, T_swing, samplesPerSwing)

        swingX_1 = swingX_0 - backSwingCoef * forwardSwingLength
        swingX_2 = swingX_0 + (1 + backSwingCoef) * forwardSwingLength
        swingX_3 = swingX_0 + forwardSwingLength

        stance1X_2 = stance1X_0 - forwardSwingLength / 3
        stance2X_2 = stance2X_0 - forwardSwingLength / 3
        stance3X_2 = stance3X_0 - forwardSwingLength / 3

        h = swingHeight

        T_x1 = t_1
        T_x2 = t_2 - t_1
        T_x3 = T_swing - t_2
        T_h1 = t_w
        T_h2 = T_swing - t_w

        if stage == 0:
            swingLatLenSign = 1
            stance1LatLenSign = 1
            stance2LatLenSign = -1
            stance3LatLenSign = -1
        elif stage == 1:
            swingLatLenSign = -1
            stance1LatLenSign = 1
            stance2LatLenSign = 1
            stance3LatLenSign = -1
        elif stage == 2:
            swingLatLenSign = -1
            stance1LatLenSign = -1
            stance2LatLenSign = 1
            stance3LatLenSign = 1
        elif stage == 3:
            swingLatLenSign = 1
            stance1LatLenSign = -1
            stance2LatLenSign = -1
            stance3LatLenSign = 1

        ######################## Define x-axis trajectories for swing leg and all 3 stance legs ########################

        xBackSwing = swingX_0 + (20 * swingX_1 - 20 * swingX_0) / (2 * T_x1 ** 3) * np.power(t_backSwing, 3) \
                     + (30 * swingX_0 - 30 * swingX_1) / (2 * T_x1 ** 4) * np.power(t_backSwing, 4) \
                     + (12 * swingX_1 - 12 * swingX_0) / (2 * T_x1 ** 5) * np.power(t_backSwing, 5)

        xStride = (swingX_2 - swingX_1) * (
                (t_stride - t_1) / T_x2 - (1 / (2 * np.pi)) * np.sin((2 * np.pi * (t_stride - t_1)) / T_x2)) \
                  + swingX_1

        xForwardSwing = swingX_2 + (20 * swingX_3 - 20 * swingX_2) / (2 * T_x3 ** 3) * np.power(t_forwardSwing - t_2, 3) \
                        + (30 * swingX_2 - 30 * swingX_3) / (2 * T_x3 ** 4) * np.power(t_forwardSwing - t_2, 4) \
                        + (12 * swingX_3 - 12 * swingX_2) / (2 * T_x3 ** 5) * np.power(t_forwardSwing - t_2, 5)

        x_swingTragectory = np.concatenate((xBackSwing, xStride, xForwardSwing), axis=0)

        x_stance1Tragectory = stance1X_0 + (20 * stance1X_2 - 20 * stance1X_0) / (2 * (T_swing) ** 3) * np.power(
            t_stance, 3) \
                              + (30 * stance1X_0 - 30 * stance1X_2) / (2 * (T_swing) ** 4) * np.power(t_stance, 4) \
                              + (12 * stance1X_2 - 12 * stance1X_0) / (2 * (T_swing) ** 5) * np.power(t_stance, 5)

        x_stance2Tragectory = stance2X_0 + (20 * stance2X_2 - 20 * stance2X_0) / (2 * (T_swing) ** 3) * np.power(
            t_stance, 3) \
                              + (30 * stance2X_0 - 30 * stance2X_2) / (2 * (T_swing) ** 4) * np.power(t_stance, 4) \
                              + (12 * stance2X_2 - 12 * stance2X_0) / (2 * (T_swing) ** 5) * np.power(t_stance, 5)

        x_stance3Tragectory = stance3X_0 + (20 * stance3X_2 - 20 * stance3X_0) / (2 * (T_swing) ** 3) * np.power(
            t_stance, 3) \
                              + (30 * stance3X_0 - 30 * stance3X_2) / (2 * (T_swing) ** 4) * np.power(t_stance, 4) \
                              + (12 * stance3X_2 - 12 * stance3X_0) / (2 * (T_swing) ** 5) * np.power(t_stance, 5)

        ######################## Define y-axis trajectories for swing leg and all 3 stance legs ########################

        y_swingTragectory = swingY_0 + hipOffset + (20 * swingLatLenSign * latSwingLength) / (
                    2 * T_swing ** 3) * np.power(t_stance, 3) \
                            - (30 * swingLatLenSign * latSwingLength) / (2 * T_swing ** 4) * np.power(t_stance, 4) \
                            + (12 * swingLatLenSign * latSwingLength) / (2 * T_swing ** 5) * np.power(t_stance, 5)

        yStance1Out = stance1Y_0 + hipOffset + (20 * stance1LatLenSign * latStanceLength) / (2 * T_h1 ** 3) * np.power(
            t_lift, 3) \
                      - (30 * stance1LatLenSign * latStanceLength) / (2 * T_h1 ** 4) * np.power(t_lift, 4) \
                      + (12 * stance1LatLenSign * latStanceLength) / (2 * T_h1 ** 5) * np.power(t_lift, 5)
        yStance1In = stance1Y_0 + hipOffset + stance1LatLenSign * latStanceLength \
                     - (20 * stance1LatLenSign * latStanceLength) / (2 * T_h2 ** 3) * np.power(t_place - t_w, 3) \
                     + (30 * stance1LatLenSign * latStanceLength) / (2 * T_h2 ** 4) * np.power(t_place - t_w, 4) \
                     - (12 * stance1LatLenSign * latStanceLength) / (2 * T_h2 ** 5) * np.power(t_place - t_w, 5)

        y_stance1Tragectory = np.concatenate((yStance1Out, yStance1In), axis=0)

        # y_stance1Tragectory = stance1Y_0 + hipOffset + (20 * latStanceLength) / (2 * T_swing ** 3) * np.power(t_stance, 3) \
        #               - (30 * latStanceLength) / (2 * T_swing ** 4) * np.power(t_stance, 4) \
        #               + (12 * latStanceLength) / (2 * T_swing ** 5) * np.power(t_stance, 5)

        yStance2Out = stance2Y_0 + hipOffset + (20 * stance2LatLenSign * latStanceLength) / (2 * T_h1 ** 3) * np.power(
            t_lift, 3) \
                      - (30 * stance2LatLenSign * latStanceLength) / (2 * T_h1 ** 4) * np.power(t_lift, 4) \
                      + (12 * stance2LatLenSign * latStanceLength) / (2 * T_h1 ** 5) * np.power(t_lift, 5)
        yStance2In = stance2Y_0 + hipOffset + stance2LatLenSign * latStanceLength \
                     - (20 * stance2LatLenSign * latStanceLength) / (2 * T_h2 ** 3) * np.power(t_place - t_w, 3) \
                     + (30 * stance2LatLenSign * latStanceLength) / (2 * T_h2 ** 4) * np.power(t_place - t_w, 4) \
                     - (12 * stance2LatLenSign * latStanceLength) / (2 * T_h2 ** 5) * np.power(t_place - t_w, 5)

        y_stance2Tragectory = np.concatenate((yStance2Out, yStance2In), axis=0)

        # y_stance2Tragectory = stance2Y_0 + hipOffset + (20 * latStanceLength) / (2 * T_swing ** 3) * np.power(t_stance, 3) \
        #               - (30 * latStanceLength) / (2 * T_swing ** 4) * np.power(t_stance, 4) \
        #               + (12 * latStanceLength) / (2 * T_swing ** 5) * np.power(t_stance, 5)

        yStance3Out = stance3Y_0 + hipOffset + (20 * stance3LatLenSign * latStanceLength) / (2 * T_h1 ** 3) * np.power(
            t_lift, 3) \
                      - (30 * stance3LatLenSign * latStanceLength) / (2 * T_h1 ** 4) * np.power(t_lift, 4) \
                      + (12 * stance3LatLenSign * latStanceLength) / (2 * T_h1 ** 5) * np.power(t_lift, 5)
        yStance3In = stance3Y_0 + hipOffset + stance3LatLenSign * latStanceLength \
                     - (20 * stance3LatLenSign * latStanceLength) / (2 * T_h2 ** 3) * np.power(t_place - t_w, 3) \
                     + (30 * stance3LatLenSign * latStanceLength) / (2 * T_h2 ** 4) * np.power(t_place - t_w, 4) \
                     - (12 * stance3LatLenSign * latStanceLength) / (2 * T_h2 ** 5) * np.power(t_place - t_w, 5)

        y_stance3Tragectory = np.concatenate((yStance3Out, yStance3In), axis=0)

        # y_stance3Tragectory = stance3Y_0 + hipOffset + (20 * latStanceLength) / (2 * T_swing ** 3) * np.power(t_stance, 3) \
        #               - (30 * latStanceLength) / (2 * T_swing ** 4) * np.power(t_stance, 4) \
        #               + (12 * latStanceLength) / (2 * T_swing ** 5) * np.power(t_stance, 5)

        ######################## Define z-axis trajectories for swing leg and all 3 stance legs ########################

        zLift = h_0 + (20 * h) / (2 * T_h1 ** 3) * np.power(t_lift, 3) \
                - (30 * h) / (2 * T_h1 ** 4) * np.power(t_lift, 4) \
                + (12 * h) / (2 * T_h1 ** 5) * np.power(t_lift, 5)

        zPlace = h_0 + h - (20 * h) / (2 * T_h2 ** 3) * np.power(t_place - t_w, 3) \
                 + (30 * h) / (2 * T_h2 ** 4) * np.power(t_place - t_w, 4) \
                 - (12 * h) / (2 * T_h2 ** 5) * np.power(t_place - t_w, 5)

        z_swingTragectory = np.concatenate((zLift, zPlace), axis=0)

        z_stanceTragectory = h_0 + np.zeros(samplesPerSwing)

        ######################## Define complete trajectory for swing leg and all 3 stance legs ########################

        swingFootPositionTrajectory = np.zeros([samplesPerSwing, 3])
        stance1FootPositionTrajectory = np.zeros([samplesPerSwing, 3])
        stance2FootPositionTrajectory = np.zeros([samplesPerSwing, 3])
        stance3FootPositionTrajectory = np.zeros([samplesPerSwing, 3])

        for i in range(0, samplesPerSwing):
            swingFootPositionTrajectory[i] = np.array([x_swingTragectory[i],
                                                       y_swingTragectory[i],
                                                       z_swingTragectory[i]])

            stance1FootPositionTrajectory[i] = np.array([x_stance1Tragectory[i],
                                                         y_stance1Tragectory[i],
                                                         z_stanceTragectory[i]])

            stance2FootPositionTrajectory[i] = np.array([x_stance2Tragectory[i],
                                                         y_stance2Tragectory[i],
                                                         z_stanceTragectory[i]])

            stance3FootPositionTrajectory[i] = np.array([x_stance3Tragectory[i],
                                                         y_stance3Tragectory[i],
                                                         z_stanceTragectory[i]])

        feetTrajectories = np.array([swingFootPositionTrajectory, stance1FootPositionTrajectory,
                                     stance2FootPositionTrajectory, stance3FootPositionTrajectory])

        return feetTrajectories


    def _calculateFK(self, jointPositions, L1=0.05955, L2=0.15, L3=0.21):
        """""
        This method takes the joint positions of a single leg of PAWS and returns the position of the foot on the x-axis
        (forward/backward direction), y-axis (sideways direction) and z-axis (vertical direction).
        Args:
            jointPositions (list of float): The list of the joint angles i.e. [theta1, theta2, theta3].
            L1 (float): Length of link1 in meters.
            L2 (float): Length of link2 (the upper leg limb) in meters.
            L3 (float): Length of link3 (the lower leg limb) in meters.
        Returns:
            list of float: the position of the foot on the x-axis (forward), y-axis (sideways) and z-axis (vertical).
        """""
        theta1 = jointPositions[0]
        theta2 = jointPositions[1]
        theta3 = jointPositions[2]

        # Equations taken from the homogeneous transformation matrix derived from the DH parameters (derivation in report)
        xPos = L2 * np.sin(theta2) + L3 * np.sin(theta2 + theta3)
        yPos = L1 * np.cos(theta1) + L2 * np.sin(theta1) * np.cos(theta2) + L3 * np.sin(theta1) * np.cos(
            theta2 + theta3)
        zPos = L1 * np.sin(theta1) - L2 * np.cos(theta1) * np.cos(theta2) - L3 * np.cos(theta1) * np.cos(
            theta2 + theta3)

        footPos = np.array([xPos, yPos, zPos])

        return footPos


    def _calRelFootVel(self, jointPositions, jointVelocities, L1=0.05955, L2=0.15, L3=0.21):
        """""
        This method takes the joint positions and velocities of a single leg of PAWS and returns the velocities of the 
        foot in the x-direction (forward/backward direction), y-direction (sideways direction) and z-direction (vertical 
        direction).
        Args:
            jointPositions (list of float): The list of the joint angles i.e. [theta1, theta2, theta3].
            jointVelocities (list of float): The list of the joint velocities i.e. [theta1_dot, theta2_dot, theta3_dot].
            L1 (float): Length of link1 in meters.
            L2 (float): Length of link2 (the upper leg limb) in meters.
            L3 (float): Length of link3 (the lower leg limb) in meters.
        Returns:
            list of float: the velocities of the foot in the x-direction (forward), y-direction (sideways) and 
                           z-direction (vertical).
        """""
        theta1 = jointPositions[0]
        theta2 = jointPositions[1]
        theta3 = jointPositions[2]

        J_11 = 0
        J_12 = L2 * np.cos(theta2) + L3 * np.cos(theta2 + theta3)
        J_13 = L3 * np.cos(theta2 + theta3)

        J_21 = -L1 * np.sin(theta1) + np.cos(theta1) * (L2 * np.cos(theta2) + L3 * np.cos(theta2 + theta3))
        J_22 = -np.sin(theta1) * (L2 * np.sin(theta2) + L3 * np.sin(theta2 + theta3))
        J_23 = -L3 * np.sin(theta1) * np.sin(theta2 + theta3)

        J_31 = L1 * np.cos(theta1) + np.sin(theta1) * (L2 * np.cos(theta2) + L3 * np.cos(theta2 + theta3))
        J_32 = np.cos(theta1) * (L2 * np.sin(theta2) + L3 * np.sin(theta2 + theta3))
        J_33 = L3 * np.cos(theta1) * np.sin(theta2 + theta3)

        jacobian = np.array([[J_11, J_12, J_13], [J_21, J_22, J_23], [J_31, J_32, J_33]])

        jointVelocities = np.array(jointVelocities)
        footVelocities = np.matmul(jacobian, jointVelocities)

        return footVelocities


    def _collectSimData(self):
        """""
        This method measures the current state of the simulated PAWS robot and appends it to a Pandas dataframe for
        later loging to a CSV file. It is not logged immediately because that would be too slow. Instead, simulation 
        data for each swing period is collected and then the entire swing period results are appended to a CSV for 
        later ploting. State measurements include the simulation time, the position and orientation of the COG of the 
        robot in world coordinates, the Cartesian position of each foot in world coordinates, and the joint angle of 
        all motors.
        Args:
            None
        Returns:
            None
        """""
        basePos, baseOrient = p.getBasePositionAndOrientation(self.simPaws)      # Base position and orientation
        baseOrient = (180/np.pi)*np.array(p.getEulerFromQuaternion(baseOrient))  # Euler orientation in degrees
        footStateFR = p.getLinkState(self.simPaws, self.FR_FOOT_JOINT)
        footStateRL = p.getLinkState(self.simPaws, self.RL_FOOT_JOINT)
        footStateFL = p.getLinkState(self.simPaws, self.FL_FOOT_JOINT)
        footStateRR = p.getLinkState(self.simPaws, self.RR_FOOT_JOINT)

        ### Get Cartisian position and orientation for each foot in world coordinates ###
        footPosFR = footStateFR[0]
        footOrientFR = footStateFR[1]
        footPosRL = footStateRL[0]
        footOrientRL = footStateRL[1]
        footPosFL = footStateFL[0]
        footOrientFL = footStateFL[1]
        footPosRR = footStateRR[0]
        footOrientRR = footStateRR[1]

        # Define the offset of the tip of the foot from the COG in link's local frame
        offset = [0, 0, -0.016]

        ### Convert the link orientation (quaternion) to a rotation matrix ###
        rotMatFR = p.getMatrixFromQuaternion(footOrientFR)
        rotMatRL = p.getMatrixFromQuaternion(footOrientRL)
        rotMatFL = p.getMatrixFromQuaternion(footOrientFL)
        rotMatRR = p.getMatrixFromQuaternion(footOrientRR)

        ### Apply the rotation to the offset ###
        worldFrameOffsetFR = [
            rotMatFR[0] * offset[0] + rotMatFR[1] * offset[1] + rotMatFR[2] * offset[2],
            rotMatFR[3] * offset[0] + rotMatFR[4] * offset[1] + rotMatFR[5] * offset[2],
            rotMatFR[6] * offset[0] + rotMatFR[7] * offset[1] + rotMatFR[8] * offset[2]
        ]

        worldFrameOffsetRL = [
            rotMatRL[0] * offset[0] + rotMatRL[1] * offset[1] + rotMatRL[2] * offset[2],
            rotMatRL[3] * offset[0] + rotMatRL[4] * offset[1] + rotMatRL[5] * offset[2],
            rotMatRL[6] * offset[0] + rotMatRL[7] * offset[1] + rotMatRL[8] * offset[2]
        ]

        worldFrameOffsetFL = [
            rotMatFL[0] * offset[0] + rotMatFL[1] * offset[1] + rotMatFL[2] * offset[2],
            rotMatFL[3] * offset[0] + rotMatFL[4] * offset[1] + rotMatFL[5] * offset[2],
            rotMatFL[6] * offset[0] + rotMatFL[7] * offset[1] + rotMatFL[8] * offset[2]
        ]

        worldFrameOffsetRR = [
            rotMatRR[0] * offset[0] + rotMatRR[1] * offset[1] + rotMatRR[2] * offset[2],
            rotMatRR[3] * offset[0] + rotMatRR[4] * offset[1] + rotMatRR[5] * offset[2],
            rotMatRR[6] * offset[0] + rotMatRR[7] * offset[1] + rotMatRR[8] * offset[2]
        ]

        ### Add the offset in the world frame to the link's position to get the new position ###
        footWorldPosFR = [footPosFR[0] + worldFrameOffsetFR[0],
                          footPosFR[1] + worldFrameOffsetFR[1],
                          footPosFR[2] + worldFrameOffsetFR[2]]

        footWorldPosRL = [footPosRL[0] + worldFrameOffsetRL[0],
                          footPosRL[1] + worldFrameOffsetRL[1],
                          footPosRL[2] + worldFrameOffsetRL[2]]

        footWorldPosFL = [footPosFL[0] + worldFrameOffsetFL[0],
                          footPosFL[1] + worldFrameOffsetFL[1],
                          footPosFL[2] + worldFrameOffsetFL[2]]

        footWorldPosRR = [footPosRR[0] + worldFrameOffsetRR[0],
                          footPosRR[1] + worldFrameOffsetRR[1],
                          footPosRR[2] + worldFrameOffsetRR[2]]

        ### Get the joint angles of of all legs ###
        jointAnglesFR = np.array(p.getJointStates(self.simPaws, [self.FR_HIP_ABDUCTOR_ADDUCTOR_JOINT,
                                                                 self.FR_HIP_FLEXOR_EXTENDOR_JOINT,
                                                                 self.FR_KNEE_JOINT]), dtype=object)[:, 0]
        jointAnglesRL = np.array(p.getJointStates(self.simPaws, [self.RL_HIP_ABDUCTOR_ADDUCTOR_JOINT,
                                                                 self.RL_HIP_FLEXOR_EXTENDOR_JOINT,
                                                                 self.RL_KNEE_JOINT]), dtype=object)[:, 0]
        jointAnglesFL = np.array(p.getJointStates(self.simPaws, [self.FL_HIP_ABDUCTOR_ADDUCTOR_JOINT,
                                                                 self.FL_HIP_FLEXOR_EXTENDOR_JOINT,
                                                                 self.FL_KNEE_JOINT]), dtype=object)[:, 0]
        jointAnglesRR = np.array(p.getJointStates(self.simPaws, [self.RR_HIP_ABDUCTOR_ADDUCTOR_JOINT,
                                                                 self.RR_HIP_FLEXOR_EXTENDOR_JOINT,
                                                                 self.RR_KNEE_JOINT]), dtype=object)[:, 0]

        if self.firstSwingLog: # If this is the first log of the current swing period, create a new data frame

            self.cog_df = pd.DataFrame({'Time (s)': self.stepCounter/self.sampleRate, 'COG x Position (m)':[basePos[0]],
                                   'COG y Position (m)': [basePos[1]], 'COG z Position (m)': [basePos[2]],
                                   'COG Roll (deg)': [baseOrient[0]], 'COG Pitch (deg)': [baseOrient[1]],
                                   'COG Yaw (deg)': [baseOrient[2]],
                                   'FR Foot x Position (m)': [footWorldPosFR[0]],
                                   'FR Foot y Position (m)': [footWorldPosFR[1]],
                                   'FR Foot z Position (m)': [footWorldPosFR[2]],
                                   'RL Foot x Position (m)': [footWorldPosRL[0]],
                                   'RL Foot y Position (m)': [footWorldPosRL[1]],
                                   'RL Foot z Position (m)': [footWorldPosRL[2]],
                                   'FL Foot x Position (m)': [footWorldPosFL[0]],
                                   'FL Foot y Position (m)': [footWorldPosFL[1]],
                                   'FL Foot z Position (m)': [footWorldPosFL[2]],
                                   'RR Foot x Position (m)': [footWorldPosRR[0]],
                                   'RR Foot y Position (m)': [footWorldPosRR[1]],
                                   'RR Foot z Position (m)': [footWorldPosRR[2]],
                                   'FR Hip Abd/Add Joint Angle': [jointAnglesFR[0]],
                                   'FR Hip Flex/Ext Joint Angle': [jointAnglesFR[1]],
                                   'FR Knee Flex/Ext Joint Angle': [jointAnglesFR[2]],
                                   'RL Hip Abd/Add Joint Angle': [jointAnglesRL[0]],
                                   'RL Hip Flex/Ext Joint Angle': [jointAnglesRL[1]],
                                   'RL Knee Flex/Ext Joint Angle': [jointAnglesRL[2]],
                                   'FL Hip Abd/Add Joint Angle': [jointAnglesFL[0]],
                                   'FL Hip Flex/Ext Joint Angle': [jointAnglesFL[1]],
                                   'FL Knee Flex/Ext Joint Angle': [jointAnglesFL[2]],
                                   'RR Hip Abd/Add Joint Angle': [jointAnglesRR[0]],
                                   'RR Hip Flex/Ext Joint Angle': [jointAnglesRR[1]],
                                   'RR Knee Flex/Ext Joint Angle': [jointAnglesRR[2]]}, dtype=np.float32)
            self.firstSwingLog = False;
        else:
            new_cog_df = pd.DataFrame({'Time (s)': self.stepCounter/self.sampleRate, 'COG x Position (m)': [basePos[0]],
                                       'COG y Position (m)': [basePos[1]], 'COG z Position (m)': [basePos[2]],
                                       'COG Roll (deg)': [baseOrient[0]], 'COG Pitch (deg)': [baseOrient[1]],
                                       'COG Yaw (deg)': [baseOrient[2]],
                                       'FR Foot x Position (m)': [footWorldPosFR[0]],
                                       'FR Foot y Position (m)': [footWorldPosFR[1]],
                                       'FR Foot z Position (m)': [footWorldPosFR[2]],
                                       'RL Foot x Position (m)': [footWorldPosRL[0]],
                                       'RL Foot y Position (m)': [footWorldPosRL[1]],
                                       'RL Foot z Position (m)': [footWorldPosRL[2]],
                                       'FL Foot x Position (m)': [footWorldPosFL[0]],
                                       'FL Foot y Position (m)': [footWorldPosFL[1]],
                                       'FL Foot z Position (m)': [footWorldPosFL[2]],
                                       'RR Foot x Position (m)': [footWorldPosRR[0]],
                                       'RR Foot y Position (m)': [footWorldPosRR[1]],
                                       'RR Foot z Position (m)': [footWorldPosRR[2]],
                                       'FR Hip Abd/Add Joint Angle': [jointAnglesFR[0]],
                                       'FR Hip Flex/Ext Joint Angle': [jointAnglesFR[1]],
                                       'FR Knee Flex/Ext Joint Angle': [jointAnglesFR[2]],
                                       'RL Hip Abd/Add Joint Angle': [jointAnglesRL[0]],
                                       'RL Hip Flex/Ext Joint Angle': [jointAnglesRL[1]],
                                       'RL Knee Flex/Ext Joint Angle': [jointAnglesRL[2]],
                                       'FL Hip Abd/Add Joint Angle': [jointAnglesFL[0]],
                                       'FL Hip Flex/Ext Joint Angle': [jointAnglesFL[1]],
                                       'FL Knee Flex/Ext Joint Angle': [jointAnglesFL[2]],
                                       'RR Hip Abd/Add Joint Angle': [jointAnglesRR[0]],
                                       'RR Hip Flex/Ext Joint Angle': [jointAnglesRR[1]],
                                       'RR Knee Flex/Ext Joint Angle': [jointAnglesRR[2]]}, dtype=np.float32)
            self.cog_df = pd.concat([self.cog_df, new_cog_df])


    def _sendMotorPosCanMsg(self, motorNum, position):
        """""
        This method consructs and sends a CAN message to control the position of a specified motor.
        Args:
            motorNum (int): number of motor to be controlled.
            position (float): desired position of the motor in radians.
        Returns:
            None
        """""
        bus = can.Bus(interface='socketcan', channel='can0', bitrate=1000000)

        ### Convert position in radians to control value ###
        pos_in_deg = position * 180.0 / np.pi
        control_value = int(pos_in_deg * 100)

        ### Construct CAN measage ###
        msg = can.Message(arbitration_id=0x140 + motorNum,
                          data=[0xa4, 0x00, 0x50, 0x00, control_value & 0xFF,
                                (control_value >> 8) & 0xFF,
                                (control_value >> 16) & 0xFF,
                                (control_value >> 24) & 0xFF],
                          is_extended_id=False)

        # print("frame", msg)
        print("pos", control_value / 100)
        time.sleep(0.002)
        bus.send(msg)
