import pybullet as p             # Physics engine
import pybullet_data             # Contains plane.urdf
import time                      # Used for delays and timing
import numpy as np               # Efficient numerical computation
import matplotlib.pyplot as plt  # Visualisation
import random                    # Random number generation
import math                      # Used in trajectory generation function
import pandas as pd              # Used for logging simulation data
import gymnasium as gym          # Used to define and interact with RL environment
from gymnasium import spaces     # Provides functions for creating action and observation spaces


class PawsEnv(gym.Env):
    """""
    This class creates a custom reinforcement learning environment for the PAWS robot, which inherits from the base 
    class gym.Env provided by the Gymnassium environment. This class adheres to the standard interface of a Gymnassium 
    API, ensuring compatibility with popular RL libraries and frameworks including Stable Baselines 3. This environment 
    is for the policy modulating trajectory generater (PMTG) architecture, which modulates the forward step length, 
    lateral step length, step height and step period of the trot gait generator deveolped in the model-based controller.

    Methods:
        __init__(renderMode, maxXdist, truncationSteps, max_tilt, maxFwdVel, minFwdVel, initFwdSwgLen, initLatSwgLen, 
                 initSwgHght, initTswg, latStanceLength, h_0, sampleRate, backSwingCoef, hipOffset, terrain, 
                 hieght_mean, height_std, xy_scale, stairStepLen, stairStepWidth, stairStepHeight, Kp, Kd):
            Initialises attributes for the termination and truncation conditions, the range of forward velocity 
            commands, initial variable step parameters, constant step parameters and PD controller gains. Defines the 
            continuous observation and action spaces. Connects to PyBullet in either direct mode (no 
            rendering) or human mode (render simulation in GUI) and creates the specified terrain.

        reset(seed, options, bodyStartPos):
            Starts a new episode. Recreates simulation environment (if ROUGH then it will be a different random Gaussian 
            distributed heightfield each episode) and reloads PAWS URDF. Sets starting position. Sets forward velocity 
            command for episode according to a uniform distribution. Creates the first stage of the first cycle of the 
            trot gait deterministically using the specified initial conditions. Simulates the first stage of the first 
            cycle of trot. Returns the observations and info dict.

        step(action):
            Denormalises action selected by policy to get the step parameters of the next stage of the trot cycle, 
            including fwdSwgLen, latSwgLen, swgHght and Tswg. Generate the foot trajectories for the next stage of the 
            trot cycle using the model-based trot trajectory generator. Simulate the next stage of the trot cycle, 
            claculating the reward throughout. Gets the observations and calculate total reward. Logs simulation data to 
            a CSV file if render mode is human. Checks for termination and truncation conditions. Returns observations, 
            reward, terminated flag, truncated flag and info dict.

        close():
            Disconnects from PyBullet.

        _getObs():
            Uses PyBullet functions to collect and return the state observations from the environment, according to the 
            MDP.

        _createPyBulletEnv(type, height_mean, height_std_dev, xy_scale, stepLen, stepWidth, stepHeight):
            Creates a simulation environment with the specified terrain type and parameters.
            
        _genTrotStep(forwardSwingLength, latSwingLength, swingHeight, latStanceLength, T_swing, h_0, sampleRate, 
                     swingX_05, stanceX_0, swing1Y_0, swing2Y_0, stance1Y_0, stance2Y_0, backSwingCoef, hipOffset):
            Returns the trajectories of all four legs for one stage of the trot cycle, that is the swing trajectory for 
            2 legs and the stance trajectory for the other 2 (which is proportional to the swing trajectory). The 
            trajectories are constructed using splines of quintic polynomials and a cycloid curve, as per the report.

        _calculateIK(targetFootPosition, kneeLagging, L1=0.06, L2=0.15, L3=0.21)
            Returns the joint angles required to place a single foot at a desired position relative to the 
            corresponding hip and a boolean error flag which is True if there was an attempted work-space violation and 
            False otherwise. 

        _calculateFK(jointPositions, L1=0.05955, L2=0.15, L3=0.21)
            Takes the joint positions of a single leg of PAWS and returns the Cartesian position of the foot relative 
            to the corresponding hip.

        _calRelFootVel(jointPositions, jointVelocities, L1=0.05955, L2=0.15, L3=0.21)
            Takes the joint positions and velocities of a single leg and returns the velocities of the foot in the 
            x-direction (forward/backward direction), y-direction (sideways direction) and z-direction (vertical 
            direction), relative to the corresponding hip.

        _collectSimData():
            Measures the current state of the simulated PAWS robot and appends it to a Pandas dataframe for later 
            loging to a CSV file.
    """""
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, renderMode=None, maxXdist=5, truncationSteps=300, max_tilt=np.pi/3, maxFwdVel=1.0, minFwdVel=0.5,
                 initFwdSwgLen=0.08, initLatSwgLen=0.0, initSwgHght=0.04, initTswg=0.2, latStanceLength=0.0, h_0=-0.28,
                 sampleRate=240, backSwingCoef=0.2, hipOffset=0.05955, terrain="FLAT", hieght_mean=0.1, height_std=0.01,
                 xy_scale=0.05, stairStepLen=0.2, stairStepWidth=2, stairStepHeight=0.02, Kp=0.15, Kd=0.55):
        """""
        Initialises attributes for the termination and truncation conditions, the range of forward velocity commands, 
        initial variable step parameters, constant step parameters and PD controller gains. Defines the continuous 
        observation and action spaces. Connects to PyBullet in either direct mode (no rendering) or human mode (render 
        simulation in GUI) and creates the specified terrain.
        Args:
            renderMode (str): Asserted in [None, "human"]. Decides whether to connect to PyBullet in direct mode with no 
                              rendering or in human mode to render the simulation in the GUI. Direct mode is used for 
                              training. Human mode is uded for testing.
            maxXdist (float): Maximum x distance before episode is terminated (success).
            truncationSteps (int): Maximum number of timestepd before episode is truncated.
            max_tilt (float): Maximum tilt in roll or pitch angle before episode is terminated (failure).
            maxFwdVel (float): Minimum forward velocity command.
            minFwdVel (float): Maximum forward velocity command.
            initFwdSwgLen (float): Initial forward swing length in x-direction relative to the corresponding hip.
            initLatSwgLen (float): Initial lateral swing length in y-direction relative to the corresponding hip.
            initSwgHght (float): Initial maximum height of swing phase relative to the stance height. 
            initTswg (float): Initial swing period.
            latStanceLength (float): Maximum y-lean during stance phase. Recommended 0 m.
            h_0 (float): z-position of feet relative to hip during the stance phase (i.e. - body height).
            sampleRate (float): Sample rate, recommended 240 Hz.
            backSwingCoef: (float): back/forward swing coefficient, default 0.2.
            hipOffset: (float): 0 m for feet directly below motors, L1 m for legs straight (recommended).
            terrain (str): type of terrain, is asserted in ["FLAT", "ROUGH", "STAIRS"].
            hieght_mean (float): mean height of terrain if terrain==ROUGH.
            height_std (float): standard deviation of terrain if terrain==ROUGH.
            xy_scale (float): how big each square is if terrain==ROUGH.
            stairStepLen (float): length in x-direction of each step if terrain==STAIRS.
            stairStepWidth (float): width in y-direction of each step if terrain==STAIRS.
            stairStepHeight(float): height in z-direction of each step if terrain==STAIRS.
            Kp (float): Proportional gain of the PD controller.
            Kd (float): Derivative gain of the PD contriller.
        Returns:
            None
        """""
        super().__init__()  # Call the constructor of the parent class gym.Env

        ### Initialise attributes ###
        assert renderMode is None or renderMode in self.metadata['render_modes']
        self._renderMode = renderMode
        self.maxXdist = maxXdist
        self.truncationSteps = truncationSteps
        self.max_tilt = max_tilt
        self.maxVel = maxFwdVel
        self.minVel = minFwdVel
        self.initFwdSwgLen = initFwdSwgLen
        self.initLatSwgLen = initLatSwgLen
        self.initSwgHght = initSwgHght
        self.initTswg = initTswg
        self.latStanceLength = latStanceLength
        self.h_0 = h_0
        self.sampleRate = sampleRate
        self.backSwingCoef = backSwingCoef
        self.hipOffset = hipOffset
        self.terrain = terrain
        self.hieght_mean = hieght_mean
        self.height_std = height_std
        self.xy_scale = xy_scale
        self.stairStepLen = stairStepLen
        self.stairStepWidth = stairStepWidth
        self.stairStepHeight = stairStepHeight
        self.Kp = Kp
        self.Kd = Kd

        ### Define joint indexes as attributes ###
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

        self.filename = r"C:\Users\Ronan\PawsQuadrupedRobot\log.csv"  # File directory for logs

        ############################################# Define Action Space ##############################################
        # The actions are:
        # [forwardSwingLength, lateralSwingLength, swingHeight, T_swing]
        #  These actions will be normalised to the range [-1, 1] and denormalised before they are applied to the robot.

        # Action space normalised in the range [-1, 1] for better convergence. When actions are not normalized, the
        # learning algorithm may struggle to converge as actions with larger magnitudes may leading to unstable updates.
        # Normalization also helps the agent explore the action space more evenly.
        self.action_space = spaces.Box(-np.ones(4), np.ones(4), dtype=np.float32)

        ### Define the min and max values used for denormalisation ###
        minFwdSwgLen = 0.0
        minLatSwgLen = -0.03
        minSwgHght = 0.03
        minTswg = 0.1

        maxFwdSwgLen = 0.15
        maxLatSwgLen = 0.03
        maxSwgHght = 0.1
        maxTswg = 0.5

        self.actionLowerLimits = np.array([minFwdSwgLen, minLatSwgLen, minSwgHght, minTswg])
        self.actionUpperLimits = np.array([maxFwdSwgLen, maxLatSwgLen, maxSwgHght, maxTswg])

        ########################################### Define Observation Space ###########################################
        # The State Observation is:
        # [COGrollAngle, COGpitchAngle, COGyawAngle,
        #  COGrollRate, COGpitchRate, COGyawRate,
        #  GOGxVelocity, COGyVelocity, COGzVelocity,
        #  xRelPosFR, yRelPosFR, zRelPosFR, xRelPosRL, yRelPosRL, zRelPosRL,
        #  xRelPosFL, yRelPosFL, zRelPosFL, xRelPosRR, yRelPosRR, zRelPosRR,
        #  xVelFR, yVelFR, zVelFR, xVelRL, yVelRL, zVelRL,
        #  xVelFL, yVelFL, zVelFL, xVelRR, yVelRR, zVelRR,
        #  desiredVel, stage, prevFwdSwgLen, prevLatSwgLen, prevSwgHght, prevTswg]

        ### Constrain observations of relative foot position to the workspace of each foot ###
        self.xMaxRelPos = 0.1
        self.yMaxRelPos = 0.04
        self.zMaxRelPos = -0.1
        self.zMinRelPos = 0.3

        ### Define lower and upper limits on the observation space ###
        stateUpperLimits = np.array([max_tilt, max_tilt, np.pi,
                                     np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                                     np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                                     self.xMaxRelPos, self.yMaxRelPos, self.zMaxRelPos, self.xMaxRelPos,
                                     self.yMaxRelPos, self.zMaxRelPos, self.xMaxRelPos, self.yMaxRelPos,
                                     self.zMaxRelPos, self.xMaxRelPos, self.yMaxRelPos, self.zMaxRelPos,
                                     np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                                     np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                                     np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                                     np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                                     self.maxVel, 1, maxFwdSwgLen, maxLatSwgLen, maxSwgHght, maxTswg])

        stateLowerLimits = -1 * np.array([max_tilt, max_tilt, np.pi,
                                          np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                                          np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                                          self.xMaxRelPos, self.yMaxRelPos, self.zMinRelPos, self.xMaxRelPos,
                                          self.yMaxRelPos, self.zMinRelPos, self.xMaxRelPos, self.yMaxRelPos,
                                          self.zMinRelPos, self.xMaxRelPos, self.yMaxRelPos, self.zMinRelPos,
                                          np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                                          np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                                          np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                                          np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                                          -self.minVel, 0, -minFwdSwgLen, -minLatSwgLen, -minSwgHght, -minTswg])

        # The Gymnassium function Box() is used for continuous spaces
        self.observation_space = spaces.Box(stateLowerLimits, stateUpperLimits, dtype=np.float32)

        ############################################# Connect to PyBullet ##############################################
        if self._renderMode == 'human':
            fig = plt.figure(1)  # Calling this first makes the GUI higher resolution
            self.physicsClient = p.connect(p.GUI)  # Connect to PyBullet in render mode
        else:
            self.physicsClient = p.connect(p.DIRECT)  # Connect to PyBullet in direct mode with no GUI (for training)

        assert self.terrain in ["FLAT", "RANDOM_ROUGH", "STAIRS"]

        self._createPyBulletEnv(self.terrain, self.hieght_mean, self.height_std, self.xy_scale, self.stairStepLen,
                                self.stairStepWidth, self.stairStepHeight)
        self.planeId = p.loadURDF("plane.urdf")
        self.simPaws = p.loadURDF(r'PawsURDF/urdf/Paws.urdf', [0, 0, 0.4], useFixedBase=0)

        return


    def reset(self, seed=None, options=None, bodyStartPos=[0, 0, 0.49]):
        """""
        Starts a new episode. Recreates simulation environment (if ROUGH then it will be a different random Gaussian 
        distributed heightfield each episode) and reloads PAWS URDF. Sets starting position. Sets forward velocity 
        command for episode according to a uniform distribution. Creates the first stage of the first cycle of the 
        trot gait deterministically using the specified initial conditions. Simulates the first stage of the first 
        cycle of trot. Returns the observations and info dict.
        Args:
            seed (int): Seed for random number generators to ensure reproducibility. Required by Gymnasium API 
                        interface. Defaults to None.
            options (dict): Required by Gymnasium API interface. Defaults to None.
            bodyStartPos (list[float]): Start position of the body of PAWS in world coordinates.
        Returns:
            tuple:
                - observation (np.ndarray): State observations according to MDP.
                - info (dict): Required by Gymnassium API interface. Just an empty dict.
        """""
        super().reset(seed=seed)  # Call the reset function of the parent class gym.Env

        ### Reset termination and truncation flags ###
        self.terminated = False
        self.truncated = False

        ### Reset PyBullet environment ###
        p.removeBody(self.simPaws)
        self._createPyBulletEnv(self.terrain, self.hieght_mean, self.height_std, self.xy_scale, self.stairStepLen,
                                self.stairStepWidth, self.stairStepHeight)
        self.planeId = p.loadURDF("plane.urdf")
        self.simPaws = p.loadURDF(r'PawsURDF/urdf/Paws.urdf', bodyStartPos, useFixedBase=0)

        self.desiredVel = np.random.uniform(low=self.minVel, high=self.maxVel)  # Set the desired forward velocity for
                                                                                # the episode
        # print(self.desiredVel)

        ### Create initial swing trajectory deterministically ###

        self.swingX_0 = -self.initFwdSwgLen / 2   # Initial position of swing feet in x-direction
        self.stanceX_0 = self.initFwdSwgLen / 2   # Initial position of stance feet in x-direction
        self.swing1Y_0 = 0   # Initial position of swing foot 1 in y-direction
        self.swing2Y_0 = 0   # Initial position of swing foot 2 in y-direction
        self.stance1Y_0 = 0  # Initial position of stance foot 1 in y-direction
        self.stance2Y_0 = 0  # Initial position of stance foot 2 in y-direction

        # Generate foot trajectories for the first swing phase
        self.feetTrajectories = self._genTrotStep(self.initFwdSwgLen, self.initLatSwgLen, self.initSwgHght,
                                                  self.latStanceLength, self.initTswg, self.h_0, self.sampleRate,
                                                  self.swingX_0, self.stanceX_0, self.swing1Y_0, self.swing2Y_0,
                                                  self.stance1Y_0, self.stance2Y_0, self.backSwingCoef,
                                                  self.hipOffset)

        self.samplesPerSwing = self.feetTrajectories[0].shape[0]  # Determine the number of samples per swing
        self.T_swing = self.initTswg

        # Initialise action history
        self.prevFwdSwgLen = self.initFwdSwgLen
        self.prevLatSwgLen = self.initLatSwgLen
        self.prevSwgHght = self.initSwgHght
        self.prevTswg = self.T_swing

        # Calculate initial joint positions for all legs using IK
        FR_targetFootPosition = self.feetTrajectories[0][0]  # Initially swing
        FR_targetJointPositions, _ = self._calculateIK(FR_targetFootPosition, kneeLagging=True)
        RL_targetFootPosition = self.feetTrajectories[1][0]  # Initially swing
        RL_targetJointPositions, _ = self._calculateIK(RL_targetFootPosition, kneeLagging=True)
        RR_targetFootPosition = self.feetTrajectories[2][0]  # Initially stance
        RR_targetJointPositions, _ = self._calculateIK(RR_targetFootPosition, kneeLagging=True)
        FL_targetFootPosition = self.feetTrajectories[3][0]  # Initially stance
        FL_targetJointPositions, _ = self._calculateIK(FL_targetFootPosition, kneeLagging=True)

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

        if self._renderMode == 'human':
            self.firstStepLog = True  # The first log creates a dataframe, subsequent logs append data
            self.firstEpisodeLog = True  # The first log creates a CSV file, subsequent logs append data
            self.logCounter = 0
            self._collectSimData()  # Collect first log data
            self.cog_df.to_csv(self.filename, index=False)  # Create CSV
            self.cog_df.drop(self.cog_df.index, inplace=True)

            self.camCounter = 0  # Counter to move debug visualiser camera as PAWS moves forwards
            self.startTime = time.perf_counter()  # Store start time
            base_Pos, baseOrient = p.getBasePositionAndOrientation(self.simPaws)
            base_Pos = np.array(base_Pos)
            p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=27.2, cameraPitch=-20.4,
                                         cameraTargetPosition=base_Pos)

        self.trotStage = 0
        self.stepCounter = 0

        for j in range(self.samplesPerSwing):  # For each sample in the current stage of the trot cycle

            if self.trotStage == 0:
                k = 0  # FR in swing
            elif self.trotStage == 1:
                k = 2  # FR in stance
            FR_targetFootPosition = self.feetTrajectories[k][j]  # Desired FR task-space position
            # Desired FR joint-space positions
            FR_targetJointPositions, errorFR = self._calculateIK(FR_targetFootPosition, kneeLagging=True)

            if self.trotStage == 0:
                k = 1  # RL in swing
            elif self.trotStage == 1:
                k = 3  # RL in stance
            RL_targetFootPosition = self.feetTrajectories[k][j]  # Desired RL task-space position
            # Desired RL joint-space positions
            RL_targetJointPositions, errorRL = self._calculateIK(RL_targetFootPosition, kneeLagging=True)

            if self.trotStage == 0:
                k = 2  # RL in stance
            elif self.trotStage == 1:
                k = 0  # RL in swing
            RR_targetFootPosition = self.feetTrajectories[k][j]  # Desired RR task-space position
            # Desired RR joint-space positions
            RR_targetJointPositions, errorRR = self._calculateIK(RR_targetFootPosition, kneeLagging=True)

            if self.trotStage == 0:
                k = 3  # FL in stance
            elif self.trotStage == 1:
                k = 1  # FL in swing
            FL_targetFootPosition = self.feetTrajectories[k][j]  # Desired FL task-space position
            # Desired FL joint-space positions
            FL_targetJointPositions, errorFL = self._calculateIK(FL_targetFootPosition, kneeLagging=True)

            ### Send commands to all motors ###
            p.setJointMotorControl2(self.simPaws, self.FR_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=FR_targetJointPositions[0],
                                    positionGain=self.Kp, velocityGain=self.Kd)
            p.setJointMotorControl2(self.simPaws, self.FR_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=FR_targetJointPositions[1],
                                    positionGain=self.Kp, velocityGain=self.Kd)
            p.setJointMotorControl2(self.simPaws, self.FR_KNEE_JOINT, p.POSITION_CONTROL,
                                    targetPosition=FR_targetJointPositions[2],
                                    positionGain=self.Kp, velocityGain=self.Kd)

            p.setJointMotorControl2(self.simPaws, self.RL_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=RL_targetJointPositions[0],
                                    positionGain=self.Kp, velocityGain=self.Kd)
            p.setJointMotorControl2(self.simPaws, self.RL_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=-RL_targetJointPositions[1],
                                    positionGain=self.Kp, velocityGain=self.Kd)
            p.setJointMotorControl2(self.simPaws, self.RL_KNEE_JOINT, p.POSITION_CONTROL,
                                    targetPosition=RL_targetJointPositions[2],
                                    positionGain=self.Kp, velocityGain=self.Kd)

            p.setJointMotorControl2(self.simPaws, self.RR_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=-RR_targetJointPositions[0],
                                    positionGain=self.Kp, velocityGain=self.Kd)
            p.setJointMotorControl2(self.simPaws, self.RR_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=RR_targetJointPositions[1],
                                    positionGain=self.Kp, velocityGain=self.Kd)
            p.setJointMotorControl2(self.simPaws, self.RR_KNEE_JOINT, p.POSITION_CONTROL,
                                    targetPosition=RR_targetJointPositions[2],
                                    positionGain=self.Kp, velocityGain=self.Kd)

            p.setJointMotorControl2(self.simPaws, self.FL_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=-FL_targetJointPositions[0],
                                    positionGain=self.Kp, velocityGain=self.Kd)
            p.setJointMotorControl2(self.simPaws, self.FL_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=-FL_targetJointPositions[1],
                                    positionGain=self.Kp, velocityGain=self.Kd)
            p.setJointMotorControl2(self.simPaws, self.FL_KNEE_JOINT, p.POSITION_CONTROL,
                                    targetPosition=FL_targetJointPositions[2],
                                    positionGain=self.Kp, velocityGain=self.Kd)

            if self._renderMode == 'human':
                elapsedTime = time.perf_counter() - self.startTime

                while elapsedTime < self.T_swing / self.sampleRate:  # Delay until next sample time
                    elapsedTime = time.perf_counter() - self.startTime
                self.startTime = time.perf_counter()

            p.stepSimulation()

            if self._renderMode == 'human':
                self.logCounter += 1
                self._collectSimData()

        if self._renderMode == 'human':
            self.cog_df.to_csv(self.filename, mode='a', index=False, header=False)
            self.cog_df.drop(self.cog_df.index, inplace=True)

        ### Update the start position of the feet for the next stage of the trot cycle ###
        self.swingX_0 = self.feetTrajectories[2][self.samplesPerSwing - 1][0]
        self.stanceX_0 = self.feetTrajectories[0][self.samplesPerSwing - 1][0]
        self.swing1Y_0 = self.feetTrajectories[2][self.samplesPerSwing - 1][1] - self.hipOffset
        self.swing2Y_0 = self.feetTrajectories[3][self.samplesPerSwing - 1][1] - self.hipOffset
        self.stance1Y_0 = self.feetTrajectories[0][self.samplesPerSwing - 1][1] - self.hipOffset
        self.stance2Y_0 = self.feetTrajectories[1][self.samplesPerSwing - 1][1] - self.hipOffset

        self.trotStage = self.trotStage + 1  # Move to the next stage of trot
        if self.trotStage == 2:
            self.trotStage = 0

        ### Sign corrections for each stage ###
        if self.trotStage == 1:
            self.latStanceLength = -self.latStanceLength

        observation = self._getObs()
        info = {}

        return observation, info


    def step(self, action):
        """""
        Denormalises action selected by policy to get the step parameters of the next stage of the trot cycle, 
        including fwdSwgLen, latSwgLen, swgHght and Tswg. Generate the foot trajectories for the next stage of the 
        trot cycle using the model-based trot trajectory generator. Simulate the next stage of the trot cycle, 
        claculating the reward throughout. Gets the observations and calculate total reward. Logs simulation data to 
        a CSV file if render mode is human. Checks for termination and truncation conditions. Returns observations, 
        reward, terminated flag, truncated flag and info dict.
        Args:
            action (np.ndarray): Action selected by the policy for the current timestep, which is an array of the 
                                 variable step parameters (forward swing length, lateral swing length, swing height and
                                 swing period) for the next stage of the trot cycle, where each element is normalised to 
                                 the range [-1, 1]. 
        Returns:
            tuple:
                - observation (np.ndarray): State observations according to MDP.
                - reward (float): Reward for the action just taken.
                - terminated (bool): True if termination conditions are met, else False.
                - truncated (bool): True if truncation conditions are met, else False.
                - info (dict): Required by Gymnassium API interface. Just an empty dict.
        """""
        ### Separate and Denormalise Action ###
        fwdSwgLen = (self.actionLowerLimits[0] * (1 - action[0]) + self.actionUpperLimits[0] * (1 + action[0])) / 2
        latSwgLen = (self.actionLowerLimits[1] * (1 - action[1]) + self.actionUpperLimits[1] * (1 + action[1])) / 2
        swgHght = (self.actionLowerLimits[2] * (1 - action[2]) + self.actionUpperLimits[2] * (1 + action[2])) / 2
        Tswg = (self.actionLowerLimits[3] * (1 - action[3]) + self.actionUpperLimits[3] * (1 + action[3])) / 2

        ### Create foot trajectories for next stage of the trot gait ###
        self.feetTrajectories = self._genTrotStep(fwdSwgLen, latSwgLen, swgHght, self.latStanceLength, Tswg, self.h_0,
                                                  self.sampleRate, self.swingX_0, self.stanceX_0, self.swing1Y_0,
                                                  self.swing2Y_0, self.stance1Y_0, self.stance2Y_0, self.backSwingCoef,
                                                  self.hipOffset)

        self.samplesPerSwing = self.feetTrajectories[0].shape[0]
        self.T_swing = Tswg

        ### Initialise reward terms which will be updated throughout the current stage of the trot cycle ###
        p_wsv = 0
        r_vel = 0
        p_bs = 0

        ### Define reward weights - alter during training for cirriculum learning ###
        w_bs = 10.0
        w_s = 15.0

        if self._renderMode == 'human':

            base_Pos, baseOrient = p.getBasePositionAndOrientation(self.simPaws)
            base_Pos = np.array(base_Pos)

            # If the x-position of the base has travelled far enough forwards, step the debug camera
            if base_Pos[0] >= 2 * self.camCounter:
                self.camCounter += 1
                base_Pos = base_Pos + np.array([1, 0, 0])
                p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=27.2, cameraPitch=-20.4,
                                             cameraTargetPosition=base_Pos)

        for j in range(self.samplesPerSwing):  # For each sample in the current stage of the trot cycle

            if self.trotStage == 0:
                k = 0  # FR in swing
            elif self.trotStage == 1:
                k = 2  # FR in stance
            FR_targetFootPosition = self.feetTrajectories[k][j]  # Desired FR task-space position
            # Desired FR joint-space positions
            FR_targetJointPositions, errorFR = self._calculateIK(FR_targetFootPosition, kneeLagging=True)

            if self.trotStage == 0:
                k = 1  # RL in swing
            elif self.trotStage == 1:
                k = 3  # RL in stance
            RL_targetFootPosition = self.feetTrajectories[k][j]  # Desired RL task-space position
            # Desired RL joint-space positions
            RL_targetJointPositions, errorRL = self._calculateIK(RL_targetFootPosition, kneeLagging=True)

            if self.trotStage == 0:
                k = 2  # RL in stance
            elif self.trotStage == 1:
                k = 0  # RL in swing
            RR_targetFootPosition = self.feetTrajectories[k][j]  # Desired RR task-space position
            # Desired RR joint-space positions
            RR_targetJointPositions, errorRR = self._calculateIK(RR_targetFootPosition, kneeLagging=True)

            if self.trotStage == 0:
                k = 3  # FL in stance
            elif self.trotStage == 1:
                k = 1  # FL in swing
            FL_targetFootPosition = self.feetTrajectories[k][j]  # Desired FL task-space position
            # Desired FL joint-space positions
            FL_targetJointPositions, errorFL = self._calculateIK(FL_targetFootPosition, kneeLagging=True)

            ### Send commands to all motors ###
            p.setJointMotorControl2(self.simPaws, self.FR_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=FR_targetJointPositions[0],
                                    positionGain=self.Kp, velocityGain=self.Kd)
            p.setJointMotorControl2(self.simPaws, self.FR_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=FR_targetJointPositions[1],
                                    positionGain=self.Kp, velocityGain=self.Kd)
            p.setJointMotorControl2(self.simPaws, self.FR_KNEE_JOINT, p.POSITION_CONTROL,
                                    targetPosition=FR_targetJointPositions[2],
                                    positionGain=self.Kp, velocityGain=self.Kd)

            p.setJointMotorControl2(self.simPaws, self.RL_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=RL_targetJointPositions[0],
                                    positionGain=self.Kp, velocityGain=self.Kd)
            p.setJointMotorControl2(self.simPaws, self.RL_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=-RL_targetJointPositions[1],
                                    positionGain=self.Kp, velocityGain=self.Kd)
            p.setJointMotorControl2(self.simPaws, self.RL_KNEE_JOINT, p.POSITION_CONTROL,
                                    targetPosition=RL_targetJointPositions[2],
                                    positionGain=self.Kp, velocityGain=self.Kd)

            p.setJointMotorControl2(self.simPaws, self.RR_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=-RR_targetJointPositions[0],
                                    positionGain=self.Kp, velocityGain=self.Kd)
            p.setJointMotorControl2(self.simPaws, self.RR_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=RR_targetJointPositions[1],
                                    positionGain=self.Kp, velocityGain=self.Kd)
            p.setJointMotorControl2(self.simPaws, self.RR_KNEE_JOINT, p.POSITION_CONTROL,
                                    targetPosition=RR_targetJointPositions[2],
                                    positionGain=self.Kp, velocityGain=self.Kd)

            p.setJointMotorControl2(self.simPaws, self.FL_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=-FL_targetJointPositions[0],
                                    positionGain=self.Kp, velocityGain=self.Kd)
            p.setJointMotorControl2(self.simPaws, self.FL_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                    targetPosition=-FL_targetJointPositions[1],
                                    positionGain=self.Kp, velocityGain=self.Kd)
            p.setJointMotorControl2(self.simPaws, self.FL_KNEE_JOINT, p.POSITION_CONTROL,
                                    targetPosition=FL_targetJointPositions[2],
                                    positionGain=self.Kp, velocityGain=self.Kd)

            if errorFR or errorRL or errorFL or errorRR:  # If there is a work-space violation
                p_wsv = -50
                break

            if self._renderMode == 'human':
                elapsedTime = time.perf_counter() - self.startTime

                while elapsedTime < self.T_swing / self.sampleRate:  # Delay until next sample time
                    elapsedTime = time.perf_counter() - self.startTime
                self.startTime = time.perf_counter()

            p.stepSimulation()

            basePos, baseOrient = p.getBasePositionAndOrientation(self.simPaws)
            baseOrient = p.getEulerFromQuaternion(baseOrient)
            [baseLinVel, baseAngVel] = p.getBaseVelocity(self.simPaws)

            rollAngle = baseOrient[0]
            pitchAngle = baseOrient[1]
            yawAngle = baseOrient[2]
            xPosCOG = basePos[0]
            zPosCOG = basePos[2]
            xVelCOG = baseLinVel[0]

            # Update forward velocity tracking reward (r_vel)
            r_vel += self.maxVel * np.exp(-(xVelCOG - self.desiredVel) ** 2 / (2 * self.maxVel ** 2))

            # Update base stability penalty (p_bs)
            p_bs += -w_bs * (rollAngle ** 2 + pitchAngle ** 2 + yawAngle ** 2)

            if self._renderMode == 'human':
                self.logCounter += 1
                self._collectSimData()

        if self._renderMode == 'human':
            self.cog_df.to_csv(self.filename, mode='a', index=False, header=False)
            self.cog_df.drop(self.cog_df.index, inplace=True)

        ### Update the start position of the feet for the next stage of the trot cycle ###
        self.swingX_0 = self.feetTrajectories[2][self.samplesPerSwing - 1][0]
        self.stanceX_0 = self.feetTrajectories[0][self.samplesPerSwing - 1][0]
        self.swing1Y_0 = self.feetTrajectories[2][self.samplesPerSwing - 1][1] - self.hipOffset
        self.swing2Y_0 = self.feetTrajectories[3][self.samplesPerSwing - 1][1] - self.hipOffset
        self.stance1Y_0 = self.feetTrajectories[0][self.samplesPerSwing - 1][1] - self.hipOffset
        self.stance2Y_0 = self.feetTrajectories[1][self.samplesPerSwing - 1][1] - self.hipOffset

        self.trotStage = self.trotStage + 1  # Move to the next stage of trot
        if self.trotStage == 2:
            self.trotStage = 0

        ### Sign corrections for each stage ###
        if self.trotStage == 1:
            self.latStanceLength = -self.latStanceLength

        ############################################### Get Observations ###############################################
        # The observations are:
        # [COGrollAngle, COGpitchAngle, COGyawAngle,
        #  COGrollRate, COGpitchRate, COGyawRate,
        #  GOGxVelocity, COGyVelocity, COGzVelocity,
        #  xRelPosFR, yRelPosFR, zRelPosFR, xRelPosRL, yRelPosRL, zRelPosRL,
        #  xRelPosFL, yRelPosFL, zRelPosFL, xRelPosRR, yRelPosRR, zRelPosRR,
        #  xVelFR, yVelFR, zVelFR, xVelRL, yVelRL, zVelRL,
        #  xVelFL, yVelFL, zVelFL, xVelRR, yVelRR, zVelRR,
        #  desiredVel, stage, prevFwdSwgLen, prevLatSwgLen, prevSwgHght, prevTswg]

        observation = self._getObs()

        ############################################### Calculate Reward ###############################################
        # Smoothness penalty
        p_s = -w_s * ((fwdSwgLen-self.prevFwdSwgLen)**2 + (latSwgLen-self.prevLatSwgLen)**2 + \
                      (swgHght-self.prevSwgHght)**2 + (Tswg-self.prevTswg)**2)

        r_vel = r_vel / self.samplesPerSwing  # Average over the last trot stage
        p_bs = p_bs / self.samplesPerSwing    # Average over the last trot stage

        reward = r_vel + p_bs + p_wsv + p_s

        ################################################### Get Info ###################################################

        info = {}

        ############################################ Truncation Conditions #############################################

        self.stepCounter += 1
        if self.stepCounter >= self.truncationSteps:
            self.truncated = True
            # print("\n\n\n Truncated!!! \n\n\n")
        else:
            self.truncated = False

        ############################################ Termination Conditions ############################################

        if abs(rollAngle) >= self.max_tilt or abs(pitchAngle) >= self.max_tilt or zPosCOG <= 0.1 or \
                np.isnan(FR_targetJointPositions).any() or np.isnan(RL_targetJointPositions).any() or \
                np.isnan(FL_targetJointPositions).any() or np.isnan(RR_targetJointPositions).any() or \
                p_wsv < 0 or xPosCOG >= self.maxXdist:

            self.terminated = True
            # print("\n\n\n Terminated!!! \n\n\n")
        else:
            self.terminated = False

        self.prevFwdSwgLen = fwdSwgLen
        self.prevLatSwgLen = latSwgLen
        self.prevSwgHght = swgHght
        self.prevTswg = Tswg

        return observation, reward, self.terminated, self.truncated, info


    def close(self):
        """""
        Disconnects from PyBullet.
        Args:
            None
        Returns:
            None
        """""
        if self.physicsClient >= 0:
            p.disconnect()
        self.physicsClient = -1
        print("Env closed.")
        return


    def _getObs(self):
        """""
        Uses PyBullet functions to collect and return the state observations from the environment.
        Args:
            None
        Returns:
            observation (np.ndarray): State observations according to MDP, which are:                                       
                                      [COGrollAngle, COGpitchAngle, COGyawAngle,
                                       COGrollRate, COGpitchRate, COGyawRate,
                                       GOGxVelocity, COGyVelocity, COGzVelocity,
                                       xRelPosFR, yRelPosFR, zRelPosFR, xRelPosRL, yRelPosRL, zRelPosRL,
                                       xRelPosFL, yRelPosFL, zRelPosFL, xRelPosRR, yRelPosRR, zRelPosRR,
                                       xVelFR, yVelFR, zVelFR, xVelRL, yVelRL, zVelRL,
                                       xVelFL, yVelFL, zVelFL, xVelRR, yVelRR, zVelRR,
                                       desiredVel, stage, prevFwdSwgLen, prevLatSwgLen, prevSwgHght, prevTswg]
        """""
        basePos, baseOrient = p.getBasePositionAndOrientation(self.simPaws)
        baseOrient = p.getEulerFromQuaternion(baseOrient)

        [baseLinVel, baseAngVel] = p.getBaseVelocity(self.simPaws)

        ### Get all joint angles ###
        jointAnglesFR = np.array(p.getJointStates(self.simPaws, [self.FR_HIP_ABDUCTOR_ADDUCTOR_JOINT,
                                                                 self.FR_HIP_FLEXOR_EXTENDOR_JOINT,
                                                                 self.FR_KNEE_JOINT]), dtype=object)[:, 0]

        jointAnglesRL = np.array(p.getJointStates(self.simPaws, [self.RL_HIP_ABDUCTOR_ADDUCTOR_JOINT,
                                                                 self.RL_HIP_FLEXOR_EXTENDOR_JOINT,
                                                                 self.RL_KNEE_JOINT]), dtype=object)[:, 0]
        jointAnglesRL = np.array([-1, -1, 1]) * jointAnglesRL  # Sign corrections for RL

        jointAnglesFL = np.array(p.getJointStates(self.simPaws, [self.FL_HIP_ABDUCTOR_ADDUCTOR_JOINT,
                                                                 self.FL_HIP_FLEXOR_EXTENDOR_JOINT,
                                                                 self.FL_KNEE_JOINT]), dtype=object)[:, 0]
        jointAnglesFL = np.array([-1, -1, 1]) * jointAnglesFL  # Sign corrections for FL

        jointAnglesRR = np.array(p.getJointStates(self.simPaws, [self.RR_HIP_ABDUCTOR_ADDUCTOR_JOINT,
                                                                 self.RR_HIP_FLEXOR_EXTENDOR_JOINT,
                                                                 self.RR_KNEE_JOINT]), dtype=object)[:, 0]
        jointAnglesRR = np.array([-1, 1, 1]) * jointAnglesRR  # Sign corrections for RR

        ### Calculate relative foot position of each foot using forward kinematics ###
        footPosFR = self._calculateFK(jointAnglesFR)
        footPosFR = np.clip(footPosFR, [-self.xMaxRelPos, -self.yMaxRelPos, -self.zMinRelPos],
                            [self.xMaxRelPos, self.yMaxRelPos, self.zMaxRelPos])

        footPosRL = self._calculateFK(jointAnglesRL)
        footPosRL = np.clip(footPosRL, [-self.xMaxRelPos, -self.yMaxRelPos, -self.zMinRelPos],
                            [self.xMaxRelPos, self.yMaxRelPos, self.zMaxRelPos])

        footPosFL = self._calculateFK(jointAnglesFL)
        footPosFL = np.clip(footPosFL, [-self.xMaxRelPos, -self.yMaxRelPos, -self.zMinRelPos],
                            [self.xMaxRelPos, self.yMaxRelPos, self.zMaxRelPos])

        footPosRR = self._calculateFK(jointAnglesRR)
        footPosRR = np.clip(footPosRR, [-self.xMaxRelPos, -self.yMaxRelPos, -self.zMinRelPos],
                            [self.xMaxRelPos, self.yMaxRelPos, self.zMaxRelPos])

        ### Get all joint velocities ###
        jointVelocitiesFR = np.array(p.getJointStates(self.simPaws, [self.FR_HIP_ABDUCTOR_ADDUCTOR_JOINT,
                                                                     self.FR_HIP_FLEXOR_EXTENDOR_JOINT,
                                                                     self.FR_KNEE_JOINT]), dtype=object)[:, 1]

        jointVelocitiesRL = np.array(p.getJointStates(self.simPaws, [self.RL_HIP_ABDUCTOR_ADDUCTOR_JOINT,
                                                                     self.RL_HIP_FLEXOR_EXTENDOR_JOINT,
                                                                     self.RL_KNEE_JOINT]), dtype=object)[:, 1]
        jointVelocitiesRL = np.array([-1, -1, 1]) * jointVelocitiesRL

        jointVelocitiesFL = np.array(p.getJointStates(self.simPaws, [self.FL_HIP_ABDUCTOR_ADDUCTOR_JOINT,
                                                                     self.FL_HIP_FLEXOR_EXTENDOR_JOINT,
                                                                     self.FL_KNEE_JOINT]), dtype=object)[:, 1]
        jointVelocitiesFL = np.array([-1, -1, 1]) * jointVelocitiesFL

        jointVelocitiesRR = np.array(p.getJointStates(self.simPaws, [self.RR_HIP_ABDUCTOR_ADDUCTOR_JOINT,
                                                                     self.RR_HIP_FLEXOR_EXTENDOR_JOINT,
                                                                     self.RR_KNEE_JOINT]), dtype=object)[:, 1]
        jointVelocitiesRR = np.array([-1, 1, 1]) * jointVelocitiesRR

        ### Calculate velocity of each foot relative to corresponding hip using differential kinematics ###
        footVelFR = self._calRelFootVel(jointAnglesFR, jointVelocitiesFR)
        footVelFR = np.clip(footVelFR, -np.finfo(np.float32).max, np.finfo(np.float32).max)

        footVelRL = self._calRelFootVel(jointAnglesRL, jointVelocitiesRL)
        footVelRL = np.clip(footVelRL, -np.finfo(np.float32).max, np.finfo(np.float32).max)

        footVelFL = self._calRelFootVel(jointAnglesFL, jointVelocitiesFL)
        footVelFL = np.clip(footVelFL, -np.finfo(np.float32).max, np.finfo(np.float32).max)

        footVelRR = self._calRelFootVel(jointAnglesRR, jointVelocitiesRR)
        footVelRR = np.clip(footVelRR, -np.finfo(np.float32).max, np.finfo(np.float32).max)

        # Return full observations, including the current stage of the trot cycle and the and previous action
        obs = np.array([baseOrient[0], baseOrient[1], baseOrient[2], baseAngVel[0], baseAngVel[1], baseAngVel[2],
                        baseLinVel[0], baseLinVel[1], baseLinVel[2],
                        footPosFR[0], footPosFR[1], footPosFR[2], footPosRL[0], footPosRL[1], footPosRL[2],
                        footPosFL[0], footPosFL[1], footPosFL[2], footPosRR[0], footPosRR[1], footPosRR[2],
                        footVelFR[0], footVelFR[1], footVelFR[2], footVelRL[0], footVelRL[1], footVelRL[2],
                        footVelFL[0], footVelFL[1], footVelFL[2], footVelRR[0], footVelRR[1], footVelRR[2],
                        self.desiredVel, self.trotStage, self.prevFwdSwgLen, self.prevLatSwgLen, self.prevSwgHght,
                        self.prevTswg],
                       dtype=np.float32)

        return obs


    def _createPyBulletEnv(self, type="FLAT", height_mean=0.1, height_std_dev=0.01, xy_scale=0.05, stepLen=0.2,
                           stepWidth=2, stepHeight=0.02):
        """""
        This function creates a PyBullet simulation environment with the specified terrain type and parameters. For 
        rough terrain, no ramdom seed is set and so the random Gaussian distributed heightfield is different for each 
        episode. 
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
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        if type == "RANDOM_ROUGH":

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

            ### Create collision object and visual object that can be rendered ###
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

            boxHalfLength = 0.5 * stepLen  # Half length of step
            boxHalfWidth = 0.5 * stepWidth  # Half width of step
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

        return feetTrajectories


    def _calculateIK(self, targetFootPosition, kneeLagging, L1=0.06, L2=0.15, L3=0.21):
        """""
        Returns the joint angles required to place a single foot at a desired position relative to the corresponding hip 
        and a boolean error flag which is true if and only if there is an attempted workspace violation. It implements 
        the IK equations derived in the report and includes error checking for positions that violate the workspace 
        constraints.
        Args:
            targetFootPosition (list[float]): Desired Cartesian position of foot relative to corresponding hip.
            kneeLagging (bool): There are two possible knee configurations for each desired foot position, which are 
                                knee leading the foot (bending in >‾‾>) or knee lagging the foot (bending out <‾‾<).
                                If kneeLagging is true then the knee will lag the foot, else it will lead it.
            L1 (float): Length of link1 in meters.
            L2 (float): Length of link2 (the upper leg limb) in meters.
            L3 (float): Length of link3 (the lower leg limb) in meters.
        Returns:
            tuple:
                - targetJointPositions (list[float]): Joint angles of the leg which are the hip abduction/adduction 
                                                      angle (theta_1), the hip flexion/extension angle (theta_2) and the 
                                                      knee flexion/extension angle (theta_3)
                - error (bool): Flag indicating if the has been an attempted workspace violation.
        """""
        x = targetFootPosition[0]
        y = targetFootPosition[1]
        z = targetFootPosition[2]
        error = False

        z_dash = -np.sqrt(z ** 2 + y ** 2 - L1 ** 2)

        # Check for workspace violation
        if np.any((abs(y) / np.sqrt(z ** 2 + y ** 2) < -1) | (abs(y) / np.sqrt(z ** 2 + y ** 2) > 1)):
            print("Workspace Violation!")
            error = True
        alpha = np.arccos(np.clip(abs(y) / np.sqrt(z ** 2 + y ** 2), -1, 1)) # Clip to enforce workspace constraints

        # Check for workspace violation
        if np.any((L1 / np.sqrt(z ** 2 + y ** 2) < -1) | (L1 / np.sqrt(z ** 2 + y ** 2) > 1)):
            print("Workspace Violation!")
            error = True
        beta = np.arccos(np.clip(L1 / np.sqrt(z ** 2 + y ** 2), -1, 1)) # Clip to enforce workspace constraints

        # Check for workspace violation
        if np.any((abs(x) / np.sqrt(x ** 2 + z_dash ** 2) < -1) | (abs(x) / np.sqrt(x ** 2 + z_dash ** 2) > 1)):
            print("Workspace Violation!")
            error = True
        phi = np.arccos(np.clip(abs(x) / np.sqrt(x ** 2 + z_dash ** 2), -1, 1)) # Clip to enforce workspace constraints

        # Check for workspace violation
        if np.any(((x ** 2 + z_dash ** 2 + L2 ** 2 - L3 ** 2) / (2 * L2 * np.sqrt(x ** 2 + z_dash ** 2)) < -1) | (
                (x ** 2 + z_dash ** 2 + L2 ** 2 - L3 ** 2) / (2 * L2 * np.sqrt(x ** 2 + z_dash ** 2)) > 1)):
            print("Workspace Violation!")
            error = True
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

        return targetJointPositions, error


    def _calculateFK(self, jointPositions, L1=0.05955, L2=0.15, L3=0.21):
        """""
        Takes the joint positions of a single leg of PAWS and returns the position of the foot on the x-axis
        (forward/backward direction), y-axis (sideways direction) and z-axis (vertical direction).
        Args:
            jointPositions (list[float]): The list of the joint angles i.e. [theta1, theta2, theta3].
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
        Takes the joint positions and velocities of a single leg of PAWS and returns the velocities of the 
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
        Measures the current state of the simulated PAWS robot and appends it to a Pandas dataframe for
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
        basePos, baseOrient = p.getBasePositionAndOrientation(self.simPaws)  # Base position and orientation
        baseOrient = (180 / np.pi) * np.array(p.getEulerFromQuaternion(baseOrient))  # Euler orientation in degrees
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

        if self.firstStepLog:  # If this is the first log of the current action timestep, create a new data frame

            self.cog_df = pd.DataFrame({'Time (s)': self.logCounter / 240, 'COG x Position (m)': [basePos[0]],
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
            self.firstStepLog = False;
        else:
            new_cog_df = pd.DataFrame({'Time (s)': self.stepCounter / 240, 'COG x Position (m)': [basePos[0]],
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
