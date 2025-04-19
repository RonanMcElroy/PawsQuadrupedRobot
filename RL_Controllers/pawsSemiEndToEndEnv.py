import pybullet as p                   # Physics engine
import pybullet_data                   # Contains plane.urdf
import time                            # Used for delays and timing
import numpy as np                     # Efficient numerical computation
import matplotlib.pyplot as plt        # Visualisation
import random                          # Random number generation
import pandas as pd                    # Used for logging simulation data
import gymnasium as gym                # Used to define and interact with RL environment
from gymnasium import spaces           # Provides functions for creating action and observation spaces


class PawsEnv(gym.Env):
    """""
    This class creates a custom reinforcement learning environment for the PAWS robot, which inherits from the base 
    class gym.Env provided by the Gymnassium environment. This class adheres to the standard interface of a Gymnassium 
    API, ensuring compatibility with popular RL libraries and frameworks including Stable Baselines 3. This environment 
    is for the semi-end-to-end RL architecture, which is a policy over task-space actions. IK and a PD controller are 
    then used to convert the task-space actions into motor position commands.
    
    Methods:
        __init__(renderMode, max_x_dis, truncationSteps, max_tilt, maxFwdVel, minFwdVel, terrain, hieght_mean, 
                 height_std, xy_scale, stairStepLen, stairStepWidth, stairStepHeight, Kp, Kd):
            Defines the continuous observation and action spaces. Connects to PyBullet in either direct mode (no 
            rendering) or human mode (render simulation in GUI) and creates the specified terrain. Initialises 
            attributes for the termination and truncation conditions as well as the range of forward velocity commands
            and the gains of the PD controller.
            
        reset(seed, options, bodyStartPos, footStartPos):
            Starts a new episode. Recreates simulation environment (if ROUGH then it will be a different random Gaussian 
            distributed heightfield each episode) and reloads PAWS URDF. Sets starting position. Sets forward velocity 
            command for episode according to a uniform distribution. Returns the observations and info dict.
        
        step(action):
            Denormalises action selected by policy to get the desired position of each foot relative to the 
            corresponding hip. Calculates the desired joint angles using IK. Sends commands to all motors. Steps the 
            simulation. Gets the observations and calculates the reward. Logs simulation data to a CSV file if render 
            mode is human. Checks for termination and truncation conditions. Returns observations, reward, terminated 
            flag, truncated flag and info dict.
        
        close():
            Disconnects from PyBullet.
        
        _getObs():
            Uses PyBullet functions to collect and return the state observations from the environment.
        
        _createPyBulletEnv(type, height_mean, height_std_dev, xy_scale, stepLen, stepWidth, stepHeight):
            Creates a simulation environment with the specified terrain type and parameters.
        
        _setFeetStartPos(footStartPos):
            Moves feet to desired start position.
        
        _calculateIK(targetFootPosition, kneeLagging, L1=0.06, L2=0.15, L3=0.21)
            Returns the joint angles required to place a single foot at a desired position relative to the 
            corresponding hip.
        
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

    def __init__(self, renderMode=None, max_x_dis= 5, truncationSteps=1200, max_tilt=np.pi/3,
                 maxFwdVel=1.0, minFwdVel=0.5, terrain="FLAT", hieght_mean=0.1, height_std=0.01, xy_scale=0.05,
                 stairStepLen=0.2, stairStepWidth=2, stairStepHeight=0.02, Kp=0.15, Kd=0.55):
        """""
        This method defines the continuous observation and action spaces, connects to PyBullet and creates the specified 
        terrain. Initialises attributes for the termination and truncation conditions as well as the range of forward 
        velocity commands and the gains of the PD controller.
        Args:
            renderMode (str): Asserted in [None, "human"]. Decides whether to connect to PyBullet in direct mode with no 
                              rendering or in human mode to render the simulation in the GUI. Direct mode is used for 
                              training. Human mode is uded for testing.
            max_x_dis (float): Maximum x distance before episode is terminated (success).
            truncationSteps (int): Maximum number of timestepd before episode is truncated. Each timestep is 0.05 s.
            max_tilt (float): Maximum tilt in roll or pitch angle before episode is terminated (failure).
            maxFwdVel(float): Minimum forward velocity command.
            minFwdVel(float): Maximum forward velocity command.
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
        self.max_x_dis = max_x_dis
        self.truncationSteps = truncationSteps
        self.max_tilt = max_tilt
        self.maxVel = maxFwdVel
        self.minVel = minFwdVel
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


        ########################################### Define Observation Space ###########################################
        # The State Observation is:
        # [COGrollAngle, COGpitchAngle, COGyawAngle,
        #  COGrollRate, COGpitchRate, COGyawRate,
        #  GOGxVelocity, COGyVelocity, COGzVelocity,
        #  xRelPosFR, yRelPosFR, zRelPosFR, xRelPosRL, yRelPosRL, zRelPosRL,
        #  xRelPosFL, yRelPosFL, zRelPosFL, xRelPosRR, yRelPosRR, zRelPosRR,
        #  xVelFR, yVelFR, zVelFR, xVelRL, yVelRL, zVelRL,
        #  xVelFL, yVelFL, zVelFL, xVelRR, yVelRR, zVelRR,
        #  desiredVel]

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
                                     self.maxVel])

        stateLowerLimits = -1*np.array([max_tilt, max_tilt, np.pi,
                                        np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                                        np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                                        self.xMaxRelPos, self.yMaxRelPos, self.zMinRelPos, self.xMaxRelPos,
                                        self.yMaxRelPos, self.zMinRelPos, self.xMaxRelPos, self.yMaxRelPos,
                                        self.zMinRelPos, self.xMaxRelPos, self.yMaxRelPos, self.zMinRelPos,
                                        np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                                        np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                                        np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                                        np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                                        -self.minVel])

        # The Gymnassium function Box() is used for continuous spaces
        self.observation_space = spaces.Box(stateLowerLimits, stateUpperLimits, dtype=np.float32)


        ############################################# Define Action Space ##############################################
        # The actions are:
        # [xRelPosFR, yRelPosFR, zRelPosFR, xRelPosRL, yRelPosRL, zRelPosRL,
        #  xRelPosFL, yRelPosFL, zRelPosFL, xRelPosRR, yRelPosRR, zRelPosRR]
        #  These actions will be normalised to the range [-1, 1] and denormalised before they are applied to the robot.

        actionLimits = np.ones(12)
        # Action space normalised in the range [-1, 1] for better convergence. When actions are not normalized, the
        # learning algorithm may struggle to converge as actions with larger magnitudes may leading to unstable updates.
        # Normalization also helps the agent explore the action space more evenly.
        self.action_space = spaces.Box(-actionLimits, actionLimits, dtype=np.float32)


        ############################################# Connect to PyBullet ##############################################
        if self._renderMode == 'human':
            fig = plt.figure(1)                    # Calling this first makes the GUI higher resolution
            self.physicsClient = p.connect(p.GUI)  # Connect to PyBullet in render mode
        else:
            self.physicsClient = p.connect(p.DIRECT)  # Connect to PyBullet in direct mode with no GUI (for training)

        assert self.terrain in ["FLAT", "RANDOM_ROUGH", "STAIRS"]

        self._createPyBulletEnv(self.terrain, self.hieght_mean, self.height_std, self.xy_scale, self.stairStepLen,
                                self.stairStepWidth, self.stairStepHeight)
        self.planeId = p.loadURDF("plane.urdf")
        self.simPaws = p.loadURDF(r'PawsURDF/urdf/Paws.urdf', [0, 0, 0.4], useFixedBase=0)

        return


    def reset(self, seed=None, options=None, bodyStartPos=[0, 0, 0.4], footStartPos=[0, 0.05955, -0.26]):
        """""
        Starts a new episode. Recreates simulation environment (if ROUGH then it will be a different random Gaussian 
        distributed heightfield each episode) and reloads PAWS URDF. Sets starting position. Sets forward velocity 
        command for episode according to a uniform distribution. Returns the observations and info dict.
        Args:
            seed (int): Seed for random number generators to ensure reproducibility. Required by Gymnasium API 
                        interface. Defaults to None.
            options (dict): Required by Gymnasium API interface. Defaults to None.
            bodyStartPos (list[float]): Start position of the body of PAWS in world coordinates.
            footStartPos (list[float]): Start position of the feet of PAWS relative to the corresponding hip.
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


        self._setFeetStartPos(footStartPos)  # Bring the feet to their starting position
        self.desiredVel = np.random.uniform(low=self.minVel, high=self.maxVel)  # Set the desired forward velocity for
                                                                                # the episode
        # print(self.desiredVel)

        if self._renderMode == 'human':
            self.firstStepLog = True  # The first log creates a dataframe, subsequent logs append data
            self.firstEpisodeLog = True  # The first log creates a CSV file, subsequent logs append data
            self.logCounter = 0
            self._collectSimData()
            self.cog_df.to_csv(self.filename, index=False)
            self.cog_df.drop(self.cog_df.index, inplace=True)

        self.stepCounter = 0
        observation = self._getObs()
        info = {}

        return observation, info


    def step(self, action):
        """""
        Denormalises action selected by policy to get the desired position of each foot relative to the corresponding 
        hip. Calculates the desired joint angles using IK. Sends commands to all motors. Steps the simulation. Gets the 
        observations and calculates the reward. Logs simulation data to a CSV file if render mode is human. Checks for 
        termination and truncation conditions. Returns observations, reward, terminated flag, truncated flag and 
        info dict.
        Args:
            action (np.ndarray): Action selected by the policy for the current timestep, which is an array of the 
                                 position of each foot relative to the corresponding hip, where each element is 
                                 normalised to the range [-1, 1]. 
        Returns:
            tuple:
                - observation (np.ndarray): State observations according to MDP.
                - reward (float): Reward for the action just taken.
                - terminated (bool): True if termination conditions are met, else False.
                - truncated (bool): True if truncation conditions are met, else False.
                - info (dict): Required by Gymnassium API interface. Just an empty dict.
        """""

        ################################ Separate and Denormalise Actions for Each Foot ################################

        actionFR = action[0:3] + np.array([0, 0, (self.zMaxRelPos-self.zMinRelPos)/(self.zMaxRelPos+self.zMinRelPos)])
        actionFR = actionFR * np.array([self.xMaxRelPos, self.yMaxRelPos, (self.zMaxRelPos + self.zMinRelPos)/2])

        actionRL = action[3:6] + np.array([0, 0, (self.zMaxRelPos-self.zMinRelPos)/(self.zMaxRelPos+self.zMinRelPos)])
        actionRL = actionRL * np.array([self.xMaxRelPos, self.yMaxRelPos, (self.zMaxRelPos + self.zMinRelPos) / 2])

        actionFL = action[6:9] + np.array([0, 0, (self.zMaxRelPos-self.zMinRelPos)/(self.zMaxRelPos+self.zMinRelPos)])
        actionFL = actionFL * np.array([self.xMaxRelPos, self.yMaxRelPos, (self.zMaxRelPos + self.zMinRelPos) / 2])

        actionRR = action[9:] + np.array([0, 0, (self.zMaxRelPos-self.zMinRelPos)/(self.zMaxRelPos+self.zMinRelPos)])
        actionRR = actionRR * np.array([self.xMaxRelPos, self.yMaxRelPos, (self.zMaxRelPos + self.zMinRelPos) / 2])

        ###################### Calculate Joint Angles Corresponding to the Desired Foot Positions ######################

        FR_targetJointPositions = self._calculateIK(actionFR, kneeLagging=True)
        RL_targetJointPositions = self._calculateIK(actionRL, kneeLagging=True)
        FL_targetJointPositions = self._calculateIK(actionFL, kneeLagging=True)
        RR_targetJointPositions = self._calculateIK(actionRR, kneeLagging=True)

        #################################### Send Joint Position Commands to Motors ####################################

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

        p.setJointMotorControl2(self.simPaws, self.FL_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=-FL_targetJointPositions[0],
                                positionGain=self.Kp, velocityGain=self.Kd)
        p.setJointMotorControl2(self.simPaws, self.FL_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=-FL_targetJointPositions[1],
                                positionGain=self.Kp, velocityGain=self.Kd)
        p.setJointMotorControl2(self.simPaws, self.FL_KNEE_JOINT, p.POSITION_CONTROL,
                                targetPosition=FL_targetJointPositions[2],
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

        ########################################## Step Simulation and Delay ###########################################

        if self._renderMode == 'human':

            # Actions are taken every 0.05 s, but each simulation step is 1/240 s. Hence 12 simulation steps are taken
            # for every action.
            for i in range(12):
                startTime = time.perf_counter()
                p.stepSimulation()
                self.logCounter += 1
                self._collectSimData()

                elapsedTime = time.perf_counter() - startTime

                while elapsedTime < 1.0 / 240.0:                   # Delay until next sample time
                    elapsedTime = time.perf_counter() - startTime
        else:
            for i in range(12):     # No delays for faster training
                p.stepSimulation()

        if self._renderMode == 'human':
            self.cog_df.to_csv(self.filename, mode='a', index=False, header=False)
            self.cog_df.drop(self.cog_df.index, inplace=True)

        ############################################### Get Observations ###############################################
        # The observations are:
        # [COGrollAngle, COGpitchAngle, COGyawAngle, COGrollRate, COGpitchRate, COGyawRate,
        #  GOGxVelocity, COGyVelocity, COGzVelocity,
        #  xRelPosFR, yRelPosFR, zRelPosFR, xRelPosRL, yRelPosRL, zRelPosRL,
        #  xRelPosFL, yRelPosFL, zRelPosFL, xRelPosRR, yRelPosRR, zRelPosRR,
        #  xVelFR, yVelFR, zVelFR, xVelRL, yVelRL, zVelRL,
        #  xVelFL, yVelFL, zVelFL, xVelRR, yVelRR, zVelRR,
        #  desiredVel]
        observation = self._getObs()

        # More readable names
        rollAngle = observation[0]
        pitchAngle = observation[1]
        yawAngle = observation[2]
        xVelCOG = observation[6]
        zPosFR = observation[11]
        zPosRL = observation[14]
        zPosFL = observation[17]
        zPosRR = observation[20]
        xVelFR = observation[21]
        yVelFR = observation[22]
        zVelFR = observation[23]
        xVelRL = observation[24]
        yVelRL = observation[25]
        zVelRL = observation[26]
        xVelFL = observation[27]
        yVelFL = observation[28]
        zVelFL = observation[29]
        xVelRR = observation[30]
        yVelRR = observation[31]
        zVelRR = observation[32]

        ############################################### Calculate Reward ###############################################
        # See report for all reward function equations

        #### Forward velocity tracking reward (r_vel) ###
        r_vel = self.maxVel * np.exp(-(xVelCOG - self.desiredVel)**2 / (2*self.maxVel**2))

        #### Foot contact reward (r_fc) to encourage trot ###
        w_fc = 0.4 # Weight of foot contact reward

        # Determine which feet are in contact with the ground
        footContactFR = not np.array(p.getContactPoints(bodyA=self.planeId, bodyB=self.simPaws,
                                     linkIndexB=self.FR_KNEE_JOINT), dtype=object).all()
        footContactRL = not np.array(p.getContactPoints(bodyA=self.planeId, bodyB=self.simPaws,
                                     linkIndexB=self.RL_KNEE_JOINT), dtype=object).all()
        footContactFL = not np.array(p.getContactPoints(bodyA=self.planeId, bodyB=self.simPaws,
                                     linkIndexB=self.FL_KNEE_JOINT), dtype=object).all()
        footContactRR = not np.array(p.getContactPoints(bodyA=self.planeId, bodyB=self.simPaws,
                                     linkIndexB=self.RR_KNEE_JOINT), dtype=object).all()

        # Reward for one pair of diagonal feet on ground while the other pair of diagonal feet are in the air
        if ((footContactFR and footContactRL) and (not footContactFL) and (not footContactRR)) or \
           ((footContactFL and footContactRR) and (not footContactFR) and (not footContactRL)):
            r_fc = w_fc
        else:
            r_fc = 0

        #### Foot height reward (r_fh) to encourage sufficient foot clearance ###
        w_fh = 2.0  # Weight of foot height reward

        r_fh = (not footContactFR)*(0.28 - abs(zPosFR))
        r_fh += (not footContactRL)*(0.28 - abs(zPosRL))
        r_fh += (not footContactFL)*(0.28 - abs(zPosFL))
        r_fh += (not footContactRR)*(0.28 - abs(zPosRR))
        r_fh = w_fh*r_fh

        #### Base stability penalty (p_bs) ###
        w_bs = 8.0
        p_bs = -w_bs*(rollAngle**2 + pitchAngle**2 + yawAngle**2)

        #### Symmetry penalty (p_sym) ###
        w_sym = 2.0
        p_sym = -w_sym*( (xVelFR-xVelRL)**2 + (yVelFR-yVelRL)**2 + (zVelFR-zVelRL)**2 + \
                         (xVelFL-xVelRR)**2 + (yVelFL-yVelRR)**2 + (zVelFL-zVelRR)**2 )

        ### Full reward ###
        reward = r_vel + r_fc + r_fh + p_bs + p_sym

        ################################################### Get Info ###################################################

        info = {}

        ############################################ Truncation Conditions #############################################

        self.stepCounter += 1
        if self.stepCounter >= self.truncationSteps:
            truncated = True
            # print("\n\n\n Truncated!!! \n\n\n")
        else:
            truncated = False

        ############################################ Termination Conditions ############################################
        basePos, _ = p.getBasePositionAndOrientation(self.simPaws)
        xPosCOG = basePos[0]
        zPosCOG = basePos[2]

        if abs(rollAngle) >= self.max_tilt or abs(pitchAngle) >= self.max_tilt or zPosCOG <= 0.1 or \
           np.isnan(FR_targetJointPositions).any() or np.isnan(RL_targetJointPositions).any() or  \
           np.isnan(FL_targetJointPositions).any() or np.isnan(RR_targetJointPositions).any() or \
           abs(xPosCOG) >= self.max_x_dis:

            terminated = True
            # print("\n\n\n Terminated!!! \n\n\n")

        else:
            terminated = False

        return observation, reward, terminated, truncated, info


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
                                      [COGrollAngle, COGpitchAngle, COGyawAngle, COGrollRate, COGpitchRate, COGyawRate,
                                       GOGxVelocity, COGyVelocity, COGzVelocity, 
                                       xRelPosFR, yRelPosFR, zRelPosFR, xRelPosRL, yRelPosRL, zRelPosRL,
                                       xRelPosFL, yRelPosFL, zRelPosFL, xRelPosRR, yRelPosRR, zRelPosRR,
                                       xVelFR, yVelFR, zVelFR, xVelRL, yVelRL, zVelRL, 
                                       xVelFL, yVelFL, zVelFL, xVelRR, yVelRR, zVelRR, 
                                       desiredVel]
        """""
        _, baseOrient = p.getBasePositionAndOrientation(self.simPaws)
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

        # Return full observations, including desired forward velocity in x-direction
        obs = np.array([baseOrient[0], baseOrient[1], baseOrient[2], baseAngVel[0], baseAngVel[1], baseAngVel[2],
                        baseLinVel[0], baseLinVel[1], baseLinVel[2],
                        footPosFR[0], footPosFR[1], footPosFR[2], footPosRL[0], footPosRL[1], footPosRL[2],
                        footPosFL[0], footPosFL[1], footPosFL[2], footPosRR[0], footPosRR[1], footPosRR[2],
                        footVelFR[0], footVelFR[1], footVelFR[2], footVelRL[0], footVelRL[1], footVelRL[2],
                        footVelFL[0], footVelFL[1], footVelFL[2], footVelRR[0], footVelRR[1], footVelRR[2],
                        self.desiredVel], dtype=np.float32)

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

            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)   # Configure debug visualiser

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
            pass   # Flat is just the default PyBullet plane


    def _setFeetStartPos(self, footStartPos):
        """""
        Moves feet to the desired start position.
        Args:
            footStartPos (list[float]): Desired Cartesian position of each foot relative to the corresponding hip.
        Returns:
            None
        """""
        # Convert task-space foot position into joint-space using IK
        startJointPos = self._calculateIK(footStartPos, kneeLagging=True)

        ### Send commands to all motors ###
        p.setJointMotorControl2(self.simPaws, self.FR_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=startJointPos[0])
        p.setJointMotorControl2(self.simPaws, self.FR_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=startJointPos[1])
        p.setJointMotorControl2(self.simPaws, self.FR_KNEE_JOINT, p.POSITION_CONTROL, targetPosition=startJointPos[2])

        p.setJointMotorControl2(self.simPaws, self.RL_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=startJointPos[0])
        p.setJointMotorControl2(self.simPaws, self.RL_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=-startJointPos[1])
        p.setJointMotorControl2(self.simPaws, self.RL_KNEE_JOINT, p.POSITION_CONTROL, targetPosition=startJointPos[2])

        p.setJointMotorControl2(self.simPaws, self.FL_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=-startJointPos[0])
        p.setJointMotorControl2(self.simPaws, self.FL_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=-startJointPos[1])
        p.setJointMotorControl2(self.simPaws, self.FL_KNEE_JOINT, p.POSITION_CONTROL, targetPosition=startJointPos[2])

        p.setJointMotorControl2(self.simPaws, self.RR_HIP_ABDUCTOR_ADDUCTOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=-startJointPos[0])
        p.setJointMotorControl2(self.simPaws, self.RR_HIP_FLEXOR_EXTENDOR_JOINT, p.POSITION_CONTROL,
                                targetPosition=startJointPos[1])
        p.setJointMotorControl2(self.simPaws, self.RR_KNEE_JOINT, p.POSITION_CONTROL, targetPosition=startJointPos[2])

        # Step the simulation 100 times (100/240 = 0.417 s) to move motors to initial positions
        for j in range(0, 100):
            p.stepSimulation()

        return


    def _calculateIK(self, targetFootPosition, kneeLagging, L1=0.06, L2=0.15, L3=0.21):
        """""
        Returns the joint angles required to place a single foot at a desired position relative to the 
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


    def _calculateFK(self, jointPositions, L1=0.05955, L2=0.15, L3=0.21):
        """""
        Takes the joint positions of a single leg of PAWS and returns the position of the foot on the x-axis
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

        if self.firstStepLog: # If this is the first log of the current action timestep, create a new data frame

            self.cog_df = pd.DataFrame({'Time (s)': self.logCounter/240, 'COG x Position (m)':[basePos[0]],
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
            new_cog_df = pd.DataFrame({'Time (s)': self.stepCounter/240, 'COG x Position (m)': [basePos[0]],
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
