# PawsQuadrupedRobot

Model_Based_Controller:

	HighLevelPaws.py: Script used to call methods of PawsModelBasedControl.py and run simulations of the model-based controller.
  
	PawsModelBasedControl.py: This is a class containing the all the functions to simulate the model-based locomotion controller in PyBullet and run it on 
                              the physical PAWS robot.
                            
	parseSimLog.py: Script used to parse the CSV log of simulation data and plot the results.
  
	paws_utils.py: Helper functions for forward, inverse and differential kinematics. Used by parseSimLog.py.


RL_Controllers:

  	pawsSemiEndToEndEnv.py: This is a class which creates a custom reinforcement learning environment for the PAWS robot, which inherits from the base 
                          	class gym.Env provided by the Gymnassium environment. This class adheres to the standard interface of a Gymnassium 
                          	API, ensuring compatibility with popular RL libraries and frameworks including Stable Baselines 3. This environment 
                          	is for the semi-end-to-end RL architecture, which is a policy over task-space actions. IK and a PD controller are 
                          	then used to convert the task-space actions into motor position commands.
                          
  	pawsEnvPMTG.py: This is a class which creates a custom reinforcement learning environment for the PAWS robot, which inherits from the base 
                  	class gym.Env provided by the Gymnassium environment. This class adheres to the standard interface of a Gymnassium 
                  	API, ensuring compatibility with popular RL libraries and frameworks including Stable Baselines 3. This environment 
                  	is for the policy modulating trajectory generater (PMTG) architecture, which modulates the forward step length, 
                  	lateral step length, step height and step period of the trot gait generator deveolped in the model-based controller.
                  
  	checkEnv.py: This script is used check that the created custom environment is valid and error free.
  
  	learnSemiEndToEnd.py: This script is used to train the semi-end-to-end RL controller, save the policy and value function networks and log to TensorBoard.
  
  	learnPMTG.py: This script is used to train the PMTG controller, save the policy and value function networks and log to TensorBoard.
  
  	runModel.py: This script is used to deploy the RL controllers in simulation. 
  
