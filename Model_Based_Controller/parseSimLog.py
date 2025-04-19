import numpy as np                  # Used for efficient numerical computation
import matplotlib.pyplot as plt     # Used for visualisation
import pandas as pd                 # Used for reading logs
import paws_utils                   # Used for forward kinematics


################### This script is used to parse the CSV log of simulation data and plot the results ###################


plt.rcParams.update({
    'font.size': 12,              # Base font size
    'axes.titlesize': 12,         # Title of the plot
    'axes.labelsize': 12,         # Axis labels
    'xtick.labelsize': 11,        # X-axis tick labels
    'ytick.labelsize': 11,        # Y-axis tick labels
    'legend.fontsize': 12,        # Legend text
    'figure.titlesize': 12        # Figure-level title (if used)
})


######## Obtain Simulated Measurements ########
simFilename = r"C:\Users\Ronan\PawsQuadrupedRobot\log.csv"
simData = pd.read_csv(simFilename)

simFRjointAngles = np.array([simData["FR Hip Abd/Add Joint Angle"], simData["FR Hip Flex/Ext Joint Angle"],
                             simData["FR Knee Flex/Ext Joint Angle"]])
simFRfootPos = paws_utils.calculateFK(simFRjointAngles)
simRLjointAngles = np.array([simData["RL Hip Abd/Add Joint Angle"], -simData["RL Hip Flex/Ext Joint Angle"],
                             simData["RL Knee Flex/Ext Joint Angle"]])
simRLfootPos = paws_utils.calculateFK(simRLjointAngles)
simFLjointAngles = np.array([-simData["FL Hip Abd/Add Joint Angle"], -simData["FL Hip Flex/Ext Joint Angle"],
                              simData["FL Knee Flex/Ext Joint Angle"]])
simFLfootPos = paws_utils.calculateFK(simFLjointAngles)
simRRjointAngles = np.array([-simData["RR Hip Abd/Add Joint Angle"], simData["RR Hip Flex/Ext Joint Angle"],
                              simData["RR Knee Flex/Ext Joint Angle"]])
simRRfootPos = paws_utils.calculateFK(simRRjointAngles)


######## Plot Simulated Relative xz Trajectory ########
plt.figure(figsize=(8.42, 3.5))
plt.plot(simFRfootPos[0], simFRfootPos[2], color="tab:blue", label="FR Foot")    # Sim FR
plt.plot(simRLfootPos[0], simRLfootPos[2], color="tab:red", label="RL Foot")     # Sim FR
plt.plot(simFLfootPos[0], simFLfootPos[2], color="tab:orange", label="FL Foot")  # Sim FR
plt.plot(simRRfootPos[0], simRRfootPos[2], color="tab:green", label="RR Foot")   # Sim FR
plt.xlabel("x Position Relative to Hip (m)")
plt.ylabel("z Position Relative to Hip (m)")
plt.title("xz Trajectory of Each Foot Relative to the Corresponding Hip")
plt.axis('equal')
plt.grid()
plt.legend()
plt.tight_layout()
# plt.show()
plt.show(block=False)


######## Plot Simulated xy Trajectory of the COG ########
plt.figure(figsize=(8.42, 3.5))
plt.plot(simData["COG x Position (m)"], simData["COG y Position (m)"], color="tab:blue")
plt.xlabel("COG x Position (m)")
plt.ylabel("COG y Position (m)")
plt.title("xy Trajectory of the COG of PAWS")
plt.axis('equal')
plt.grid()
plt.tight_layout()
# plt.show()
plt.show(block=False)


######## Plot Simulated xz Trajectory of the COG ########
plt.figure(figsize=(8.42, 3.5))
plt.plot(simData["COG x Position (m)"], simData["COG z Position (m)"], color="tab:blue")
plt.xlabel("COG x Position (m)")
plt.ylabel("COG z Position (m)")
plt.title("xz Trajectory of the COG of PAWS")
# plt.axis('equal')
plt.grid()
plt.tight_layout()
# plt.show()
plt.show(block=False)


######## Plot Simulated x, y and z Position of the COG over Time ########
plt.figure(figsize=(8.42, 3.5))
plt.plot(simData["Time (s)"], simData["COG x Position (m)"], color="tab:blue", label="x Position")
plt.plot(simData["Time (s)"], simData["COG y Position (m)"], color="tab:red", label="y Position")
plt.plot(simData["Time (s)"], simData["COG z Position (m)"], color="tab:green", label="z Position")
plt.xlabel("Time (s)")
plt.ylabel("World Position (m)")
plt.title("x, y and z Position of the COG of PAWS Over Time")
plt.grid()
plt.legend()
plt.tight_layout()
# plt.show()
plt.show(block=False)


######## Plot Simulated roll, pitch and yaw of the COG over Time ########
plt.figure(figsize=(8.42, 3.5))
plt.plot(simData["Time (s)"], -simData["COG Roll (deg)"], color="tab:blue", label="Roll")
plt.plot(simData["Time (s)"], -simData["COG Pitch (deg)"], color="tab:red", label="Pitch")
plt.plot(simData["Time (s)"], simData["COG Yaw (deg)"], color="tab:green", label="Yaw")
plt.xlabel("Time (s)")
plt.ylabel("Angle (°)")
plt.title("Roll, Pitch and Yaw of the COG of PAWS Over Time")
plt.grid()
plt.legend()
plt.tight_layout()
# plt.show()
plt.show(block=False)


######## Plot xz Trajectory of Feet in World Coordinates ########
plt.figure(figsize=(8.42, 3.5))
plt.plot(simData["FR Foot x Position (m)"], simData["FR Foot z Position (m)"], color="tab:blue", label="FR Foot")
plt.plot(simData["RL Foot x Position (m)"], simData["RL Foot z Position (m)"], color="tab:red", label="RL Foot")
plt.plot(simData["FL Foot x Position (m)"], simData["FL Foot z Position (m)"], color="tab:orange", label="FL Foot")
plt.plot(simData["RR Foot x Position (m)"], simData["RR Foot z Position (m)"], color="tab:green", label="RR Foot")
plt.xlabel("x World Position (m)")
plt.ylabel("z World Position (m)")
plt.title("xz Trajectory of Feet in World Coordinates")
plt.grid()
plt.legend()
plt.axis('equal')
plt.tight_layout()
# plt.show()
plt.show(block=False)


######## Plot z Position of Feet in World Coordinates Over Time ########
plt.figure(figsize=(8.42, 3.5))
plt.plot(simData["Time (s)"], simData["FR Foot z Position (m)"], color="tab:blue", label="FR Foot")
plt.plot(simData["Time (s)"], simData["RL Foot z Position (m)"], color="tab:red", label="RL Foot")
plt.plot(simData["Time (s)"], simData["FL Foot z Position (m)"], color="tab:orange", label="FL Foot")
plt.plot(simData["Time (s)"], simData["RR Foot z Position (m)"], color="tab:green", label="RR Foot")
plt.xlabel("Time (s)")
plt.ylabel("z World Position (m)")
plt.title("z Position of Feet in World Coordinates over Time")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
