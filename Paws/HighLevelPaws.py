import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt
import can
import can_utils
import os
import Paws

paws = Paws.PawsRobot()

paws.createTrotTrajectory(stepLength=0.1, stepHeight=0.05, T=2, sampleRate=1000, x_0=0, h_0=-0.25)
paws.plotTrajectory(axis="xz")
paws.simulateTrotGate()

# paws.createWalkTrajectory(stepLength=0.1, stepHeight=0.08, T=1, sampleRate=500, x_0=-0.05, h_0=-0.25)
# paws.plotTrajectory(axis="xz")
# paws.simulateWalkGate()
