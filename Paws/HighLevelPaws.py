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

paws.createTrotTrajectory()
paws.plotTrajectory(axis="xz")
paws.simulateTrotGate()

# paws.createWalkTrajectory()
# paws.plotTrajectory(axis="xz")
# paws.simulateWalkGate()
