import PawsModelBasedControl    # Import Paws Class


# Create instance of PAWS robot
paws = PawsModelBasedControl.Paws()

# In the following code, the first uncommented function call will run in an infinite loop until the program is ended
####################################################### Trotting #######################################################

### Simulate Trotting on Flat ground ###
paws.simulateTrotGate(forwardSwingLength = 0.09, latSwingLength = 0.0, swingHeight = 0.05,
                       latStanceLength = 0.0, T_swing = 0.2, h_0 = -0.28, sampleRate=212, backSwingCoef=0.2,
                       hipOffset=0.05955, terrain="FLAT", fixBase=False, logData=True)

### Simulate Trotting on Rough Ground ###
paws.simulateTrotGate(forwardSwingLength = 0.08, latSwingLength = 0.0, swingHeight = 0.06,
                       latStanceLength = 0.0, T_swing = 0.3, h_0 = -0.28, sampleRate=240, backSwingCoef=0.2,
                       hipOffset=0.05955, terrain="RANDOM_ROUGH", hieght_mean=0.1, height_std=0.011, xy_scale=0.045,
                       fixBase=False, logData=False)

### Simulate Trotting on Stairs ###
paws.simulateTrotGate(forwardSwingLength = 0.08, latSwingLength = 0.0, swingHeight = 0.05,
                       latStanceLength = 0.0, T_swing = 0.2, h_0 = -0.28, sampleRate=240, backSwingCoef=0.2,
                       hipOffset=0.05955, terrain="STAIRS", stairStepLen=0.2, stairStepWidth=2, stairStepHeight=0.02,
                       fixBase=False, logData=True)

### Simulate Trotting Suspended in the Air ###
paws.simulateTrotGate(forwardSwingLength = 0.08, latSwingLength = 0.0, swingHeight = 0.06,
                       latStanceLength = 0.0, T_swing = 0.9, h_0 = -0.32, sampleRate=240, backSwingCoef=0.2,
                       hipOffset=0.05955, terrain="FLAT", fixBase=True, logData=False)


####################################################### Walking ########################################################

### Simulate Walking on Flat ground with default PD values - use this ###
paws.simulateWalkGate(forwardSwingLength = 0.1, latSwingLength = 0, swingHeight = 0.08, leanLength = 0.02,
                       h_0 = -0.28, T_swing = 0.5, T_lean=0.3, sampleRate=240, backSwingCoef=0.2, hipOffset=0.05955,
                       terrain="FLAT", fixBase=False, logData=False)

### Simulate Walking on Rough Ground with default PD values - use this ###
paws.simulateWalkGate(forwardSwingLength=0.1, latSwingLength=0, swingHeight=0.07, leanLength=0.025,
                       h_0=-0.28, T_swing=0.4, T_lean=0.15, sampleRate=240, backSwingCoef=0.2, hipOffset=0.05955,
                       terrain="RANDOM_ROUGH", hieght_mean=0.1, height_std=0.015, xy_scale=0.045, fixBase=False,
                       logData=False)

### Simulate Walking on Stairs with default PD values - use this  ###
paws.simulateWalkGate(forwardSwingLength = 0.08, latSwingLength = 0, swingHeight = 0.06, leanLength = 0.04,
                       h_0 = -0.28, T_swing = 0.4, T_lean=0.3, sampleRate=240, backSwingCoef=0.2, hipOffset=0.05955,
                       terrain="STAIRS", stairStepLen=0.2, stairStepWidth=2, stairStepHeight=0.03, fixBase=False,
                      logData=False)

### Simulate Walking Suspended in the Air ###
paws.simulateWalkGate(forwardSwingLength = 0.1, latSwingLength = 0, swingHeight = 0.06, leanLength = 0.0,
                       h_0 = -0.28, T_swing = 0.72, T_lean=0.02, sampleRate=60, backSwingCoef=0.2, hipOffset=0.05955,
                       terrain="FLAT", fixBase=True, logData=False)
