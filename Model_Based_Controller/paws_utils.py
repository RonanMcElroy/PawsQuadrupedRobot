import numpy as np

############################################# Helper functions for PAWS ################################################

def calculateIK(targetFootPosition, kneeLagging, L1=0.06, L2=0.15, L3=0.21):
    """""
    This function returns the joint angles required to place a single foot at a desired position relative to the 
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
    alpha = np.arccos(np.clip(abs(y) / np.sqrt(z ** 2 + y ** 2), -1, 1))  # Clip to enforce workspace constraints

    # Check for workspace violation
    if np.any((L1 / np.sqrt(z ** 2 + y ** 2) < -1) | (L1 / np.sqrt(z ** 2 + y ** 2) > 1)):
        print("Workspace Violation!")
    beta = np.arccos(np.clip(L1 / np.sqrt(z ** 2 + y ** 2), -1, 1))  # Clip to enforce workspace constraints

    # Check for workspace violation
    if np.any((abs(x) / np.sqrt(x ** 2 + z_dash ** 2) < -1) | (abs(x) / np.sqrt(x ** 2 + z_dash ** 2) > 1)):
        print("Workspace Violation!")
    phi = np.arccos(np.clip(abs(x) / np.sqrt(x ** 2 + z_dash ** 2), -1, 1))  # Clip to enforce workspace constraints

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


def calculateFK(jointPositions, L1=0.05955, L2=0.15, L3=0.21):
    """""
    This function takes the joint positions of a single leg of PAWS and returns the position of the foot on the x-axis
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
    yPos = L1 * np.cos(theta1) + L2 * np.sin(theta1) * np.cos(theta2) + L3 * np.sin(theta1) * np.cos(theta2 + theta3)
    zPos = L1 * np.sin(theta1) - L2 * np.cos(theta1) * np.cos(theta2) - L3 * np.cos(theta1) * np.cos(theta2 + theta3)

    footPos = np.array([xPos, yPos, zPos])

    return footPos


def calFootVel(jointPositions, jointVelocities, L1=0.05955, L2=0.15, L3=0.21):
    """""
    This function takes the joint positions and velocities of a single leg of PAWS and returns the velocities of the 
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
