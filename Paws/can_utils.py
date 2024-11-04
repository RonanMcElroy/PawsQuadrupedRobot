import can


def radToControlValue(rad):
    # Conversion factor: 1 radian = 5729.57795 control value (0.01-degree)
    pos_in_deg = (rad * (180.0 / np.pi))
    control_value = int(pos_in_deg * 100)  # 0.01degree/LSB
    return control_value


def genCanMsg(motorNum, position):
    motor_num = motorNum
    bus = can.Bus(interface='socketcan', channel='can0', bitrate=1000000)
    control_value_position = radToControlValue(position)
    msg = can.Message(arbitration_id=0x140 + motor_num,
                      data=[0xa4, 0x00, 0x50, 0x00, control_value_position & 0xFF,
                            (control_value_position >> 8) & 0xFF,
                            (control_value_position >> 16) & 0xFF,
                            (control_value_position >> 24) & 0xFF],
                      is_extended_id=False)
    # print("frame", msg)
    print("pos", control_value_position / 100)
    time.sleep(0.002)
    bus.send(msg)