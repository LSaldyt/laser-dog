import json, datetime
import bluetooth
import struct
from time import sleep

from bt_serial import initialize

# Emperical notes on hardware robot

# Front left
# Servo 0: Non-function
# Servo 1: Rest @ 360, bigger = closer to rest
# Servo 2: Rest @ 270, bigger = closer to rest

# Front right
# Servo 3: Rest @ 320, bigger = closer to rest
# Servo 4: Rest @ 380, bigger = closer to rest
# Servo 5: Rest @ 300, bigger = further from rest

# Back left
# Servo 6: Rest @ 70, bigger = further from rest
# Servo 7: Rest @ 310, bigger = closer to rest
# Servo 8: Rest @ 220, bigger = closer to rest

# Back right
# Servo 9: Rest @ 460, bigger = closer to rest
# Servo 10: Rest @ 330, bigger = further from rest
# Servo 11: Rest @ 200, ??

# Notes from angle calibration

# Leg calibration (front left)
# 57.1mm (open)
# 19.6mm (closed)
# 81.2mm (leg length)
# Leg: 40 pulse = 0.48 radians
# 1 pulse = 0.012 radians

# Knee calibration (front left)
# Rest: 280
# 90 open: 110
# Knee: 170 pulse = 1.571 radians
# 1 pulse = 0.00924 radians

# Shoulder calibration (back left)
# Rest: 310
# 45 down: 240
# Shoulder: 70 pulse = 0.785 radians
# 1 pulse = 0.0112 radians

# Notes from standup() task

# Order: Shoulder, Knee, Leg
# signal = [
#     fl_angles[0], fl_angles[1], fl_angles[2],
#     fr_angles[0], fr_angles[1], fr_angles[2],
#     rl_angles[0], rl_angles[1], rl_angles[2],
#     rr_angles[0], rr_angles[1], rr_angles[2] ]

# Motors value t0:
# [ 0.         -1.35539923  1.99077815  0.         -1.35539923  1.99077815
#   0.         -1.35539923  1.99077815  0.         -1.35539923  1.99077815]

# Motors value tf:
# [ 0.         -0.88643435  1.30197369  0.         -0.88643435  1.30197369
#   0.         -0.88643435  1.30197369  0.         -0.88643435  1.30197369]

# Emperical rest positions in servo pulse width
# Order: Leg, Shoulder, Knee
#                0    1    2  * 3    4    5  * 6   7    8  * 9    10   11
initial_pulse = [90, 340, 280, 320, 380, 300, 155, 310, 240, 460, 310, 160]
initial_rad   = [1.99, 0, -1.355] * 4
# cal_mask      = [0.012, 0.112, 0.009] * 4
# cal_mask      = [0.015] * 12
# cal_mask      = [0.004, 0.006, 0.006] * 2 + [0.003, 0.006, 0.006] * 2 # Universal PWM:Radian Ratio
# cal_mask      = [0.004] * 12
# cal_mask      = [0.002, 0.006, 0.006] * 4
cal_mask = [0.005, # Front left leg
            0.006, # Front left shoulder
            0.006, # Front left knee
            0.004, # Front right leg
            0.006, # Front right shoulder
            0.006, # Front right knee
            0.0035, # Back left leg
            0.006, # Back left shoulder
            0.008, # Back left knee
            0.0035, # Back right leg
            0.006, # Back right shoulder
            0.008] # Back right knee

direction = [False,  True, False,
             True, False, True,
             False,  False, False,
             True, True, True]

def reorder(signal):
    # In : Shoulder, Knee, Leg (x4)
    # Out: Leg, Shoulder, Knee (x4)

    result = [0] * len(signal)
    mask   = [1, 1, -2] * 4    # Indices to swap based on In/Out (3x4=12)
    mask   = [i + m for i, m in zip(range(len(signal)), mask)]
    for i, x in zip(mask, signal):
        result[i] = x
    return result

def convert(signal):
    signal   = reorder(signal)
    rad_diff = [ri - si for ri, si in zip(initial_rad, signal)]
    pwm_diff = [rad / cal_rad for rad, cal_rad in zip(rad_diff, cal_mask)]
    print(pwm_diff)
    pwm_diff = [-1 * pwm if direc else pwm for pwm, direc in zip(pwm_diff, direction)]
    print(pwm_diff)
    result   = [base + pwm for base, pwm in zip(initial_pulse, pwm_diff)]
    result   = [int(x) for x in result]
    return result

def verify():
    signal = [0, -1.355, 1.99] * 4 # Rest position
    target = convert(signal)
    deltas = [i - t for i, t in zip(initial_pulse, target)]
    for delta in deltas:
        assert delta == 0, 'Moving from initial position to initial position does not produce zero movement..'

def communicate(socket, target):
    byt = lambda x : str(x).encode('utf-8')
    legs    = [(b'fl ', 0),
               (b'fr ', 3),
               (b'bl ', 6),
               (b'br ', 9)]
    for leg, i in legs:
        command = leg + b' '.join(byt(target[i]) for i in range(i, i + 3)) + b'\n'
        print(command)
        socket.send(command)

def rest_stand(i):
    if i % 2 == 0:
        signal = [0, -0.886, 1.3] * 4  # Standing position
    else:
        signal = [0, -1.355, 1.99] * 4 # Rest position
    return signal

raw_files = ['walk_raw.txt', 'turn_raw.txt']

raw_runs = dict()
for fname in raw_files:
    raw = []
    with open(fname, 'r') as infile:
        for line in infile:
            raw.append([float(x) for x in line.split(' ') if len(x.strip()) > 0])
    raw_runs[fname.split('.')[0]] = raw

def copy_walk(i):
    walk_raw = raw_runs['walk_raw']
    return walk_raw[i % len(walk_raw)]

def copy_turn(i):
    turn_raw = raw_runs['turn_raw']
    return turn_raw[i % len(turn_raw)]

def rest(i):
    return [0, -1.355, 1.99] * 4 # Rest position

def main():
    verify()
    socket = initialize(address='3C:61:05:30:37:56')
    try:
        for i in range(10000000):
            signal = rest_stand(i)
            # signal = copy_walk(i)
            # signal = copy_turn(i)
            # signal = rest(i)

            print(i)
            print(signal)
            target = convert(signal)
            print(target)
            communicate(socket, target)
            # sleep(0.001)
            sleep(3)
    finally:
        socket.close()

if __name__ == '__main__':
    main()
