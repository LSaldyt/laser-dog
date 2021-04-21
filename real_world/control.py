import json, datetime
import bluetooth
import struct
from time import sleep

from bt_serial import initialize

# Notes from standup() task

# Motors value t0:
# [ 0.         -1.35539923  1.99077815  0.         -1.35539923  1.99077815
#   0.         -1.35539923  1.99077815  0.         -1.35539923  1.99077815]

# Motors value tf:
# [ 0.         -0.88643435  1.30197369  0.         -0.88643435  1.30197369
#   0.         -0.88643435  1.30197369  0.         -0.88643435  1.30197369]


#            0    1    2    3    4    5    6   7    8    9    10   11
initial   = [200, 360, 270, 320, 380, 300, 70, 310, 220, 460, 330, 200]

# direction = [True,

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

def main():
    socket = initialize(address='3C:61:05:30:37:56')
    try:
        i = 0
        while True:
            i += 1
            try:
                # if i % 2 == 0:
                #     socket.send(b's 3 200 \n')
                # else:
                state = initial
                # command = b' '.join(str(x).encode('utf-8') for x in state)
                # socket.send(b'c ' + command + b' \n')
                for servo_i, setting in enumerate(state):
                    socket.send(b's ' + str(servo_i).encode('utf-8') + b' ' + str(setting).encode('utf-8') + b' \n')
                sleep(1)
            except bluetooth.btcommon.BluetoothError:
                pass
    finally:
        socket.close()

if __name__ == '__main__':
    main()
