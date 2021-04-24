import json, datetime
import bluetooth
import struct
from time import sleep, time

from bt_serial import initialize

from collect import extract

def main():
    headers = 'timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z'
    buff = b''
    socket = initialize(address='3C:61:05:30:37:56')
    try:
        with open('robot_data.csv', 'w') as outfile:
            outfile.write(headers)
            while True:
                try:
                    print('Receiving..')
                    buff += socket.recv(1024)
                    t = time()
                    result, buff = extract(buff)
                    for line in result:
                        print(line)
                        line = [t] + line
                        outfile.write(','.join(map(str, line)) + '\n')
                    sleep(1)
                except bluetooth.btcommon.BluetoothError as e:
                    print(e)
    finally:
        socket.close()

if __name__ == '__main__':
    main()
