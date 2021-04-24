import json, datetime
import bluetooth
import struct
from time import sleep, time

from bt_serial import initialize

from collect import extract

def main():
    headers = 'timestamp,cm,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,laser (1=off 0=on)\n'
    buff = b''
    socket = initialize(address='3C:61:05:3D:EB:2A')
    try:
        with open('pointer_data.csv', 'w') as outfile:
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
                        print(t)
                        outfile.write(','.join(map(str, line)) + '\n')
                    sleep(1)
                except bluetooth.btcommon.BluetoothError:
                    pass
    finally:
        socket.close()

if __name__ == '__main__':
    main()
