import json, datetime
import bluetooth
import struct
from time import sleep

from bt_serial import initialize

def extract(buff):
    result = []
    if b'\n' in buff:
        chunks = buff.split(b'\n')
        buff = chunks[-1]
        for chunk in chunks:
            parts = chunk.split(b',')
            try:
                raw = [struct.unpack('f', x)[0]
                       for x in parts if len(x) > 0]
                if raw:
                    result.append(raw)
            except Exception as e:
                print(e)
                print(parts)
                print([len(p) for p in parts])
    return result, buff


def main():
    headers = 'cm,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,temp'
    buff = b''
    socket = initialize(address='3C:61:05:3D:EB:2A')
    try:
        with open('raw_data.csv', 'w') as outfile:
            outfile.write(headers)
            while True:
                try:
                    buff += socket.recv(1024)
                    result, buff = extract(buff)
                    for line in result:
                        print(line)
                        outfile.write(','.join(map(str, line)) + '\n')
                    sleep(1)
                except bluetooth.btcommon.BluetoothError:
                    pass
    finally:
        socket.close()

if __name__ == '__main__':
    main()
