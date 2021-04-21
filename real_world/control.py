import json, datetime
import bluetooth
import struct
from time import sleep

from bt_serial import initialize

def main():
    socket = initialize(address='3C:61:05:30:37:54')
    try:
        while True:
            try:
                socket.send(b's 3 200')
            except bluetooth.btcommon.BluetoothError:
                pass
    finally:
        socket.close()

if __name__ == '__main__':
    main()
