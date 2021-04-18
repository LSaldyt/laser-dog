import json, datetime
import bluetooth
from time import sleep

address = '3C:61:05:3D:EB:2A'


def main():
    service = bluetooth.find_service(address=address)

    if not service:
        print(f'Could not find bluetooth device: {address}')
        return

    first = service[0]
    port = first['port']
    name = first['name']
    host = first['host']

    socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    socket.connect((host, port))
    try:
        while True:
            socket.send(input())
            print(socket.recv(1024))
    finally:
        socket.close()

if __name__ == '__main__':
    main()
