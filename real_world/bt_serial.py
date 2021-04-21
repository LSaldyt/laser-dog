import json, datetime
import bluetooth
import struct
from time import sleep

def initialize(address=''):
    service = bluetooth.find_service(address=address)

    if not service:
        raise ValueError(f'Could not find bluetooth device: {address}')

    first = service[0]
    port = first['port']
    name = first['name']
    host = first['host']

    socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    socket.connect((host, port))
    socket.settimeout(0.1)
    return socket
