#!/usr/bin/env python
import serial
from http.server import BaseHTTPRequestHandler, HTTPServer
import time

ser = serial.Serial(
        # port='/dev/ttyS0', #Replace ttyS0 with ttyAM0 for Pi1,Pi2,Pi0
        port='/dev/cu.usbserial-02762B0D',
        baudrate=115200,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=2
)

f = open('control.csv', 'w')

num = 0
while num < 6:
    serial_input = ser.readline()
    if serial_input[:2] == b'd:':
        num += 1
        samples_bytes = str(serial_input[2:])[2:-5]
        samples = list(samples_bytes.split(','))
        f.write(str(samples_bytes) + '\n')
    else:
        print(serial_input)

f.close()
