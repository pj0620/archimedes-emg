#!/usr/bin/env python
import serial

ser = serial.Serial(
        # port='/dev/ttyS0', #Replace ttyS0 with ttyAM0 for Pi1,Pi2,Pi0
        port='/dev/cu.usbserial-02762B0D',
        baudrate=115200,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=2
)

while True:
    samples_bytes = ser.readline()
    if samples_bytes[:2] != b'd:':
        continue
    samples_bytes = samples_bytes[2:]
    samples_bytes_list = samples_bytes.split(b',')
    samples = map(int, samples_bytes_list)
    print(list(samples))
