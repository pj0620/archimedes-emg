#!/usr/bin/env python
import serial

from config.sampling_config import sample_size

ser = serial.Serial(
        # port='/dev/ttyS0', #Replace ttyS0 with ttyAM0 for Pi1,Pi2,Pi0
        port='/dev/cu.usbserial-02762B0D',
        baudrate=115200,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=2
)

write_to_file = False

f = open('data_1/clench_grabbing_button_short.csv', 'a')

idx = 0
sample = [0] * sample_size
num_samples = 0
while num_samples < 100:
    serial_input = ser.readline()
    try:
        sample_value = int(str(serial_input)[2:-5])
        sample[idx] = sample_value
        idx += 1
    except:
        print('text: ' + str(serial_input))

    if idx == sample_size-1:
        print(sample)
        f.write(','.join(str(s) for s in sample) + '\n')
        idx = 0
        sample = [0] * sample_size
        num_samples += 1

f.close()

