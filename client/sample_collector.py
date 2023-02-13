from config.sampling_config import sample_size
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


def collect_n_samples(n: int):
    idx = 0
    sample = [0] * sample_size
    num_samples = 0
    samples = [None] * n
    while num_samples < n:
        serial_input = ser.readline()
        try:
            sample_value = int(str(serial_input)[2:-5])
            sample[idx] = sample_value
            idx += 1
        except:
            print('[TEXT] ' + str(serial_input))

        if idx == sample_size - 1:
            samples.append(sample)
            idx = 0
            sample = [0] * sample_size
            num_samples += 1