#!/usr/bin/env python
from config.sampling_config import num_training_samples
from sample_collector import collect_n_samples
from utils import write_samples_to_disk

print("Gathering control data, Stay Still!")
control_samples = collect_n_samples(num_training_samples)
write_samples_to_disk(control_samples, 'control')
print("done.")

print("Gathering clench data, clench your fist")
train_samples = collect_n_samples(num_training_samples)
write_samples_to_disk(train_samples, 'train')
print("done.")


