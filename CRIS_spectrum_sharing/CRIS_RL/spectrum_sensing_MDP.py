import numpy as np

# energy detection
def energy_detection(samples, threshold):
    energy = np.sum(np.square(samples))
    if energy > threshold:
        return True
    else:
        return False

# input the received signal energy 
def spectrum_sensing(received_signal):

    threshold = 10
    result_list = []
    for i in range(len(received_signal)):
        samples_in_range = received_signal[i]
        result = energy_detection(samples_in_range, threshold)
        result_list.append(result)
    
    pu_freq_usage = [[0 if i != j else result_list[j] for i in range(len(result_list))] for j in range(len(result_list))]
    pu_freq_usage = np.array(pu_freq_usage)

    return pu_freq_usage
