import numpy as np
import scipy.fftpack as fft

def calc_tx_power(power, path_loss, samples):
    signal_power = power - path_loss
    samples = samples * 10 ** (signal_power/10)  # dBm to mw

    return samples

def spectrum_sensing(BS_position, RIS_position, pu_power, bandwidth, num_subchannel, noise_power, channel_state):
    # para setting
    fs = 3000e6
    samples_len = int(fs/bandwidth)

    range_start = 1399.5e6
    range_end = range_start + bandwidth * num_subchannel

    BS_x, BS_y, BS_z = BS_position

    RIS_x, RIS_y, RIS_z = RIS_position

    distance = np.sqrt((BS_x - RIS_x)**2 + (BS_y - RIS_y)
                       ** 2 + (BS_z - RIS_z)**2)
    n = -2.2
    path_loss = np.sqrt(10**(-30/10)*np.power(distance, n))

    # primary user freq.
    candidate = [i for i in range(
        int(range_start/1e6), int(range_end/1e6), int(bandwidth/1e6))]
    primary_user_freq = candidate * channel_state
    primary_user_freq = [x * 1e6 for x in primary_user_freq]
    num = np.sum(channel_state)
    # primary_user_freq = (np.random.choice(candidate, num, replace=False))*1e6

    if num != 0:
        time = np.arange(samples_len)/fs
        samples = 0
        for i in range(len(channel_state)):
            if channel_state[i] != 0:
                power = pu_power[i]
                freq = primary_user_freq[i]
                signal = np.sin(2 * np.pi * freq * time)
                samples += calc_tx_power(power, path_loss, signal)

        # generate noise
        noise_mean = 0
        noise_std = 1e-2
        noise = np.random.normal(noise_mean, noise_std, len(samples))
        noise = noise * 10 ** (noise_power/10)

        samples = samples + noise

    else:
        samples = [0]*3000

    # frequency analysis
    freq = fft.fft(samples)
    freq = np.abs(freq)
    freq_index_start = int(range_start * len(samples)/fs)
    freq_index_end = int(range_end * len(samples)/fs)
    freq_index_bandwidth = int(bandwidth*len(samples)/fs)
    samples = np.ones(3)
    num = 0

    for i in range(freq_index_start, freq_index_end, freq_index_bandwidth):

        samples_in_range = freq[i:i+freq_index_bandwidth]
    
        samples[num] = samples_in_range
        num += 1

    return samples

def TrainData(BS_position, RIS_position):
    pu_msg = np.ones(shape=(41,6))
    bandwidth = 1e6
    num_subchannel = 3
    noise_power = -147

    channel_state = np.random.choice(2, num_subchannel)

    # transition matrix
    stayGood_prob = np.random.uniform(0.5, 1, num_subchannel)
    stayBad_prob = np.random.uniform(0, 0.5, num_subchannel)

    for i in range(41):
           
        pu_power = np.random.randint(20, 30, 3)
        pu_power = pu_power * channel_state

        pu_signal= spectrum_sensing(
            BS_position, RIS_position, pu_power, bandwidth, num_subchannel, noise_power, channel_state)

        pu_msg[i,0] = 10**(pu_power[0]/10)/1000 if pu_power[0]!= 0 else 0
        pu_msg[i,1] = 10**(pu_power[1]/10)/1000 if pu_power[1]!= 0 else 0
        pu_msg[i,2] = 10**(pu_power[2]/10)/1000 if pu_power[2]!= 0 else 0
        pu_msg[i,3:] = pu_signal

        # The probability of staying in current state in next time slot
        stay_prob = channel_state * stayGood_prob + (1-channel_state)*stayBad_prob
        tmp_dice = np.random.uniform(0, 1, num_subchannel) # roll the dice between 0 and 1
        stay_index = tmp_dice < stay_prob # 1: stay in current state, 0: change state

        # Update the channel state
        channel_state = channel_state * stay_index + (1-channel_state) * (1-stay_index)
    
    np.savetxt("Train_PU_Spectrum_MDP.csv", pu_msg, delimiter=',')	

def TestData(BS_position, RIS_position):
    pu_msg = np.ones(shape=(41,6))
    bandwidth = 1e6
    num_subchannel = 3
    noise_power = -147
    channel_state = np.random.choice(2, num_subchannel)

    stayGood_prob = np.random.uniform(0.5, 1, num_subchannel)
    stayBad_prob = np.random.uniform(0, 0.5, num_subchannel)

    for i in range(41):
        
        pu_power = np.random.randint(20, 30, 3)
        pu_power = pu_power * channel_state

        pu_signal= spectrum_sensing(
            BS_position, RIS_position, pu_power, bandwidth, num_subchannel, noise_power, channel_state)
        pu_msg[i,0] = 10**(pu_power[0]/10)/1000 if pu_power[0]!= 0 else 0
        pu_msg[i,1] = 10**(pu_power[1]/10)/1000 if pu_power[1]!= 0 else 0
        pu_msg[i,2] = 10**(pu_power[2]/10)/1000 if pu_power[2]!= 0 else 0
        pu_msg[i,3:] = pu_signal

        stay_prob = channel_state * stayGood_prob + (1-channel_state)*stayBad_prob
        tmp_dice = np.random.uniform(0, 1, num_subchannel) # roll the dice between 0 and 1
        stay_index = tmp_dice < stay_prob # 1: stay in current state, 0: change state

        # Update the channel state
        channel_state = channel_state*stay_index + (1-channel_state)*(1-stay_index)   

    np.savetxt("Test_PU_Spectrum_MDP.csv", pu_msg, delimiter=',')

if __name__ == '__main__':
    np.random.seed(1000)
    BS_position = np.array([-50, -50, 10])
    RIS_position = np.array([0, 0, 20])
    TrainData(BS_position, RIS_position)
    TestData(BS_position, RIS_position)
