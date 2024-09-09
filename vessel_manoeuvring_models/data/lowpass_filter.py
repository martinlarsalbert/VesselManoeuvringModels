from scipy import signal

def butter_lowpass(cutoff, fs, order=5, btype='low'):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype=btype, analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5, btype='low'):

    if cutoff is None or order is None:
        # Disabled filter:
        return data

    b, a = butter_lowpass(cutoff, fs, order=order, btype=btype)
    #y = signal.lfilter(b, a, data)
    y = signal.filtfilt(b, a, data)

    return y