import numpy as np
from scipy import fftpack


class FFT:
    """
    Wrapper of scipy.fftpack. Provide compatible method.

    Property
    --------
    fs: int | float
        Sampling frequency.
    n: int
        Length of data of signal
    fft_freq
    """

    def __init__(self, signal, dt=None, fs=None, window=None):
        """
        Parameters
        ----------
        signal: np.ndarray | pd.Series
            Array like objects of signal.
        dt: int | float
            Time interval of sampling
        fs: int | float
            Samping frequency
        window: window function
        """
        if dt is None and fs is None:
            raise Exception("Arg of either 'dt' or 'fs' is required.")

        self.fs = fs if fs is not None else 1/dt
        self.n = len(signal)
        self.fft_freq = fftpack.fftfreq(n=self.n, d=1/self.fs)
        self.window = window
        _signal = signal if window is None else signal * window(self.n)
        self.fft_res = fftpack.fft(_signal)

    def __repr__(self):
        return f"""
        Result of FFT 
        -------------
        Signal length: {self.n}
        Sampling frequency: {self.fs}
        Window function: {'rectangle' if self.window is None else self.window}
        """

    def fft_raw(self):
        return self.fft_res

    def freq_raw(self):
        return self.fft_freq

    def summary(self):
        """
        Returns summary dict of FFT result. They are in positive space of frequency.

        Return
        ------
        Dict[str, numpy.ndarray]
            frequency: numpy.ndarray
                Frequencies of spector
            amplitude: numpy.ndarray
                Coefficients of Fourier spector
            power: numpy.ndarray
                Power density of spector
            fft: numpy.ndarra
                Complex values of result of FFT

        """
        return {
            "frequency": FFT.positive_space_of(self.fft_freq, self.n),
            "period": 1/FFT.positive_space_of(self.fft_freq, self.n),
            "amplitude": FFT.fft_fourier_spector(FFT.positive_space_of(self.fft_res, self.n), self.n),
            "power": FFT.fft_power(FFT.positive_space_of(self.fft_res, self.n)),
            "fft": FFT.positive_space_of(self.fft_res, self.n)
        }

    @staticmethod
    def positive_space_of(fft_res, data_length):
        return fft_res[1:int(data_length/2)]

    @staticmethod
    def fft_power(c):
        return (c.real*c.real+c.imag*c.imag)

    @staticmethod
    def fft_fourier_spector(c, data_length):
        return np.abs(c) * 2/data_length
