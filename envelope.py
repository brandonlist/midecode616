from scipy import fftpack
from matplotlib import pyplot as plt
import numpy as np

t = np.arange(0, 1024, 1)
#operate on x_en
def envelop(x_en,display=0):
    x_en_a = x_en - x_en.mean()
    hx_a = fftpack.hilbert(x_en_a)
    x_en_up = np.sqrt(x_en_a**2 + hx_a**2)+ x_en.mean()
    # x_en_a_dw = -x_en_a
    # hx_a_dw = fftpack.hilbert(x_en_a_dw)
    # x_en_dw = -np.sqrt(x_en_a_dw**2 + hx_a_dw**2)+ x_en.mean()
    # x_en_mean = (x_en_dw+x_en_up)/2
    if display:
        plt.plot(x_en,"b",linewidth=2, label='signal')
        plt.plot(x_en_up,"r",linewidth=2, label='envelop')
        plt.legend()
        plt.show()
    return x_en_up

