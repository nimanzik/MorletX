import polars as pl

from ai_kit.time_freq.stft import STFT
from ai_kit.time_freq.wavelet import ComplexMorletFilterBank

df = pl.read_csv(
    '/home/nima/projects/bdim/nima_wip/ai-kit/playground/data/089d8e39-105b-4cfc-abcb-08209531162e/itendo2_rn-93_pn-Halter_data_fix20240715.csv',
    try_parse_dates=True,
)
data = df['data'].to_numpy()

deltat = 65 * 1e-6
sampling_freq = 1 / deltat

# Spectrogram
print('Plotting spectrogram...')
stft = STFT(sampling_freq, window_duration=0.2, overlap=0.8, array_engine='numpy')
fig_specgram = stft.plot_spectrogram_mpl(data)
fig_specgram.savefig('spectrogram.png')

# Scalogram
print('Plotting scalogram...')
fbank = ComplexMorletFilterBank(
    n_octaves=8,
    n_intervals=8,
    shape_ratio=5,
    duration=5.0,
    sampling_freq=sampling_freq,
    array_engine='cupy',
)
fig_scalogram = fbank.plot_scalogram_mpl(data, mode='power')
fig_scalogram.savefig('scalogram.png')
