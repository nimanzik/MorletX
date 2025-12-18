import marimo

__generated_with = "0.15.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from pathlib import Path

    import cupy as cp
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import scipy as sp
    from plotly import graph_objects as go

    from morletx.core import MorletFilterBank

    plt.style.use("seaborn-v0_8")
    return MorletFilterBank, Path, cp, np, plt


@app.cell
def _(Path, np):
    data_dir = Path('/home/nima/projects/personal/dev/MorletX/examples/data')
    sig_npz_file = data_dir / 'signal.npz'

    sig_npz = np.load(sig_npz_file)
    values = sig_npz["values"]
    times = sig_npz["times"]

    mean = values.mean()
    values = values - mean

    print(values.shape)
    print(values)
    return times, values


@app.cell
def _(times):
    duration = times.max() - times.min()
    print(f'{duration=}')
    return


@app.cell
def _(mo, plt, times, values):
    fig_sig, ax_sig = plt.subplots()
    ax_sig.plot(times, values)
    ax_sig.set(xlabel="Time [s]", ylabel="Amplitude values")
    fig_sig.set_layout_engine("tight")
    fig_sig.savefig("example_signal.png")
    mo.mpl.interactive(fig_sig)
    return


@app.cell
def _(MorletFilterBank):
    # fs = 16_000
    deltat = 65 * 1e-6
    fs = 1 / deltat

    fbank = MorletFilterBank(
        n_octaves=8,
        n_intervals=4,
        shape_ratio=5,
        duration=2.0,
        sampling_freq=fs,
        array_engine='numpy',
    )
    return (fbank,)


@app.cell
def _(fbank):
    freqs, responses = fbank.magnitude_responses()
    print(freqs.shape, responses.shape)

    print(fbank.center_freqs.min(), fbank.center_freqs.max())
    return freqs, responses


@app.cell
def _(cp, freqs, mo, plt, responses):
    fig_fbank, ax_fbank = plt.subplots()
    for resp in responses:
        ax_fbank.plot(cp.asnumpy(freqs), cp.asnumpy(resp))

    # for resp in cp.fft.fft(fbank.waveforms):
    #     ax.plot(abs(resp.get()))

    ax_fbank.set(xlabel="Frequency [Hz]", ylabel="Spectral amplitude, normalised")
    fig_fbank.set_layout_engine("tight")
    fig_fbank.savefig("filter_bank.png")
    mo.mpl.interactive(fig_fbank)
    return


@app.cell
def _(values):
    x_in = values
    return (x_in,)


@app.cell
def _(fbank, x_in):
    x_trans = fbank.transform(x_in)
    return (x_trans,)


@app.cell
def _(fbank, x_trans):
    if fbank.array_engine == 'cupy':
        x_trans_np = x_trans.get()
    else:
        x_trans_np = x_trans

    print(x_trans_np.shape)
    return


@app.cell
def _(fbank, plt, x_in):
    fig_scalo2 = fbank.plot_scalogram_mpl(x_in, mode='power')
    plt.grid(False)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
