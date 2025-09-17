import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from dataclasses import dataclass
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    from morletx import MorletFilterBank

    plt.style.use("seaborn-v0_8")
    return MorletFilterBank, Path, dataclass, np, plt


@app.cell
def _(mo):
    mo.md(r"""## Load example signal data""")
    return


@app.cell
def _(dataclass, np):
    @dataclass
    class Signal:
        times: np.ndarray
        values: np.ndarray
        fs: float

        @property
        def duration(self):
            return self.times.max() - self.times.min()

        @property
        def num_samples(self):
            return len(self.values)

        def __repr__(self):
            return f"Signal duration: {self.duration} s | num. of samples: {self.num_samples} | sampling frequency: {self.fs}"
    return (Signal,)


@app.cell
def _(Path, Signal, np):
    npz_file = Path(__file__).parents[1] / "data" / "fin_whale_song.npz"
    npz_data = np.load(npz_file)

    signal = Signal(
        times=npz_data["times"], values=npz_data["values"], fs=npz_data["fs"]
    )

    print(signal)
    return (signal,)


@app.cell
def _(mo):
    mo.md(r"""## Fin-Whale song recording signal""")
    return


@app.cell
def _(plt, signal):
    fig_sig, ax_sig = plt.subplots(figsize=(10, 6))
    ax_sig.plot(signal.times, signal.values)
    ax_sig.set(
        xlabel="Time [s]",
        ylabel="Acoustic Amplitude",
        title="Fin-Whale song recording | Bandpass filtered 12–30 Hz",
    )
    # mo.mpl.interactive(fig_sig)
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""## Apply the CWT and compute scalogram""")
    return


@app.cell
def _(MorletFilterBank, signal):
    filt_bank = MorletFilterBank(
        n_octaves=2,
        n_intervals=10,
        shape_ratio=5.0,
        duration=1.0,
        sampling_freq=signal.fs,
        array_engine="numpy",
    )

    mode = "magnitude"
    scalogram = filt_bank.transform(signal.values, mode=mode, detach_from_device=True)

    print(f"Scalogram array shape: {scalogram.shape}")
    return filt_bank, mode, scalogram


@app.cell
def _(mo):
    mo.md(r"""## Display the scalogram""")
    return


@app.cell
def _(filt_bank, mode, plt, scalogram):
    fig_sgram, ax_sgram = plt.subplots(figsize=(10, 6))
    filt_bank.plot_scalogram(
        ax=ax_sgram,
        scalogram=scalogram,
        mode=mode,
    )

    ax_sgram.grid(False)
    # mo.mpl.interactive(fig_sgram)
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""## Display the frequency responses of the filterbank""")
    return


@app.cell
def _(filt_bank, plt):
    fig_resps, ax_resps = plt.subplots(figsize=(10, 6))
    filt_bank.plot_responses(ax_resps, n_fft=512)
    # mo.mpl.interactive(fig_resps)
    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
