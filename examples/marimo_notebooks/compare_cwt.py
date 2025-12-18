import marimo

__generated_with = "0.11.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import math

    import matplotlib.pyplot as plt
    import numpy as np
    import scipy as sp

    from ai_kit.time_freq.wavelet import ComplexMorletWaveletGroup, compute_cmorlet_center_freqs

    kappa = 2
    w = (kappa * math.pi) / (math.sqrt(2 * math.log(2)))
    # w_act = 5.336446256636997  # np.sqrt(2 / np.log(2)) / 2 * 2 * np.pi

    fs = 500
    dt = 1 / fs
    tau = 3.0
    nt = int(round(tau / dt)) + 1
    t = np.arange(nt) * dt
    sig = 5 * np.cos(2 * math.pi * (50 + 10 * t) * t) + 3 * np.sin(40 * math.pi * t)

    fs = 1 / dt
    center_freqs = compute_cmorlet_center_freqs(n_octaves=4, n_intervals=25, shape_ratio=kappa, sampling_freq=fs)

    widths = (w * fs) / (2 * math.pi * center_freqs)
    cwtm = sp.signal.cwt(sig, sp.signal.morlet2, widths, w=w)

    # ============================================================

    wg = ComplexMorletWaveletGroup(
        center_freqs=center_freqs,
        shape_ratios=kappa,
        duration=1.5,
        sampling_freq=fs,
    )
    sig_trans = wg.transform(sig, demean=False, tukey_alpha=None, mode='complex')

    # ============================================================

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    ax0, ax1 = axs

    im0 = ax0.pcolormesh(t, center_freqs, np.abs(cwtm), cmap='viridis')
    fig.colorbar(im0, ax=ax0)
    ax0.set_title('SciPy')

    norm_fact = np.pi**0.25
    im1 = ax1.pcolormesh(t, wg.center_freqs, np.abs(sig_trans) / norm_fact, cmap='viridis')
    fig.colorbar(im1, ax=ax1)
    ax1.set_title('Mine')

    plt.show()
    return (
        ComplexMorletWaveletGroup,
        ax0,
        ax1,
        axs,
        center_freqs,
        compute_cmorlet_center_freqs,
        cwtm,
        dt,
        fig,
        fs,
        im0,
        im1,
        kappa,
        math,
        norm_fact,
        np,
        nt,
        plt,
        sig,
        sig_trans,
        sp,
        t,
        tau,
        w,
        wg,
        widths,
    )


@app.cell
def _(cwtm, norm_fact, np, plt, sig_trans):
    a = np.abs(sig_trans) / norm_fact
    b = np.abs(cwtm)
    diff = a- b
    mu = np.mean(diff)
    sigma = np.std(diff, ddof=1)
    vmin = max(diff.min(), mu - 3 * sigma)
    vmax = min(diff.max(), mu + 3 * sigma)
    v = max(abs(diff.min()), abs(diff.max()))

    fig_diff, ax_diff = plt.subplots(figsize=(5.5, 4))
    im_diff = ax_diff.pcolormesh(diff, cmap='RdGy', vmin=-v, vmax=+v)
    fig_diff.colorbar(im_diff, ax=ax_diff)
    ax_diff.set_title('Difference')

    plt.show()
    return a, ax_diff, b, diff, fig_diff, im_diff, mu, sigma, v, vmax, vmin


@app.cell
def _(compute_cmorlet_center_freqs, fs, math, np):
    k = 4
    freqs = compute_cmorlet_center_freqs(4, 25, k, fs)
    w0 = (k * math.pi) / (math.sqrt(2.0 * math.log(2.0)))

    s1 = w0 * fs / (2 * math.pi * freqs)
    s2 = k / (freqs * math.sqrt(8 * math.log(2))) * fs
    np.testing.assert_almost_equal(s1, s2)

    print(s1[:10])
    print(s2[:10])
    return freqs, k, s1, s2, w0


@app.cell
def _(wg):
    wg.plot_responses_mpl(normalize=False)
    return


@app.cell
def _(np, plt, wg):
    wform = wg.waveforms[0]

    what = np.fft.rfft(wform.real)

    _, resps = wg.magnitude_responses(normalize=False)

    # The adjustment for using the real FFT (rfft) -> DFT * 2
    # The Fourier Transform and DFT coefficients differ only by the constant Δt -> FT = Δt * DFT
    plt.plot(abs(what) * 2 / wg.sampling_freq, label='DFT')
    plt.plot(resps[0], '--', label='Analytical FT')
    plt.legend()
    plt.show()
    return resps, wform, what


if __name__ == "__main__":
    app.run()
