// Copyright (c) 2026 Elias S. G. Carotti

use rustfft::{num_complex::Complex, FftPlanner};

use crate::window::{window_value, Window};

/// Direct matched filter / valid cross-correlation.
///
/// This returns one output sample for each position where the whole probe fits
/// inside the input signal. It is O(NM), which is intentionally simple for the
/// first version. Use FFT convolution later for long probes or recordings.
pub fn matched_filter(signal: &[f32], probe: &[f32]) -> Vec<f32> {
    if probe.is_empty() || signal.len() < probe.len() {
        return Vec::new();
    }

    let out_len = signal.len() - probe.len() + 1;
    let mut out = vec![0.0; out_len];

    for lag in 0..out_len {
        let mut acc = 0.0;

        for i in 0..probe.len() {
            acc += signal[lag + i] * probe[i];
        }

        out[lag] = acc;
    }

    out
}

/// FFT-based cross-correlation with configurable phase-transform whitening.
///
/// `phat_weight` controls the amount of spectral whitening:
/// - `0.0`: plain matched filter (no whitening, equivalent to time-domain)
/// - `1.0`: full GCC-PHAT (magnitude completely discarded, sharpest peaks)
/// - `0.0..1.0`: intermediate, raising the denominator to this power
///
/// Returns the linear (aperiodic) cross-correlation for non-negative lags
/// `0..signal.len() - probe.len() + 1`, matching the output size of
/// `matched_filter`.
pub fn gcc_phat(signal: &[f32], probe: &[f32], phat_weight: f32) -> Vec<f32> {
    if probe.is_empty() || signal.len() < probe.len() {
        return Vec::new();
    }

    let out_len = signal.len() - probe.len() + 1;
    let fft_len = (signal.len() + probe.len() - 1).next_power_of_two();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_len);
    let ifft = planner.plan_fft_inverse(fft_len);

    let mut sig_buf: Vec<Complex<f32>> = signal
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .chain(std::iter::repeat_n(
            Complex::new(0.0, 0.0),
            fft_len - signal.len(),
        ))
        .collect();

    let mut probe_buf: Vec<Complex<f32>> = probe
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .chain(std::iter::repeat_n(
            Complex::new(0.0, 0.0),
            fft_len - probe.len(),
        ))
        .collect();

    fft.process(&mut sig_buf);
    fft.process(&mut probe_buf);

    let epsilon = 1e-10;
    for (s, p) in sig_buf.iter_mut().zip(probe_buf.iter()) {
        let cross = *s * p.conj();

        if phat_weight > 0.0 {
            let mag = cross.norm();
            let denom = mag.powf(phat_weight).max(epsilon);
            *s = cross / denom;
        } else {
            *s = cross;
        }
    }

    ifft.process(&mut sig_buf);

    let scale = 1.0 / fft_len as f32;
    sig_buf.iter().take(out_len).map(|c| c.re * scale).collect()
}

/// FFT-based cross-correlation with spectral whitening and a spectral window.
///
/// Same as `gcc_phat`, but applies `window` to the cross-spectrum bins before
/// the IFFT. This suppresses correlation sidelobes at the cost of slightly
/// wider peaks. Pass `Window::None` for no windowing (equivalent to
/// `gcc_phat`).
pub fn gcc_phat_windowed(
    signal: &[f32],
    probe: &[f32],
    phat_weight: f32,
    window: Window,
) -> Vec<f32> {
    if probe.is_empty() || signal.len() < probe.len() {
        return Vec::new();
    }

    let out_len = signal.len() - probe.len() + 1;
    let fft_len = (signal.len() + probe.len() - 1).next_power_of_two();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_len);
    let ifft = planner.plan_fft_inverse(fft_len);

    let mut sig_buf: Vec<Complex<f32>> = signal
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .chain(std::iter::repeat_n(
            Complex::new(0.0, 0.0),
            fft_len - signal.len(),
        ))
        .collect();

    let mut probe_buf: Vec<Complex<f32>> = probe
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .chain(std::iter::repeat_n(
            Complex::new(0.0, 0.0),
            fft_len - probe.len(),
        ))
        .collect();

    fft.process(&mut sig_buf);
    fft.process(&mut probe_buf);

    let epsilon = 1e-10;
    for (k, (s, p)) in sig_buf.iter_mut().zip(probe_buf.iter()).enumerate() {
        let cross = *s * p.conj();

        let weighted = if phat_weight > 0.0 {
            let mag = cross.norm();
            let denom = mag.powf(phat_weight).max(epsilon);
            cross / denom
        } else {
            cross
        };

        let w = window_value(window, k, fft_len);
        *s = weighted * w;
    }

    ifft.process(&mut sig_buf);

    let scale = 1.0 / fft_len as f32;
    sig_buf.iter().take(out_len).map(|c| c.re * scale).collect()
}

/// FFT-based matched filter with a spectral window.
///
/// Equivalent to `gcc_phat_windowed` with `phat_weight = 0.0`.
pub fn matched_filter_windowed(signal: &[f32], probe: &[f32], window: Window) -> Vec<f32> {
    gcc_phat_windowed(signal, probe, 0.0, window)
}

/// Full aperiodic autocorrelation for lags `-(N-1)..=(N-1)`.
///
/// The zero-lag result is located at `out.len() / 2`.
pub fn aperiodic_autocorrelation(signal: &[f32]) -> Vec<f32> {
    if signal.is_empty() {
        return Vec::new();
    }

    let n = signal.len();
    let mut out = vec![0.0; 2 * n - 1];
    let center = n - 1;

    for lag in -(n as isize - 1)..=(n as isize - 1) {
        let mut acc = 0.0;

        for i in 0..n {
            let j = i as isize - lag;

            if (0..n as isize).contains(&j) {
                acc += signal[i] * signal[j as usize];
            }
        }

        out[(center as isize + lag) as usize] = acc;
    }

    out
}

/// Normalize a correlation-like sequence by its largest absolute sample.
pub fn normalize_abs(samples: &mut [f32]) {
    let peak = samples.iter().copied().map(f32::abs).fold(0.0, f32::max);

    if peak <= 0.0 {
        return;
    }

    for x in samples {
        *x /= peak;
    }
}

// vim: set ts=4 sw=4 et:
