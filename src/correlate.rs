// Copyright (c) 2026 Elias S. G. Carotti
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
