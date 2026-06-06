// Copyright (c) 2026 Elias S. G. Carotti

/// Generate a binary Golay complementary pair of length `2^order`.
/// See: <https://en.wikipedia.org/wiki/Complementary_sequences>
///
/// The recursive construction is:
///
/// ```text
/// A' = [ A,  B ]
/// B' = [ A, -B ]
/// ```
///
/// Starting with `A = [1]`, `B = [1]`.
pub fn generate_golay_pair(order: usize) -> (Vec<f32>, Vec<f32>) {
    let mut a = vec![1.0];
    let mut b = vec![1.0];

    for _ in 0..order {
        let mut next_a = Vec::with_capacity(a.len() * 2);
        let mut next_b = Vec::with_capacity(b.len() * 2);

        next_a.extend_from_slice(&a);
        next_a.extend_from_slice(&b);

        next_b.extend_from_slice(&a);
        next_b.extend(b.iter().map(|x| -*x));

        a = next_a;
        b = next_b;
    }

    (a, b)
}

/// Expand a `+1/-1` chip sequence into a simple BPSK carrier waveform.
///
/// Each chip is represented by `chip_samples` carrier samples. A `-1` chip
/// flips the carrier phase by multiplying the carrier by `-1`.
pub fn bpsk_modulate_chips(
    chips: &[f32],
    sample_rate: f32,
    carrier_hz: f32,
    chip_samples: usize,
) -> Vec<f32> {
    use std::f32::consts::TAU;

    assert!(sample_rate > 0.0);
    assert!(carrier_hz >= 0.0);
    assert!(chip_samples > 0);

    let mut out = Vec::with_capacity(chips.len() * chip_samples);

    for &chip in chips {
        let sign = if chip >= 0.0 { 1.0 } else { -1.0 };
        let base = out.len();

        for i in 0..chip_samples {
            let n = base + i;
            let t = n as f32 / sample_rate;
            out.push(sign * (TAU * carrier_hz * t).sin());
        }
    }

    out
}

// vim: set ts=4 sw=4 et:
