// Copyright (c) 2026 Elias S. G. Carotti
//! Small DSP primitives for acoustic ranging toy projects.
//!
//! `batkit` is intentionally platform-agnostic: no Android, JNI, microphone,
//! or speaker code lives here. Feed it samples, get signal-processing results
//! back.

pub mod correlate;
pub mod golay;
pub mod peak;
pub mod probe;
pub mod range;

/// Speed of sound in air in m/s
/// See: <https://en.wikipedia.org/wiki/Speed_of_sound>
pub const SPEED_OF_SOUND_M_S: f32 = 343.0;

#[cfg(test)]
mod tests {
    use crate::{
        correlate::{aperiodic_autocorrelation, matched_filter},
        golay::generate_golay_pair,
        peak::find_peaks,
        probe::{apply_hann_window, linear_chirp, normalize_peak},
        range::{delay_samples_to_distance_m, distance_m_to_delay_samples},
    };

    #[test]
    fn detects_synthetic_echo() {
        let sample_rate = 48_000.0;

        let mut probe = linear_chirp(sample_rate, 0.020, 16_000.0, 22_000.0);
        apply_hann_window(&mut probe);
        normalize_peak(&mut probe);

        let echo_distance_m = 1.25;
        let echo_delay = distance_m_to_delay_samples(echo_distance_m, sample_rate);

        let mut rx = vec![0.0; echo_delay + probe.len() + 1024];

        for (i, &x) in probe.iter().enumerate() {
            rx[i] += x; // direct leakage
            rx[echo_delay + i] += 0.25 * x; // echo
        }

        let corr = matched_filter(&rx, &probe);

        let direct_peak = find_peaks(&corr, 0, 10.0, 64)
            .first()
            .copied()
            .expect("direct peak");

        let echo_peak = find_peaks(&corr, direct_peak.index + 64, 1.0, 64)
            .first()
            .copied()
            .expect("echo peak");

        let measured_delay = echo_peak.index - direct_peak.index;
        let measured_distance = delay_samples_to_distance_m(measured_delay, sample_rate);

        assert!((measured_distance - echo_distance_m).abs() < 0.05);
    }

    #[test]
    fn golay_pair_sidelobes_cancel() {
        let (a, b) = generate_golay_pair(4);
        let ra = aperiodic_autocorrelation(&a);
        let rb = aperiodic_autocorrelation(&b);

        assert_eq!(ra.len(), rb.len());

        let center = ra.len() / 2;

        for i in 0..ra.len() {
            let sum = ra[i] + rb[i];

            if i == center {
                assert_eq!(sum, 2.0 * a.len() as f32);
            } else {
                assert!(sum.abs() < 1e-6, "nonzero sidelobe at {i}: {sum}");
            }
        }
    }
}

// vim: set ts=4 sw=4 et:
