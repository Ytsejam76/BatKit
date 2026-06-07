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
pub mod window;

/// Speed of sound in air in m/s
/// See: <https://en.wikipedia.org/wiki/Speed_of_sound>
pub const SPEED_OF_SOUND_M_S: f32 = 343.0;

#[cfg(test)]
mod tests {
    use crate::{
        correlate::{aperiodic_autocorrelation, matched_filter, normalize_abs},
        golay::generate_golay_pair,
        peak::{find_peaks, Peak},
        probe::{apply_hann_window, linear_chirp, normalize_peak},
        range::{delay_samples_to_distance_m, distance_m_to_delay_samples},
    };

    const SAMPLE_RATE: f32 = 48_000.0;

    fn synthetic_recording(
        probe: &[f32],
        direct_gain: f32,
        echoes: &[(f32, f32)],
    ) -> Vec<f32> {
        let max_echo_delay = echoes
            .iter()
            .map(|(distance_m, _)| distance_m_to_delay_samples(*distance_m, SAMPLE_RATE))
            .max()
            .unwrap_or(0);
        let mut rx = vec![0.0; probe.len() + max_echo_delay + 1024];

        for (i, &x) in probe.iter().enumerate() {
            rx[i] += direct_gain * x;
        }

        for &(distance_m, gain) in echoes {
            let delay = distance_m_to_delay_samples(distance_m, SAMPLE_RATE);

            for (i, &x) in probe.iter().enumerate() {
                rx[delay + i] += gain * x;
            }
        }

        rx
    }

    fn locate_echo(
        signal: &[f32],
        probe: &[f32],
        max_echo_distance_m: f32,
    ) -> Option<(Peak, Peak, f32)> {
        let mut corr = matched_filter(signal, probe);
        normalize_abs(&mut corr);

        let max = corr.iter().copied().map(f32::abs).fold(0.0, f32::max);
        let threshold = max * 0.15;
        let min_spacing = (SAMPLE_RATE as usize / 1_000).max(1);
        let direct_window_end = (SAMPLE_RATE as usize / 4).min(corr.len().saturating_sub(1));

        let direct = find_peaks(&corr, 0, threshold, min_spacing)
            .into_iter()
            .find(|peak| peak.index <= direct_window_end)?;

        let echo_start = direct.index.saturating_add(SAMPLE_RATE as usize / 2_000);
        let echo_end = direct
            .index
            .saturating_add(distance_m_to_delay_samples(max_echo_distance_m, SAMPLE_RATE))
            .min(corr.len().saturating_sub(1));

        let echo = find_peaks(&corr, echo_start, threshold, min_spacing)
            .into_iter()
            .find(|peak| peak.index <= echo_end)?;

        let measured_delay = echo.index - direct.index;
        let measured_distance = delay_samples_to_distance_m(measured_delay, SAMPLE_RATE);

        Some((direct, echo, measured_distance))
    }

    #[test]
    fn detects_synthetic_echo() {
        let mut probe = linear_chirp(SAMPLE_RATE, 0.020, 16_000.0, 22_000.0);
        apply_hann_window(&mut probe);
        normalize_peak(&mut probe);

        let echo_distance_m = 1.25;
        let rx = synthetic_recording(&probe, 1.0, &[(echo_distance_m, 0.25)]);
        let (_, _, measured_distance) =
            locate_echo(&rx, &probe, 4.0).expect("synthetic echo should be found");

        assert!((measured_distance - echo_distance_m).abs() < 0.05);
    }

    #[test]
    fn longer_chirp_accumulates_more_matched_energy() {
        let mut short = linear_chirp(SAMPLE_RATE, 0.020, 16_000.0, 22_000.0);
        apply_hann_window(&mut short);
        normalize_peak(&mut short);

        let mut long = linear_chirp(SAMPLE_RATE, 0.040, 16_000.0, 22_000.0);
        apply_hann_window(&mut long);
        normalize_peak(&mut long);

        let short_peak = matched_filter(&short, &short)[0];
        let long_peak = matched_filter(&long, &long)[0];
        let ratio = long_peak / short_peak;

        assert!(
            ratio > 1.8,
            "expected the 40 ms probe to collect materially more energy than 20 ms, got ratio {ratio:.3}"
        );
    }

    #[test]
    fn prefers_near_echo_over_stronger_late_clutter() {
        let mut probe = linear_chirp(SAMPLE_RATE, 0.020, 16_000.0, 22_000.0);
        apply_hann_window(&mut probe);
        normalize_peak(&mut probe);

        let wall_distance_m = 0.50;
        let clutter_distance_m = 3.50;
        let rx = synthetic_recording(
            &probe,
            1.0,
            &[
                (wall_distance_m, 0.22),
                (clutter_distance_m, 0.90),
            ],
        );

        let (_, echo, measured_distance) =
            locate_echo(&rx, &probe, 4.0).expect("synthetic echo should be found");

        assert!((measured_distance - wall_distance_m).abs() < 0.08);
        assert!(
            measured_distance < clutter_distance_m - 1.0,
            "selected late clutter peak at index {}",
            echo.index
        );
    }

    #[test]
    fn rejects_echo_beyond_search_window() {
        let mut probe = linear_chirp(SAMPLE_RATE, 0.020, 16_000.0, 22_000.0);
        apply_hann_window(&mut probe);
        normalize_peak(&mut probe);

        let rx = synthetic_recording(&probe, 1.0, &[(0.50, 0.22), (59.0, 0.95)]);
        let (_, echo, measured_distance) =
            locate_echo(&rx, &probe, 4.0).expect("near echo should still be found");

        assert!((measured_distance - 0.50).abs() < 0.08);
        assert!(echo.index < distance_m_to_delay_samples(4.0, SAMPLE_RATE) + 64);
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
