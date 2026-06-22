// Copyright (c) 2026 Elias S. G. Carotti

use crate::correlate::gcc_phat;

/// Estimated direct-coupling profile for a specific probe signal.
///
/// Constructed from one or more calibration recordings made with no nearby
/// reflectors. Rather than estimating the channel impulse response (which is
/// ill-conditioned for narrowband probes), this stores the correlation
/// signature of the direct coupling path. At measurement time, this signature
/// is subtracted from the measurement's correlation, revealing only echoes.
///
/// The profile is tied to a specific probe waveform — if the probe changes,
/// recalibrate.
#[derive(Debug, Clone)]
pub struct CouplingProfile {
    /// Averaged correlation of calibration recordings against the probe.
    reference_correlation: Vec<f32>,
    sample_rate: f32,
}

impl CouplingProfile {
    /// Build a coupling profile from one or more calibration recordings.
    ///
    /// `probe` is the signal played through the speaker.
    /// `recordings` are open-air captures (no nearby reflectors).
    /// `phat_weight` should match the weight used during measurement.
    ///
    /// Multiple recordings are correlated individually and then averaged.
    /// This is robust to timing jitter between recordings — correlation peaks
    /// are always at the same lag regardless of when recording started.
    pub fn from_calibrations(
        probe: &[f32],
        recordings: &[&[f32]],
        sample_rate: f32,
        phat_weight: f32,
    ) -> Self {
        assert!(!probe.is_empty());
        assert!(!recordings.is_empty());
        assert!(sample_rate > 0.0);

        let max_len = recordings.iter().map(|r| r.len()).max().unwrap();
        let out_len = max_len.saturating_sub(probe.len()) + 1;
        let mut reference_correlation = vec![0.0; out_len];

        for recording in recordings {
            let corr = gcc_phat(recording, probe, phat_weight);

            for (i, &v) in corr.iter().enumerate() {
                if i < reference_correlation.len() {
                    reference_correlation[i] += v;
                }
            }
        }

        let n = recordings.len() as f32;
        for v in reference_correlation.iter_mut() {
            *v /= n;
        }

        Self {
            reference_correlation,
            sample_rate,
        }
    }

    /// Subtract the direct-coupling signature from a measurement correlation.
    ///
    /// `measurement_correlation` is the gcc_phat output of the actual recording.
    /// Returns the residual correlation where the direct coupling has been
    /// suppressed, leaving echoes.
    ///
    /// An optional `gain` factor scales the reference before subtraction,
    /// accounting for volume differences between calibration and measurement.
    /// Pass `None` to auto-estimate the gain from the peak ratio.
    pub fn subtract_correlation(
        &self,
        measurement_correlation: &[f32],
        gain: Option<f32>,
    ) -> Vec<f32> {
        let g = gain.unwrap_or_else(|| self.estimate_gain(measurement_correlation));

        measurement_correlation
            .iter()
            .enumerate()
            .map(|(i, &m)| {
                let r = self.reference_correlation.get(i).copied().unwrap_or(0.0);
                m - g * r
            })
            .collect()
    }

    /// Estimate the gain ratio between a measurement and the stored reference,
    /// based on the direct-path peak amplitude.
    fn estimate_gain(&self, measurement_correlation: &[f32]) -> f32 {
        let search_window = (self.sample_rate as usize / 200).max(1);

        let ref_peak = self
            .reference_correlation
            .iter()
            .take(search_window)
            .copied()
            .map(f32::abs)
            .fold(0.0_f32, f32::max);

        let meas_peak = measurement_correlation
            .iter()
            .take(search_window)
            .copied()
            .map(f32::abs)
            .fold(0.0_f32, f32::max);

        if ref_peak > 1e-10 {
            meas_peak / ref_peak
        } else {
            1.0
        }
    }

    pub fn reference_correlation(&self) -> &[f32] {
        &self.reference_correlation
    }

    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::correlate::normalize_abs;
    use crate::peak::find_peaks;
    use crate::probe::{apply_hann_window, linear_chirp, normalize_peak};
    use crate::range::{delay_samples_to_distance_m, distance_m_to_delay_samples};

    const SAMPLE_RATE: f32 = 48_000.0;

    fn make_probe() -> Vec<f32> {
        let mut probe = linear_chirp(SAMPLE_RATE, 0.020, 16_000.0, 22_000.0);
        apply_hann_window(&mut probe);
        normalize_peak(&mut probe);
        probe
    }

    fn synthetic_recording(probe: &[f32], direct_gain: f32, echoes: &[(f32, f32)]) -> Vec<f32> {
        let max_delay = echoes
            .iter()
            .map(|(d, _)| distance_m_to_delay_samples(*d, SAMPLE_RATE))
            .max()
            .unwrap_or(0);
        let mut rx = vec![0.0; probe.len() + max_delay + 1024];

        for (i, &x) in probe.iter().enumerate() {
            rx[i] += direct_gain * x;
        }

        for &(dist, gain) in echoes {
            let delay = distance_m_to_delay_samples(dist, SAMPLE_RATE);
            for (i, &x) in probe.iter().enumerate() {
                rx[delay + i] += gain * x;
            }
        }

        rx
    }

    #[test]
    fn subtracts_direct_path_reveals_echo() {
        let probe = make_probe();
        let phat_weight = 0.7;
        let echo_distance = 1.0;

        // Calibration: direct path only
        let calibration = synthetic_recording(&probe, 1.0, &[]);
        let profile =
            CouplingProfile::from_calibrations(&probe, &[&calibration], SAMPLE_RATE, phat_weight);

        // Measurement: direct path + echo
        let recording = synthetic_recording(&probe, 1.0, &[(echo_distance, 0.3)]);
        let meas_corr = gcc_phat(&recording, &probe, phat_weight);

        let residual = profile.subtract_correlation(&meas_corr, None);

        // The direct-path peak (near lag 0) should be heavily suppressed
        let direct_window = 10;
        let direct_peak = residual
            .iter()
            .take(direct_window)
            .copied()
            .map(f32::abs)
            .fold(0.0_f32, f32::max);

        let echo_delay = distance_m_to_delay_samples(echo_distance, SAMPLE_RATE);
        let echo_search_start = echo_delay.saturating_sub(10);
        let echo_search_end = (echo_delay + 10).min(residual.len());
        let echo_peak = residual[echo_search_start..echo_search_end]
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0_f32, f32::max);

        assert!(
            echo_peak > direct_peak * 3.0,
            "echo ({echo_peak:.4}) should dominate over suppressed direct path ({direct_peak:.4})"
        );
    }

    #[test]
    fn handles_volume_difference() {
        let probe = make_probe();
        let phat_weight = 0.7;
        let echo_distance = 1.0;

        // Calibrate at gain 1.0
        let calibration = synthetic_recording(&probe, 1.0, &[]);
        let profile =
            CouplingProfile::from_calibrations(&probe, &[&calibration], SAMPLE_RATE, phat_weight);

        // Measure at gain 0.6 (different volume)
        let recording = synthetic_recording(&probe, 0.6, &[(echo_distance, 0.2)]);
        let meas_corr = gcc_phat(&recording, &probe, phat_weight);

        let residual = profile.subtract_correlation(&meas_corr, None);
        let mut norm_residual = residual.clone();
        normalize_abs(&mut norm_residual);

        let echo_delay = distance_m_to_delay_samples(echo_distance, SAMPLE_RATE);
        let echo_search_start = echo_delay.saturating_sub(10);
        let echo_search_end = (echo_delay + 10).min(norm_residual.len());

        let echo_peak = norm_residual[echo_search_start..echo_search_end]
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0_f32, f32::max);

        assert!(
            echo_peak > 0.3,
            "echo should be visible after gain-compensated subtraction: {echo_peak:.4}"
        );
    }

    #[test]
    fn multiple_calibrations_reduce_noise() {
        let probe = make_probe();
        let phat_weight = 0.7;

        // Two calibrations with slight additive noise
        let mut cal1 = synthetic_recording(&probe, 1.0, &[]);
        let mut cal2 = synthetic_recording(&probe, 1.0, &[]);
        for (i, v) in cal1.iter_mut().enumerate() {
            *v += 0.02 * ((i as f32 * 1.3).sin());
        }
        for (i, v) in cal2.iter_mut().enumerate() {
            *v -= 0.02 * ((i as f32 * 1.3).sin());
        }

        let profile =
            CouplingProfile::from_calibrations(&probe, &[&cal1, &cal2], SAMPLE_RATE, phat_weight);

        // Clean measurement with echo
        let recording = synthetic_recording(&probe, 1.0, &[(0.50, 0.25)]);
        let meas_corr = gcc_phat(&recording, &probe, phat_weight);
        let residual = profile.subtract_correlation(&meas_corr, None);

        let echo_delay = distance_m_to_delay_samples(0.50, SAMPLE_RATE);
        let echo_search_start = echo_delay.saturating_sub(10);
        let echo_search_end = (echo_delay + 10).min(residual.len());
        let echo_peak = residual[echo_search_start..echo_search_end]
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0_f32, f32::max);

        assert!(echo_peak > 0.0, "echo should be detectable: {echo_peak:.6}");
    }

    #[test]
    fn detects_close_echo_after_subtraction() {
        let probe = make_probe();
        let phat_weight = 0.7;
        let echo_distance = 0.15; // 15 cm — very close

        let calibration = synthetic_recording(&probe, 1.0, &[]);
        let profile =
            CouplingProfile::from_calibrations(&probe, &[&calibration], SAMPLE_RATE, phat_weight);

        let recording = synthetic_recording(&probe, 1.0, &[(echo_distance, 0.4)]);
        let meas_corr = gcc_phat(&recording, &probe, phat_weight);
        let residual = profile.subtract_correlation(&meas_corr, None);

        let mut norm = residual.clone();
        normalize_abs(&mut norm);

        let min_spacing = (SAMPLE_RATE as usize / 2000).max(1);
        let peaks = find_peaks(&norm, 1, 0.2, min_spacing);

        let expected_delay = distance_m_to_delay_samples(echo_distance, SAMPLE_RATE);
        let best_peak = peaks
            .iter()
            .min_by_key(|p| (p.index as isize - expected_delay as isize).unsigned_abs());

        let best_peak = best_peak.expect("should find at least one peak in residual");
        let measured = delay_samples_to_distance_m(best_peak.index, SAMPLE_RATE);

        assert!(
            (measured - echo_distance).abs() < 0.10,
            "measured {measured:.3} m, expected {echo_distance} m"
        );
    }
}

// vim: set ts=4 sw=4 et:
