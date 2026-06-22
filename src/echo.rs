// Copyright (c) 2026 Elias S. G. Carotti

use crate::peak::{find_peaks, parabolic_peak_offset};

/// Configuration for echo detection on a correlation signal.
#[derive(Debug, Clone)]
pub struct EchoDetector {
    sample_rate: f32,
    min_distance_m: f32,
    max_distance_m: f32,
    /// Fraction of the correlation tail used to estimate noise floor (0.0–1.0).
    noise_tail_fraction: f32,
    /// Detection threshold as a multiplier over the estimated noise floor.
    threshold_snr: f32,
}

/// A detected echo with sub-sample precision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Echo {
    /// Discrete sample index within the correlation.
    pub index: usize,
    /// Sub-sample offset from parabolic interpolation, in [-0.5, 0.5].
    pub offset: f32,
    /// Estimated one-way distance in meters.
    pub distance_m: f32,
    /// Peak strength relative to the noise floor.
    pub snr: f32,
}

impl EchoDetector {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            min_distance_m: 0.20,
            max_distance_m: 5.0,
            noise_tail_fraction: 0.25,
            threshold_snr: 4.0,
        }
    }

    pub fn min_distance_m(mut self, meters: f32) -> Self {
        self.min_distance_m = meters;
        self
    }

    pub fn max_distance_m(mut self, meters: f32) -> Self {
        self.max_distance_m = meters;
        self
    }

    pub fn noise_tail_fraction(mut self, fraction: f32) -> Self {
        self.noise_tail_fraction = fraction.clamp(0.05, 0.9);
        self
    }

    pub fn threshold_snr(mut self, snr: f32) -> Self {
        self.threshold_snr = snr;
        self
    }

    /// Detect echoes in a correlation signal.
    ///
    /// `correlation` is the output of `matched_filter`, `gcc_phat`, or a
    /// coupling-subtracted residual. The direct-path peak (at/near lag 0) is
    /// used as a reference — echoes are searched between `min_distance_m` and
    /// `max_distance_m` from it.
    ///
    /// Returns all detected echoes sorted by distance (nearest first).
    pub fn detect(&self, correlation: &[f32]) -> Vec<Echo> {
        if correlation.len() < 3 {
            return Vec::new();
        }

        let abs_corr: Vec<f32> = correlation.iter().map(|x| x.abs()).collect();

        let direct_idx = self.find_direct_peak(&abs_corr);
        let direct_peak = abs_corr[direct_idx];

        if direct_peak <= 0.0 {
            return Vec::new();
        }

        let noise_floor = self.estimate_noise_floor(&abs_corr, direct_idx);

        // Threshold: max of (noise_floor * snr_threshold) and a fraction of
        // the direct peak. The direct-peak fraction prevents autocorrelation
        // sidelobes from triggering false detections.
        let noise_threshold = noise_floor * self.threshold_snr;
        let sidelobe_threshold = direct_peak * 0.15;
        let threshold = noise_threshold.max(sidelobe_threshold);

        let min_lag = direct_idx + distance_m_to_lag(self.min_distance_m, self.sample_rate);
        let max_lag = direct_idx + distance_m_to_lag(self.max_distance_m, self.sample_rate);
        let max_lag = max_lag.min(correlation.len().saturating_sub(2));

        if min_lag >= max_lag {
            return Vec::new();
        }

        let min_spacing = (self.sample_rate as usize / 2_000).max(1);
        let peaks = find_peaks(&abs_corr, min_lag, threshold, min_spacing);

        peaks
            .iter()
            .filter(|p| p.index <= max_lag)
            .map(|p| {
                let offset = if p.index > 0 && p.index < abs_corr.len() - 1 {
                    parabolic_peak_offset(
                        abs_corr[p.index - 1],
                        abs_corr[p.index],
                        abs_corr[p.index + 1],
                    )
                } else {
                    0.0
                };

                let delay_from_direct = p.index - direct_idx;
                let precise_delay = delay_from_direct as f32 + offset;
                let distance_m = precise_delay / self.sample_rate * crate::SPEED_OF_SOUND_M_S / 2.0;

                let snr = if noise_floor > 0.0 {
                    p.value / noise_floor
                } else {
                    f32::INFINITY
                };

                Echo {
                    index: p.index,
                    offset,
                    distance_m,
                    snr,
                }
            })
            .collect()
    }

    fn estimate_noise_floor(&self, abs_corr: &[f32], direct_idx: usize) -> f32 {
        // Estimate from the tail, excluding the region around the direct peak
        let tail_len = (abs_corr.len() as f32 * self.noise_tail_fraction).round() as usize;
        let tail_start = abs_corr.len().saturating_sub(tail_len);

        // Also exclude a region after max_distance from direct to avoid
        // counting far echoes as noise
        let safe_start = direct_idx + distance_m_to_lag(self.max_distance_m, self.sample_rate);
        let start = tail_start.max(safe_start).min(abs_corr.len());

        if start >= abs_corr.len() {
            // Fallback: use the last 10% regardless
            let fallback_start = abs_corr.len().saturating_sub(abs_corr.len() / 10);
            let tail = &abs_corr[fallback_start..];
            if tail.is_empty() {
                return 0.0;
            }
            let sq_sum: f32 = tail.iter().map(|x| x * x).sum();
            return (sq_sum / tail.len() as f32).sqrt();
        }

        let tail = &abs_corr[start..];
        if tail.is_empty() {
            return 0.0;
        }

        let sq_sum: f32 = tail.iter().map(|x| x * x).sum();
        (sq_sum / tail.len() as f32).sqrt()
    }

    fn find_direct_peak(&self, abs_corr: &[f32]) -> usize {
        // Direct peak should be within the first few ms
        let search_window = (self.sample_rate as usize / 100).min(abs_corr.len());

        abs_corr
            .iter()
            .take(search_window)
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

fn distance_m_to_lag(distance_m: f32, sample_rate: f32) -> usize {
    let round_trip_s = 2.0 * distance_m / crate::SPEED_OF_SOUND_M_S;
    (round_trip_s * sample_rate).round() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::correlate::{gcc_phat, matched_filter, normalize_abs};
    use crate::probe::{apply_hann_window, linear_chirp, normalize_peak};
    use crate::range::distance_m_to_delay_samples;

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
        let mut rx = vec![0.0; probe.len() + max_delay + 2048];

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
    fn detects_single_echo_matched_filter() {
        let probe = make_probe();
        let echo_distance = 1.5;
        let rx = synthetic_recording(&probe, 1.0, &[(echo_distance, 0.3)]);

        let corr = matched_filter(&rx, &probe);

        let detector = EchoDetector::new(SAMPLE_RATE).max_distance_m(4.0);
        let echoes = detector.detect(&corr);

        assert!(!echoes.is_empty(), "should detect at least one echo");
        assert!(
            (echoes[0].distance_m - echo_distance).abs() < 0.05,
            "measured {:.3} m, expected {echo_distance} m",
            echoes[0].distance_m
        );
    }

    #[test]
    fn detects_single_echo_gcc_phat() {
        let probe = make_probe();
        let echo_distance = 1.5;
        let rx = synthetic_recording(&probe, 1.0, &[(echo_distance, 0.3)]);

        let corr = gcc_phat(&rx, &probe, 0.7);

        let detector = EchoDetector::new(SAMPLE_RATE).max_distance_m(4.0);
        let echoes = detector.detect(&corr);

        assert!(!echoes.is_empty(), "should detect at least one echo");
        assert!(
            (echoes[0].distance_m - echo_distance).abs() < 0.05,
            "measured {:.3} m, expected {echo_distance} m",
            echoes[0].distance_m
        );
    }

    #[test]
    fn respects_min_distance() {
        let probe = make_probe();
        let rx = synthetic_recording(&probe, 1.0, &[(0.40, 0.5), (1.5, 0.3)]);

        let corr = matched_filter(&rx, &probe);

        let detector = EchoDetector::new(SAMPLE_RATE)
            .min_distance_m(1.0)
            .max_distance_m(4.0);
        let echoes = detector.detect(&corr);

        assert!(!echoes.is_empty());
        assert!(
            echoes[0].distance_m >= 0.95,
            "should skip echo below min_distance, got {:.3} m",
            echoes[0].distance_m
        );
    }

    #[test]
    fn respects_max_distance() {
        let probe = make_probe();
        let rx = synthetic_recording(&probe, 1.0, &[(1.0, 0.3), (8.0, 0.5)]);

        let corr = matched_filter(&rx, &probe);

        let detector = EchoDetector::new(SAMPLE_RATE).max_distance_m(3.0);
        let echoes = detector.detect(&corr);

        for echo in &echoes {
            assert!(
                echo.distance_m <= 3.5,
                "echo at {:.3} m should be rejected by max_distance",
                echo.distance_m
            );
        }
    }

    #[test]
    fn detects_multiple_echoes() {
        let probe = make_probe();
        let rx = synthetic_recording(&probe, 1.0, &[(0.80, 0.4), (2.0, 0.25)]);

        let corr = matched_filter(&rx, &probe);

        let detector = EchoDetector::new(SAMPLE_RATE).max_distance_m(4.0);
        let echoes = detector.detect(&corr);

        assert!(
            echoes.len() >= 2,
            "should detect both echoes, got {}",
            echoes.len()
        );
        assert!((echoes[0].distance_m - 0.80).abs() < 0.08);
        assert!((echoes[1].distance_m - 2.0).abs() < 0.08);
    }

    #[test]
    fn returns_snr_above_threshold() {
        let probe = make_probe();
        let rx = synthetic_recording(&probe, 1.0, &[(1.0, 0.3)]);

        let corr = matched_filter(&rx, &probe);

        let detector = EchoDetector::new(SAMPLE_RATE)
            .max_distance_m(4.0)
            .threshold_snr(3.0);
        let echoes = detector.detect(&corr);

        for echo in &echoes {
            assert!(
                echo.snr >= 3.0,
                "echo SNR {:.1} should be above threshold",
                echo.snr
            );
        }
    }

    #[test]
    fn no_false_echoes_in_silence() {
        let probe = make_probe();
        // Only direct path, no reflections, long tail
        let rx = synthetic_recording(&probe, 1.0, &[]);

        let corr = matched_filter(&rx, &probe);

        let detector = EchoDetector::new(SAMPLE_RATE).max_distance_m(4.0);
        let echoes = detector.detect(&corr);

        assert!(
            echoes.is_empty(),
            "should not detect echoes in clean direct-only signal, got {} echoes",
            echoes.len()
        );
    }

    #[test]
    fn sub_sample_refinement() {
        let probe = make_probe();
        let echo_distance = 1.0;
        let rx = synthetic_recording(&probe, 1.0, &[(echo_distance, 0.3)]);

        let corr = matched_filter(&rx, &probe);

        let detector = EchoDetector::new(SAMPLE_RATE).max_distance_m(4.0);
        let echoes = detector.detect(&corr);

        assert!(!echoes.is_empty());
        // With parabolic interpolation, offset should be non-zero in general
        // (it's exactly zero only if the peak happens to land on a sample)
        // Just verify it's in range
        assert!(echoes[0].offset.abs() <= 0.5);
    }
}

// vim: set ts=4 sw=4 et:
