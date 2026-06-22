// Copyright (c) 2026 Elias S. G. Carotti

use crate::correlate::{gcc_phat_windowed, matched_filter_windowed};
use crate::coupling::CouplingProfile;
use crate::golay::{bpsk_modulate_chips, generate_golay_pair};
use crate::peak::{find_peaks, parabolic_peak_offset};
use crate::window::Window;

/// Cross-correlation method used by the echo detector.
#[derive(Debug, Clone, Copy)]
pub enum Correlator {
    MatchedFilter,
    GccPhat {
        weight: f32,
    },
    /// Golay complementary pair correlation.
    ///
    /// Transmits two sequences (A and B) separately and sums their
    /// correlations. Sidelobes cancel perfectly, leaving only true echo peaks.
    ///
    /// `order`: the pair length is `2^order` chips.
    /// `carrier_hz`: BPSK carrier frequency for the modulated probe.
    /// `chip_samples`: samples per chip.
    Golay {
        order: usize,
        carrier_hz: f32,
        chip_samples: usize,
    },
}

/// Input recording for the echo detector.
///
/// Must match the configured `Correlator`:
/// - `Single` for `MatchedFilter` or `GccPhat`
/// - `Pair` for `Golay`
///
/// Mismatches panic.
pub enum Recording<'a> {
    Single(&'a [f32]),
    Pair(&'a [f32], &'a [f32]),
}

/// Configuration for echo detection.
#[derive(Debug, Clone)]
pub struct EchoDetector {
    sample_rate: f32,
    correlator: Correlator,
    window: Window,
    min_distance_m: f32,
    max_distance_m: f32,
    noise_tail_fraction: f32,
    threshold_snr: f32,
    coupling: Option<CouplingProfile>,
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
            correlator: Correlator::GccPhat { weight: 0.7 },
            window: Window::None,
            min_distance_m: 0.20,
            max_distance_m: 5.0,
            noise_tail_fraction: 0.25,
            threshold_snr: 4.0,
            coupling: None,
        }
    }

    pub fn correlator(mut self, correlator: Correlator) -> Self {
        self.correlator = correlator;
        self
    }

    /// Spectral window applied to the cross-spectrum before the IFFT.
    ///
    /// Suppresses sidelobes in the correlation output at the cost of slightly
    /// wider main-lobe peaks. `Window::None` (default) leaves the spectrum
    /// untouched.
    pub fn window(mut self, window: Window) -> Self {
        self.window = window;
        self
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

    pub fn coupling(mut self, profile: CouplingProfile) -> Self {
        self.coupling = Some(profile);
        self
    }

    /// Detect echoes in a recording.
    ///
    /// Runs the full pipeline: correlate → coupling subtraction → peak
    /// detection. Returns all detected echoes sorted by distance (nearest
    /// first).
    ///
    /// # Panics
    ///
    /// Panics if the `Recording` variant doesn't match the configured
    /// `Correlator` (e.g. `Single` with `Golay`, or `Pair` with `GccPhat`).
    pub fn detect(&self, recording: Recording, probe: &[f32]) -> Vec<Echo> {
        let correlation = match self.correlator {
            Correlator::MatchedFilter => {
                let signal = expect_single(recording, "MatchedFilter");
                matched_filter_windowed(signal, probe, self.window)
            }
            Correlator::GccPhat { weight } => {
                let signal = expect_single(recording, "GccPhat");
                gcc_phat_windowed(signal, probe, weight, self.window)
            }
            Correlator::Golay {
                order,
                carrier_hz,
                chip_samples,
            } => {
                let (rec_a, rec_b) = expect_pair(recording, "Golay");
                self.golay_correlate(rec_a, rec_b, order, carrier_hz, chip_samples)
            }
        };

        self.detect_in_correlation(&correlation)
    }

    fn golay_correlate(
        &self,
        recording_a: &[f32],
        recording_b: &[f32],
        order: usize,
        carrier_hz: f32,
        chip_samples: usize,
    ) -> Vec<f32> {
        let (chips_a, chips_b) = generate_golay_pair(order);
        let probe_a = bpsk_modulate_chips(&chips_a, self.sample_rate, carrier_hz, chip_samples);
        let probe_b = bpsk_modulate_chips(&chips_b, self.sample_rate, carrier_hz, chip_samples);

        let corr_a = gcc_phat_windowed(recording_a, &probe_a, 0.0, self.window);
        let corr_b = gcc_phat_windowed(recording_b, &probe_b, 0.0, self.window);

        let len = corr_a.len().max(corr_b.len());
        let mut combined = vec![0.0; len];

        for (i, v) in combined.iter_mut().enumerate() {
            let a = corr_a.get(i).copied().unwrap_or(0.0);
            let b = corr_b.get(i).copied().unwrap_or(0.0);
            *v = a + b;
        }

        combined
    }

    /// Detect echoes in a pre-computed correlation signal.
    ///
    /// Use this if you need access to the raw correlation yourself, or if you
    /// computed it with custom parameters.
    pub fn detect_in_correlation(&self, correlation: &[f32]) -> Vec<Echo> {
        if correlation.len() < 3 {
            return Vec::new();
        }

        let abs_orig: Vec<f32> = correlation.iter().map(|x| x.abs()).collect();
        let direct_idx = self.find_direct_peak(&abs_orig);
        let direct_peak = abs_orig[direct_idx];

        let working = match &self.coupling {
            Some(profile) => profile.subtract_correlation(correlation, None),
            None => correlation.to_vec(),
        };

        let abs_corr: Vec<f32> = working.iter().map(|x| x.abs()).collect();

        if direct_peak <= 0.0 {
            return Vec::new();
        }

        let noise_floor = self.estimate_noise_floor(&abs_corr, direct_idx);

        let noise_threshold = noise_floor * self.threshold_snr;
        let sidelobe_threshold = direct_peak * 0.15;
        let threshold = noise_threshold.max(sidelobe_threshold);

        let min_lag = direct_idx + distance_m_to_lag(self.min_distance_m, self.sample_rate);
        let max_lag = direct_idx + distance_m_to_lag(self.max_distance_m, self.sample_rate);
        let max_lag = max_lag.min(abs_corr.len().saturating_sub(2));

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
        let tail_len = (abs_corr.len() as f32 * self.noise_tail_fraction).round() as usize;
        let tail_start = abs_corr.len().saturating_sub(tail_len);

        let safe_start = direct_idx + distance_m_to_lag(self.max_distance_m, self.sample_rate);
        let start = tail_start.max(safe_start).min(abs_corr.len());

        if start >= abs_corr.len() {
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

fn expect_single<'a>(recording: Recording<'a>, correlator_name: &str) -> &'a [f32] {
    match recording {
        Recording::Single(s) => s,
        Recording::Pair(_, _) => {
            panic!("Correlator::{correlator_name} requires Recording::Single, got Pair")
        }
    }
}

fn expect_pair<'a>(recording: Recording<'a>, correlator_name: &str) -> (&'a [f32], &'a [f32]) {
    match recording {
        Recording::Pair(a, b) => (a, b),
        Recording::Single(_) => {
            panic!("Correlator::{correlator_name} requires Recording::Pair, got Single")
        }
    }
}

fn distance_m_to_lag(distance_m: f32, sample_rate: f32) -> usize {
    let round_trip_s = 2.0 * distance_m / crate::SPEED_OF_SOUND_M_S;
    (round_trip_s * sample_rate).round() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::golay::{bpsk_modulate_chips, generate_golay_pair};
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

        let detector = EchoDetector::new(SAMPLE_RATE)
            .correlator(Correlator::MatchedFilter)
            .max_distance_m(4.0);
        let echoes = detector.detect(Recording::Single(&rx), &probe);

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

        let detector = EchoDetector::new(SAMPLE_RATE)
            .correlator(Correlator::GccPhat { weight: 0.7 })
            .max_distance_m(4.0);
        let echoes = detector.detect(Recording::Single(&rx), &probe);

        assert!(!echoes.is_empty(), "should detect at least one echo");
        assert!(
            (echoes[0].distance_m - echo_distance).abs() < 0.05,
            "measured {:.3} m, expected {echo_distance} m",
            echoes[0].distance_m
        );
    }

    #[test]
    fn default_correlator_is_gcc_phat() {
        let probe = make_probe();
        let echo_distance = 1.5;
        let rx = synthetic_recording(&probe, 1.0, &[(echo_distance, 0.3)]);

        let detector = EchoDetector::new(SAMPLE_RATE).max_distance_m(4.0);
        let echoes = detector.detect(Recording::Single(&rx), &probe);

        assert!(!echoes.is_empty(), "default should detect echo");
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

        let detector = EchoDetector::new(SAMPLE_RATE)
            .correlator(Correlator::MatchedFilter)
            .min_distance_m(1.0)
            .max_distance_m(4.0);
        let echoes = detector.detect(Recording::Single(&rx), &probe);

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

        let detector = EchoDetector::new(SAMPLE_RATE)
            .correlator(Correlator::MatchedFilter)
            .max_distance_m(3.0);
        let echoes = detector.detect(Recording::Single(&rx), &probe);

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

        let detector = EchoDetector::new(SAMPLE_RATE)
            .correlator(Correlator::MatchedFilter)
            .max_distance_m(4.0);
        let echoes = detector.detect(Recording::Single(&rx), &probe);

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

        let detector = EchoDetector::new(SAMPLE_RATE)
            .correlator(Correlator::MatchedFilter)
            .max_distance_m(4.0)
            .threshold_snr(3.0);
        let echoes = detector.detect(Recording::Single(&rx), &probe);

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
        let rx = synthetic_recording(&probe, 1.0, &[]);

        let detector = EchoDetector::new(SAMPLE_RATE)
            .correlator(Correlator::MatchedFilter)
            .max_distance_m(4.0);
        let echoes = detector.detect(Recording::Single(&rx), &probe);

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

        let detector = EchoDetector::new(SAMPLE_RATE)
            .correlator(Correlator::MatchedFilter)
            .max_distance_m(4.0);
        let echoes = detector.detect(Recording::Single(&rx), &probe);

        assert!(!echoes.is_empty());
        assert!(echoes[0].offset.abs() <= 0.5);
    }

    #[test]
    fn coupling_subtraction_reveals_close_echo() {
        let probe = make_probe();
        let phat_weight = 0.7;
        let echo_distance = 0.30;

        let calibration = synthetic_recording(&probe, 1.0, &[]);
        let profile =
            CouplingProfile::from_calibrations(&probe, &[&calibration], SAMPLE_RATE, phat_weight);

        let recording = synthetic_recording(&probe, 1.0, &[(echo_distance, 0.4)]);

        let detector = EchoDetector::new(SAMPLE_RATE)
            .correlator(Correlator::GccPhat {
                weight: phat_weight,
            })
            .min_distance_m(0.10)
            .max_distance_m(2.0)
            .coupling(profile);
        let echoes = detector.detect(Recording::Single(&recording), &probe);

        assert!(
            !echoes.is_empty(),
            "should detect close echo with coupling subtraction"
        );
        assert!(
            (echoes[0].distance_m - echo_distance).abs() < 0.10,
            "measured {:.3} m, expected {echo_distance} m",
            echoes[0].distance_m
        );
    }

    #[test]
    fn golay_detects_echo() {
        let order = 6;
        let carrier_hz = 18_000.0;
        let chip_samples = 4;

        let (chips_a, chips_b) = generate_golay_pair(order);
        let probe_a = bpsk_modulate_chips(&chips_a, SAMPLE_RATE, carrier_hz, chip_samples);
        let probe_b = bpsk_modulate_chips(&chips_b, SAMPLE_RATE, carrier_hz, chip_samples);

        let echo_distance = 1.5;
        let rx_a = synthetic_recording(&probe_a, 1.0, &[(echo_distance, 0.3)]);
        let rx_b = synthetic_recording(&probe_b, 1.0, &[(echo_distance, 0.3)]);

        let detector = EchoDetector::new(SAMPLE_RATE)
            .correlator(Correlator::Golay {
                order,
                carrier_hz,
                chip_samples,
            })
            .max_distance_m(4.0);

        // probe argument is unused for Golay (probes are generated internally)
        let echoes = detector.detect(Recording::Pair(&rx_a, &rx_b), &[]);

        assert!(!echoes.is_empty(), "Golay should detect echo");
        assert!(
            (echoes[0].distance_m - echo_distance).abs() < 0.10,
            "Golay measured {:.3} m, expected {echo_distance} m",
            echoes[0].distance_m
        );
    }

    #[test]
    fn golay_cancels_sidelobes() {
        let order = 6;
        let carrier_hz = 18_000.0;
        let chip_samples = 4;

        let (chips_a, chips_b) = generate_golay_pair(order);
        let probe_a = bpsk_modulate_chips(&chips_a, SAMPLE_RATE, carrier_hz, chip_samples);
        let probe_b = bpsk_modulate_chips(&chips_b, SAMPLE_RATE, carrier_hz, chip_samples);

        // Direct path only — no echoes
        let rx_a = synthetic_recording(&probe_a, 1.0, &[]);
        let rx_b = synthetic_recording(&probe_b, 1.0, &[]);

        let detector = EchoDetector::new(SAMPLE_RATE)
            .correlator(Correlator::Golay {
                order,
                carrier_hz,
                chip_samples,
            })
            .max_distance_m(4.0);

        let echoes = detector.detect(Recording::Pair(&rx_a, &rx_b), &[]);

        assert!(
            echoes.is_empty(),
            "Golay should produce no false echoes from sidelobes, got {}",
            echoes.len()
        );
    }

    #[test]
    #[should_panic(expected = "requires Recording::Single")]
    fn panics_on_pair_with_matched_filter() {
        let probe = make_probe();
        let rx = synthetic_recording(&probe, 1.0, &[]);

        let detector = EchoDetector::new(SAMPLE_RATE).correlator(Correlator::MatchedFilter);
        detector.detect(Recording::Pair(&rx, &rx), &probe);
    }

    #[test]
    #[should_panic(expected = "requires Recording::Pair")]
    fn panics_on_single_with_golay() {
        let probe = make_probe();
        let rx = synthetic_recording(&probe, 1.0, &[]);

        let detector = EchoDetector::new(SAMPLE_RATE).correlator(Correlator::Golay {
            order: 4,
            carrier_hz: 18_000.0,
            chip_samples: 4,
        });
        detector.detect(Recording::Single(&rx), &probe);
    }
}

// vim: set ts=4 sw=4 et:
