// Copyright (c) 2026 Elias S. G. Carotti

use std::f32::consts::TAU;

use crate::window::{window_value, Window};

#[derive(Debug, Clone)]
pub struct Probe {
    pub sample_rate: f32,
    pub samples: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct ProbeBuilder {
    sample_rate: f32,
    samples: Vec<f32>,
}

impl ProbeBuilder {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            samples: Vec::new(),
        }
    }

    pub fn silence(mut self, duration_s: f32) -> Self {
        let len = self.samples_for(duration_s);
        self.samples.extend(std::iter::repeat_n(0.0, len));
        self
    }

    /// Append a linear chirp segment.
    ///
    /// `start_hz` and `end_hz` define the frequency sweep.
    /// `gain` scales this segment before it is appended.
    /// `window` is applied only to this segment, not to the whole probe.
    pub fn chirp(
        mut self,
        duration_s: f32,
        start_hz: f32,
        end_hz: f32,
        gain: f32,
        window: Window,
    ) -> Self {
        let len = self.samples_for(duration_s);
        let k = (end_hz - start_hz) / duration_s;

        for n in 0..len {
            let t = n as f32 / self.sample_rate;
            let phase = TAU * (start_hz * t + 0.5 * k * t * t);
            let w = window_value(window, n, len);

            self.samples.push(gain * w * phase.sin());
        }

        self
    }

    /// Append an audible decaying sine-burst segment.
    ///
    /// This produces a simple "submarine ping"-style tone: a sine wave at
    /// `freq_hz` multiplied by an exponential decay envelope.
    ///
    /// This is mostly useful for audible debugging/demo purposes. Since it is
    /// narrowband, it has poor timing resolution compared with a chirp or coded
    /// broadband probe.
    ///
    /// `decay` controls how quickly the tone fades out. Larger values decay
    /// faster.
    pub fn sine_ping(
        mut self,
        duration_s: f32,
        freq_hz: f32,
        decay: f32,
        gain: f32,
        window: Window,
    ) -> Self {
        let len = self.samples_for(duration_s);

        for n in 0..len {
            let t = n as f32 / self.sample_rate;
            let envelope = (-decay * t / duration_s).exp();
            let w = window_value(window, n, len);
            let phase = TAU * freq_hz * t;

            self.samples.push(gain * w * envelope * phase.sin());
        }

        self
    }

    /// Normalize the whole accumulated probe by peak absolute amplitude.
    pub fn normalize(mut self) -> Self {
        normalize_peak(&mut self.samples);
        self
    }

    pub fn build(self) -> Probe {
        Probe {
            sample_rate: self.sample_rate,
            samples: self.samples,
        }
    }

    fn samples_for(&self, duration_s: f32) -> usize {
        (self.sample_rate * duration_s).round() as usize
    }
}

impl Probe {
    pub fn builder(sample_rate: f32) -> ProbeBuilder {
        ProbeBuilder::new(sample_rate)
    }
}

/// Generate a linear frequency sweep.
pub fn linear_chirp(sample_rate: f32, duration_s: f32, start_hz: f32, end_hz: f32) -> Vec<f32> {
    ProbeBuilder::new(sample_rate)
        .chirp(duration_s, start_hz, end_hz, 1.0, Window::None)
        .build()
        .samples
}

/// Generate an audible decaying sine-burst probe.
///
/// This produces a simple "submarine ping"-style tone: a sine wave at
/// `freq_hz` multiplied by an exponential decay envelope.
///
/// This is mostly useful for audible debugging/demo purposes. Since it is
/// narrowband, it has poor timing resolution compared with a chirp or coded
/// broadband probe.
///
/// `decay` controls how quickly the tone fades out. Larger values decay faster.
pub fn sine_ping(sample_rate: f32, duration_s: f32, freq_hz: f32, decay: f32) -> Vec<f32> {
    ProbeBuilder::new(sample_rate)
        .sine_ping(duration_s, freq_hz, decay, 1.0, Window::None)
        .build()
        .samples
}

pub fn apply_hann_window(samples: &mut [f32]) {
    crate::window::apply_window(samples, Window::Hann);
}

pub fn normalize_peak(samples: &mut [f32]) {
    let peak = samples.iter().copied().map(f32::abs).fold(0.0, f32::max);

    if peak > 0.0 {
        for x in samples {
            *x /= peak;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_composed_probe() {
        let sample_rate = 48_000.0_f32;

        let probe = ProbeBuilder::new(sample_rate)
            .sine_ping(0.120, 1_200.0, 6.0, 0.20, Window::None)
            .silence(0.040)
            .chirp(0.020, 16_000.0, 22_000.0, 1.0, Window::Hann)
            .normalize()
            .build();

        let expected_len = ((0.120 + 0.040 + 0.020) * sample_rate).round() as usize;

        assert_eq!(probe.samples.len(), expected_len);
        assert_eq!(probe.sample_rate, sample_rate);

        let peak = probe
            .samples
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0, f32::max);

        assert!((peak - 1.0).abs() < 1e-6);
    }

    #[test]
    fn linear_chirp_keeps_legacy_api() {
        let chirp = linear_chirp(48_000.0, 0.020, 16_000.0, 22_000.0);
        assert_eq!(chirp.len(), 960);
    }

    #[test]
    fn sine_ping_keeps_legacy_api() {
        let ping = sine_ping(48_000.0, 0.120, 1_200.0, 6.0);
        assert_eq!(ping.len(), 5760);
    }
}

// vim: set ts=4 sw=4 et:
