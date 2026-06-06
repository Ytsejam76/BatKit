// Copyright (c) 2026 Elias S. G. Carotti
use crate::SPEED_OF_SOUND_M_S;

/// Convert a round-trip delay in samples to one-way distance in meters.
pub fn delay_samples_to_distance_m(delay_samples: usize, sample_rate: f32) -> f32 {
    assert!(sample_rate > 0.0);

    let delay_s = delay_samples as f32 / sample_rate;
    SPEED_OF_SOUND_M_S * delay_s / 2.0
}

/// Convert one-way distance in meters to round-trip delay in samples.
pub fn distance_m_to_delay_samples(distance_m: f32, sample_rate: f32) -> usize {
    assert!(distance_m >= 0.0);
    assert!(sample_rate > 0.0);

    let round_trip_s = 2.0 * distance_m / SPEED_OF_SOUND_M_S;
    (round_trip_s * sample_rate).round() as usize
}

/// Convert a delay difference between two microphones to a rough bearing.
///
/// Returns an angle in radians relative to broadside/front, using the simple
/// far-field approximation `sin(theta) = c * dt / baseline`.
pub fn tdoa_to_bearing_rad(delta_t_s: f32, mic_spacing_m: f32) -> Option<f32> {
    assert!(mic_spacing_m > 0.0);

    let x = SPEED_OF_SOUND_M_S * delta_t_s / mic_spacing_m;

    if x.abs() > 1.0 {
        None
    } else {
        Some(x.asin())
    }
}

// vim: set ts=4 sw=4 et:
