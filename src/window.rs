// Copyright (c) 2026 Elias S. G. Carotti

use std::f32::consts::TAU;

#[derive(Debug, Clone, Copy, Default)]
pub enum Window {
    #[default]
    None,
    Hann,
    Hamming,
    Blackman,
    Kaiser {
        beta: f32,
    },
}

pub fn window_value(window: Window, n: usize, len: usize) -> f32 {
    if len <= 1 {
        return 1.0;
    }

    match window {
        Window::None => 1.0,

        Window::Hann => 0.5 - 0.5 * (TAU * n as f32 / (len - 1) as f32).cos(),

        Window::Hamming => 0.54 - 0.46 * (TAU * n as f32 / (len - 1) as f32).cos(),

        Window::Blackman => {
            let x = TAU * n as f32 / (len - 1) as f32;

            0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos()
        }

        Window::Kaiser { beta } => {
            let x = 2.0 * n as f32 / (len - 1) as f32 - 1.0;
            let arg = beta * (1.0 - x * x).max(0.0).sqrt();

            bessel_i0(arg) / bessel_i0(beta)
        }
    }
}

pub fn apply_window(samples: &mut [f32], window: Window) {
    let len = samples.len();

    for (n, sample) in samples.iter_mut().enumerate() {
        *sample *= window_value(window, n, len);
    }
}

fn bessel_i0(x: f32) -> f32 {
    // Numerical Recipes-style approximation for modified Bessel I0.
    let ax = x.abs();

    if ax < 3.75 {
        let y = (x / 3.75) * (x / 3.75);

        1.0 + y
            * (3.515_622_9
                + y * (3.089_942_4
                    + y * (1.206_749_2 + y * (0.265_973_2 + y * (0.036_076_8 + y * 0.004_581_3)))))
    } else {
        let y = 3.75 / ax;

        (ax.exp() / ax.sqrt())
            * (0.398_942_28
                + y * (0.013_285_92
                    + y * (0.002_253_19
                        + y * (-0.001_575_65
                            + y * (0.009_162_81
                                + y * (-0.020_577_06
                                    + y * (0.026_355_37
                                        + y * (-0.016_476_33 + y * 0.003_923_77))))))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_edges_are_finite() {
        for window in [
            Window::None,
            Window::Hann,
            Window::Hamming,
            Window::Blackman,
            Window::Kaiser { beta: 6.0 },
        ] {
            for n in 0..16 {
                assert!(window_value(window, n, 16).is_finite());
            }
        }
    }

    #[test]
    fn kaiser_is_symmetric() {
        let window = Window::Kaiser { beta: 6.0 };

        for n in 0..32 {
            let a = window_value(window, n, 32);
            let b = window_value(window, 31 - n, 32);

            assert!((a - b).abs() < 1e-6);
        }
    }
}

//  vim: set ts=4 sw=4 et:
