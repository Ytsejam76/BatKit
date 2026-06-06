// Copyright (c) 2026 Elias S. G. Carotti

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Peak {
    pub index: usize,
    pub value: f32,
}

/// Find simple positive local maxima.
///
/// `min_spacing` suppresses nearby duplicate peaks by keeping only the stronger
/// peak within each spacing window.
pub fn find_peaks(
    samples: &[f32],
    min_index: usize,
    min_value: f32,
    min_spacing: usize,
) -> Vec<Peak> {
    let mut peaks: Vec<Peak> = Vec::new();

    if samples.len() < 3 {
        return peaks;
    }

    let mut last_peak_index: Option<usize> = None;

    for i in min_index.max(1)..samples.len() - 1 {
        let y = samples[i];

        if y < min_value {
            continue;
        }

        if y <= samples[i - 1] || y <= samples[i + 1] {
            continue;
        }

        if let Some(prev_i) = last_peak_index {
            if i - prev_i < min_spacing {
                let replace_previous = peaks
                    .last()
                    .map(|previous| y > previous.value)
                    .unwrap_or(true);

                if replace_previous {
                    peaks.pop();
                    peaks.push(Peak { index: i, value: y });
                    last_peak_index = Some(i);
                }

                continue;
            }
        }

        peaks.push(Peak { index: i, value: y });
        last_peak_index = Some(i);
    }

    peaks
}

/// Refine a discrete peak location using parabolic interpolation.
///
/// Returns an offset relative to the center sample, typically in `[-0.5, 0.5]`.
pub fn parabolic_peak_offset(y0: f32, y1: f32, y2: f32) -> f32 {
    let denom = y0 - 2.0 * y1 + y2;

    if denom.abs() < 1e-12 {
        return 0.0;
    }

    0.5 * (y0 - y2) / denom
}

// vim: set ts=4 sw=4 et:
