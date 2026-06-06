// Copyright (c) 2026 Elias S. G. Carotti
use batkit::{
    correlate::matched_filter,
    peak::find_peaks,
    probe::{apply_hann_window, linear_chirp, normalize_peak},
    range::{delay_samples_to_distance_m, distance_m_to_delay_samples},
};

fn main() {
    let sample_rate = 48_000.0;

    let mut probe = linear_chirp(sample_rate, 0.020, 16_000.0, 22_000.0);
    apply_hann_window(&mut probe);
    normalize_peak(&mut probe);

    let echo_distance_m = 1.25;
    let echo_delay = distance_m_to_delay_samples(echo_distance_m, sample_rate);

    let mut rx = vec![0.0; echo_delay + probe.len() + 1024];

    for (i, &x) in probe.iter().enumerate() {
        rx[i] += x;
        rx[echo_delay + i] += 0.25 * x;
    }

    let corr = matched_filter(&rx, &probe);
    let peaks = find_peaks(&corr, 0, 1.0, 64);

    println!("detected peaks:");

    for peak in peaks {
        let distance_m = delay_samples_to_distance_m(peak.index, sample_rate);
        println!(
            "  sample {:5} | {:5.2} m | value {:.3}",
            peak.index, distance_m, peak.value
        );
    }
}
