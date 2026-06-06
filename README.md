# BatKit
***Tools for things that go chirp in the night.***

[![Docs](https://github.com/Ytsejam76/BatKit/actions/workflows/docs.yml/badge.svg)](https://Ytsejam76.github.io/BatKit/batkit/index.html)

`BatKit` is a small pure-Rust DSP crate for acoustic ranging toy projects.

As of now the project is still in development, and it's mostly for fun, so no expectations at all.

## Current scope

- linear chirp generation
- Hann windowing
- peak normalization
- direct matched filtering
- autocorrelation
- simple peak detection
- distance/delay conversion
- Golay complementary pair generation

The first implementation favors clarity over speed. FFT convolution can be added
later without changing the high-level API too much.

## Example

```rust
use batkit::{
    correlate::matched_filter,
    peak::find_peaks,
    probe::{apply_hann_window, linear_chirp, normalize_peak},
    range::{delay_samples_to_distance_m, distance_m_to_delay_samples},
};

let sample_rate = 48_000.0;

let mut probe = linear_chirp(sample_rate, 0.020, 16_000.0, 22_000.0);
apply_hann_window(&mut probe);
normalize_peak(&mut probe);

let echo_distance_m = 1.25;
let echo_delay = distance_m_to_delay_samples(echo_distance_m, sample_rate);

let mut rx = vec![0.0; echo_delay + probe.len() + 1024];

for (i, &x) in probe.iter().enumerate() {
    rx[i] += x;                     // direct speaker-to-mic leakage
    rx[echo_delay + i] += 0.25 * x;  // synthetic echo
}

let corr = matched_filter(&rx, &probe);
let peaks = find_peaks(&corr, 0, 1.0, 64);

for peak in peaks {
    let distance_m = delay_samples_to_distance_m(peak.index, sample_rate);
    println!("peak at {:.2} m: {:.3}", distance_m, peak.value);
}
```

Run the bundled example:

```sh
cargo run --example synthetic_echo
```

Run tests:

```sh
cargo test
```

## Roadmap

Likely next modules:

- FFT-based matched filtering
- random-phase multitone probes
- band-limited pseudo-noise probes
- GCC-PHAT / TDOA helpers
- track smoothing and false-positive rejection
- WAV import/export examples
