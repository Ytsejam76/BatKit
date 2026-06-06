# batkit-cli

Small command-line utilities for testing `batkit` on a desktop 

## Commands

Generate a 20 ms ultrasonic-ish chirp probe:

```sh
cargo run -p batkit-cli -- probe \
  --out probe.wav \
  --sample-rate 48000 \
  --duration-ms 20 \
  --start-hz 16000 \
  --end-hz 22000
```

Play the probe and record the default microphone at the same time:

```sh
cargo run -p batkit-cli -- play-record \
  --play probe.wav \
  --out rx.wav \
  --record-ms 300
```

Analyze the recording:

```sh
cargo run -p batkit-cli -- analyze \
  --probe probe.wav \
  --recording rx.wav \
  --min-distance-m 0.30 \
  --max-distance-m 5.0 \
  --threshold 0.15
```

The analyzer treats the strongest matched-filter peak as the direct
speaker-to-mic path. Candidate echo distances are reported relative to that
peak.

## Notes

- `play-record` currently requests `f32` streams from the default CPAL input and
  output devices. Some systems may need sample-format negotiation later.
- Keep the probe WAV and recording WAV at the same sample rate.
- Start with low speaker volume. Ultrasonic-ish chirps can still be audible or
  annoying.
