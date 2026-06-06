// Copyright (c) 2026 Elias S. G. Carotti

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use batkit::{
    correlate::matched_filter,
    peak::find_peaks,
    probe::{Probe, ProbeBuilder, Window},
    range::delay_samples_to_distance_m,
};
use clap::{ArgAction, Parser, Subcommand};

#[cfg(feature = "audio")]
mod audio;

mod wav;

#[derive(Debug, Parser)]
#[command(name = "batkit-cli")]
#[command(about = "Small CLI tools for batkit acoustic ranging experiments")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Generate a probe WAV file.
    ///
    /// Segment syntax:
    ///
    ///   chirp:20ms:16000:22000:gain=1:window=hann
    ///   sine:120ms:1200:decay=6:gain=0.2:window=none
    ///   silence:40ms
    ///
    /// Segments are appended in order.
    Probe {
        #[arg(long)]
        out: PathBuf,

        #[arg(long, default_value_t = 48_000.0)]
        sample_rate: f32,

        #[arg(long = "segment", action = ArgAction::Append)]
        segments: Vec<String>,
    },

    /// Play a WAV file while recording microphone input.
    PlayRecord {
        #[arg(long)]
        play: PathBuf,

        #[arg(long)]
        out: PathBuf,

        #[arg(long, default_value_t = 300)]
        record_ms: u64,
    },

    /// Matched-filter a recording against a probe WAV.
    Analyze {
        #[arg(long)]
        probe: PathBuf,

        #[arg(long)]
        recording: PathBuf,

        #[arg(long, default_value_t = 0.30)]
        min_distance_m: f32,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Probe {
            out,
            sample_rate,
            segments,
        } => {
            let probe = build_probe(sample_rate, &segments)?;
            wav::write_mono_f32_wav(&out, probe.sample_rate as u32, &probe.samples)?;
            println!("wrote {}", out.display());
        }

        Command::PlayRecord {
            play,
            out,
            record_ms,
        } => {
            play_record(&play, &out, record_ms)?;
        }

        Command::Analyze {
            probe,
            recording,
            min_distance_m,
        } => {
            analyze(&probe, &recording, min_distance_m)?;
        }
    }

    Ok(())
}

fn build_probe(sample_rate: f32, segments: &[String]) -> Result<Probe> {
    let mut builder = ProbeBuilder::new(sample_rate);

    if segments.is_empty() {
        builder = builder.chirp(0.020, 16_000.0, 22_000.0, 1.0, Window::Hann);
        return Ok(builder.normalize().build());
    }

    for segment in segments {
        builder = append_segment(builder, segment)
            .with_context(|| format!("invalid segment: {segment}"))?;
    }

    Ok(builder.normalize().build())
}

fn append_segment(builder: ProbeBuilder, spec: &str) -> Result<ProbeBuilder> {
    let parts = spec.split(':').collect::<Vec<_>>();

    match parts.first().copied() {
        Some("chirp") => {
            if parts.len() < 4 {
                bail!("chirp needs: chirp:DURATION:START_HZ:END_HZ[:gain=N][:window=hann]");
            }

            let duration_s = parse_duration(parts[1])?;
            let start_hz = parse_hz(parts[2])?;
            let end_hz = parse_hz(parts[3])?;
            let gain = option_f32(&parts[4..], "gain")?.unwrap_or(1.0);
            let window = option_window(&parts[4..])?.unwrap_or(Window::Hann);

            Ok(builder.chirp(duration_s, start_hz, end_hz, gain, window))
        }

        Some("sine") | Some("ping") | Some("sine-ping") => {
            if parts.len() < 3 {
                bail!("sine needs: sine:DURATION:FREQ_HZ[:decay=N][:gain=N][:window=hann]");
            }

            let duration_s = parse_duration(parts[1])?;
            let freq_hz = parse_hz(parts[2])?;
            let decay = option_f32(&parts[3..], "decay")?.unwrap_or(6.0);
            let gain = option_f32(&parts[3..], "gain")?.unwrap_or(1.0);
            let window = option_window(&parts[3..])?.unwrap_or(Window::None);

            Ok(builder.sine_ping(duration_s, freq_hz, decay, gain, window))
        }

        Some("silence") => {
            if parts.len() != 2 {
                bail!("silence needs: silence:DURATION");
            }

            Ok(builder.silence(parse_duration(parts[1])?))
        }

        Some(kind) => bail!("unknown segment kind: {kind}"),
        None => bail!("empty segment"),
    }
}

fn analyze(probe_path: &Path, recording_path: &Path, min_distance_m: f32) -> Result<()> {
    let (probe_rate, probe) = wav::read_mono_f32_wav(probe_path)?;
    let (recording_rate, recording) = wav::read_mono_f32_wav(recording_path)?;

    if probe_rate != recording_rate {
        bail!("sample-rate mismatch: probe={probe_rate}, recording={recording_rate}");
    }

    let sample_rate = probe_rate as f32;
    let corr = matched_filter(&recording, &probe);

    if corr.is_empty() {
        bail!("recording is shorter than probe");
    }

    let max = corr.iter().copied().map(f32::abs).fold(0.0, f32::max);
    let threshold = max * 0.15;

    let direct = find_peaks(&corr, 0, threshold, 64)
        .first()
        .copied()
        .context("no direct peak found")?;

    let min_delay_samples = ((2.0 * min_distance_m / 343.0) * sample_rate).round() as usize;
    let peaks = find_peaks(&corr, direct.index + min_delay_samples, threshold, 64);

    println!("direct_peak={} value={:.3}", direct.index, direct.value);

    for peak in peaks.iter().take(16) {
        let delay = peak.index - direct.index;
        let distance = delay_samples_to_distance_m(delay, sample_rate);

        println!(
            "echo_peak={} delay={} samples distance={:.3} m value={:.3}",
            peak.index, delay, distance, peak.value
        );
    }

    Ok(())
}

#[cfg(feature = "audio")]
fn play_record(play: &Path, out: &Path, record_ms: u64) -> Result<()> {
    let (sample_rate, playback) = wav::read_mono_f32_wav(play)?;
    let duration = std::time::Duration::from_millis(record_ms);

    let recorded = audio::play_and_record(
        &playback,
        sample_rate,
        duration,
        1, // input channels
        1, // output channels
    )?;

    wav::write_mono_f32_wav(out, sample_rate, &recorded)?;

    println!("wrote {}", out.display());
    Ok(())
}

#[cfg(not(feature = "audio"))]
fn play_record(_play: &Path, _out: &Path, _record_ms: u64) -> Result<()> {
    bail!(
        "audio support is disabled. Rebuild with: cargo run -p batkit-cli --features audio -- play-record ..."
    );
}

fn parse_duration(s: &str) -> Result<f32> {
    if let Some(ms) = s.strip_suffix("ms") {
        Ok(ms.parse::<f32>()? / 1000.0)
    } else if let Some(sec) = s.strip_suffix('s') {
        Ok(sec.parse::<f32>()?)
    } else {
        Ok(s.parse::<f32>()?)
    }
}

fn parse_hz(s: &str) -> Result<f32> {
    let lower = s.to_ascii_lowercase();

    if let Some(khz) = lower.strip_suffix("khz") {
        Ok(khz.parse::<f32>()? * 1000.0)
    } else if let Some(hz) = lower.strip_suffix("hz") {
        Ok(hz.parse::<f32>()?)
    } else {
        Ok(lower.parse::<f32>()?)
    }
}

fn option_f32(options: &[&str], key: &str) -> Result<Option<f32>> {
    let prefix = format!("{key}=");

    for option in options {
        if let Some(value) = option.strip_prefix(&prefix) {
            return Ok(Some(value.parse()?));
        }
    }

    Ok(None)
}

fn option_window(options: &[&str]) -> Result<Option<Window>> {
    for option in options {
        if let Some(value) = option.strip_prefix("window=") {
            return match value {
                "none" => Ok(Some(Window::None)),
                "hann" => Ok(Some(Window::Hann)),
                _ => bail!("unknown window: {value}"),
            };
        }
    }

    Ok(None)
}

//  vim: set ts=4 sw=4 et:
