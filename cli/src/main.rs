// Copyright (c) 2026 Elias S. G. Carotti

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use batkit::{
    correlate::matched_filter,
    peak::find_peaks,
    probe::{Probe, ProbeBuilder},
    range::delay_samples_to_distance_m,
    window::Window,
};
use clap::{Args, Parser, Subcommand, ValueEnum};

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
    Probe(ProbeArgs),

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

#[derive(Debug, Args)]
struct ProbeArgs {
    #[arg(long)]
    out: PathBuf,

    #[arg(long, default_value_t = 48_000.0)]
    sample_rate: f32,

    #[command(subcommand)]
    kind: ProbeCommand,
}

#[derive(Debug, Subcommand)]
enum ProbeCommand {
    /// Generate an ultrasonic chirp.
    Chirp(ChirpArgs),

    /// Generate an audible decaying sine ping.
    Ping(PingArgs),

    /// Generate an audible ping followed by silence and an ultrasonic chirp.
    Demo(DemoArgs),
}

#[derive(Debug, Args)]
struct ChirpArgs {
    #[arg(long, default_value_t = 20.0)]
    duration_ms: f32,

    #[arg(long, default_value_t = 16_000.0)]
    start_hz: f32,

    #[arg(long, default_value_t = 22_000.0)]
    end_hz: f32,

    #[arg(long, default_value_t = 1.0)]
    gain: f32,

    #[arg(long, value_enum, default_value_t = CliWindow::Hann)]
    window: CliWindow,

    #[arg(long, default_value_t = 6.0)]
    beta: f32,
}

#[derive(Debug, Args)]
struct PingArgs {
    #[arg(long, default_value_t = 120.0)]
    duration_ms: f32,

    #[arg(long, default_value_t = 1_200.0)]
    freq_hz: f32,

    #[arg(long, default_value_t = 6.0)]
    decay: f32,

    #[arg(long, default_value_t = 1.0)]
    gain: f32,

    #[arg(long, value_enum, default_value_t = CliWindow::None)]
    window: CliWindow,

    #[arg(long, default_value_t = 6.0)]
    beta: f32,
}

#[derive(Debug, Args)]
struct DemoArgs {
    #[arg(long, default_value_t = 120.0)]
    ping_ms: f32,

    #[arg(long, default_value_t = 1_200.0)]
    ping_hz: f32,

    #[arg(long, default_value_t = 6.0)]
    ping_decay: f32,

    #[arg(long, default_value_t = 0.20)]
    ping_gain: f32,

    #[arg(long, default_value_t = 40.0)]
    silence_ms: f32,

    #[arg(long, default_value_t = 20.0)]
    chirp_ms: f32,

    #[arg(long, default_value_t = 16_000.0)]
    chirp_start_hz: f32,

    #[arg(long, default_value_t = 22_000.0)]
    chirp_end_hz: f32,

    #[arg(long, default_value_t = 1.0)]
    chirp_gain: f32,

    #[arg(long, value_enum, default_value_t = CliWindow::Hann)]
    chirp_window: CliWindow,

    #[arg(long, default_value_t = 6.0)]
    beta: f32,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliWindow {
    None,
    Hann,
    Hamming,
    Blackman,
    Kaiser,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Probe(args) => {
            let probe = build_probe(args.sample_rate, args.kind)?;

            wav::write_mono_f32_wav(&args.out, probe.sample_rate as u32, &probe.samples)?;

            println!("wrote {}", args.out.display());
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

fn build_probe(sample_rate: f32, command: ProbeCommand) -> Result<Probe> {
    let probe = match command {
        ProbeCommand::Chirp(args) => ProbeBuilder::new(sample_rate)
            .chirp(
                args.duration_ms / 1000.0,
                args.start_hz,
                args.end_hz,
                args.gain,
                args.window.into_window(args.beta),
            )
            .normalize()
            .build(),

        ProbeCommand::Ping(args) => ProbeBuilder::new(sample_rate)
            .sine_ping(
                args.duration_ms / 1000.0,
                args.freq_hz,
                args.decay,
                args.gain,
                args.window.into_window(args.beta),
            )
            .normalize()
            .build(),

        ProbeCommand::Demo(args) => ProbeBuilder::new(sample_rate)
            .sine_ping(
                args.ping_ms / 1000.0,
                args.ping_hz,
                args.ping_decay,
                args.ping_gain,
                Window::None,
            )
            .silence(args.silence_ms / 1000.0)
            .chirp(
                args.chirp_ms / 1000.0,
                args.chirp_start_hz,
                args.chirp_end_hz,
                args.chirp_gain,
                args.chirp_window.into_window(args.beta),
            )
            .normalize()
            .build(),
    };

    Ok(probe)
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

    let recorded = audio::play_and_record(&playback, sample_rate, duration, 1, 1)?;

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

impl CliWindow {
    fn into_window(self, beta: f32) -> Window {
        match self {
            CliWindow::None => Window::None,
            CliWindow::Hann => Window::Hann,
            CliWindow::Hamming => Window::Hamming,
            CliWindow::Blackman => Window::Blackman,
            CliWindow::Kaiser => Window::Kaiser { beta },
        }
    }
}

//  vim: set ts=4 sw=4 et:
