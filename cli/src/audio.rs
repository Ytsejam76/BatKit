// Copyright (c) 2026 Elias S. G. Carotti
use std::{
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};

use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

pub fn play_and_record(
    playback: &[f32],
    sample_rate: u32,
    duration: Duration,
    input_channels: u16,
    output_channels: u16,
) -> Result<Vec<f32>> {
    let host = cpal::default_host();

    let input_device = host
        .default_input_device()
        .context("no default input device")?;
    let output_device = host
        .default_output_device()
        .context("no default output device")?;

    let input_config = cpal::StreamConfig {
        channels: input_channels,
        sample_rate: cpal::SampleRate(sample_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    let output_config = cpal::StreamConfig {
        channels: output_channels,
        sample_rate: cpal::SampleRate(sample_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    let recorded = Arc::new(Mutex::new(Vec::<f32>::new()));
    let recorded_for_stream = Arc::clone(&recorded);

    let input_stream = input_device
        .build_input_stream(
            &input_config,
            move |data: &[f32], _| {
                if let Ok(mut recorded) = recorded_for_stream.lock() {
                    for frame in data.chunks(input_channels as usize) {
                        let mono = frame.iter().copied().sum::<f32>() / frame.len() as f32;
                        recorded.push(mono);
                    }
                }
            },
            move |err| eprintln!("input stream error: {err}"),
            None,
        )
        .context("failed to build input stream; try a different sample rate/channel count")?;

    let playback = playback.to_vec();
    let mut playback_index = 0usize;

    let output_stream = output_device
        .build_output_stream(
            &output_config,
            move |data: &mut [f32], _| {
                for frame in data.chunks_mut(output_channels as usize) {
                    let sample = playback.get(playback_index).copied().unwrap_or(0.0);
                    playback_index = playback_index.saturating_add(1);

                    for out in frame {
                        *out = sample;
                    }
                }
            },
            move |err| eprintln!("output stream error: {err}"),
            None,
        )
        .context("failed to build output stream; try a different sample rate/channel count")?;

    input_stream
        .play()
        .context("failed to start input stream")?;
    output_stream
        .play()
        .context("failed to start output stream")?;

    thread::sleep(duration);

    drop(output_stream);
    drop(input_stream);

    let recorded = Arc::try_unwrap(recorded)
        .map_err(|_| anyhow::anyhow!("failed to unwrap recording buffer"))?
        .into_inner()
        .map_err(|_| anyhow::anyhow!("recording buffer lock was poisoned"))?;

    if recorded.is_empty() {
        bail!("recorded no samples");
    }

    Ok(recorded)
}
