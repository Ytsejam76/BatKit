// Copyright (c) 2026 Elias S. G. Carotti
use std::path::Path;

use anyhow::{bail, Context, Result};

#[derive(Debug, Clone)]
pub struct MonoWav {
    pub sample_rate: u32,
    pub samples: Vec<f32>,
}

pub fn read_wav_mono_f32(path: &Path) -> Result<MonoWav> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    if spec.channels == 0 {
        bail!("WAV has zero channels");
    }

    let samples = match spec.sample_format {
        hound::SampleFormat::Float => {
            let raw: Vec<f32> = reader
                .samples::<f32>()
                .collect::<Result<Vec<_>, _>>()
                .context("failed reading float WAV samples")?;
            downmix_to_mono(&raw, spec.channels)
        }
        hound::SampleFormat::Int => {
            read_int_samples_as_f32(&mut reader, spec.bits_per_sample, spec.channels)?
        }
    };

    Ok(MonoWav {
        sample_rate: spec.sample_rate,
        samples,
    })
}

pub fn write_mono_f32_wav(path: &Path, sample_rate: u32, samples: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)?;

    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let scaled = (clamped * i16::MAX as f32).round() as i16;
        writer.write_sample(scaled)?;
    }

    writer.finalize()?;
    Ok(())
}

fn read_int_samples_as_f32(
    reader: &mut hound::WavReader<std::io::BufReader<std::fs::File>>,
    bits_per_sample: u16,
    channels: u16,
) -> Result<Vec<f32>> {
    match bits_per_sample {
        8 => {
            let raw: Vec<i8> = reader
                .samples::<i8>()
                .collect::<Result<Vec<_>, _>>()
                .context("failed reading 8-bit WAV samples")?;
            let f: Vec<f32> = raw.into_iter().map(|x| x as f32 / i8::MAX as f32).collect();
            Ok(downmix_to_mono(&f, channels))
        }
        16 => {
            let raw: Vec<i16> = reader
                .samples::<i16>()
                .collect::<Result<Vec<_>, _>>()
                .context("failed reading 16-bit WAV samples")?;
            let f: Vec<f32> = raw
                .into_iter()
                .map(|x| x as f32 / i16::MAX as f32)
                .collect();
            Ok(downmix_to_mono(&f, channels))
        }
        24 | 32 => {
            let scale = ((1_i64 << (bits_per_sample - 1)) - 1) as f32;
            let raw: Vec<i32> = reader
                .samples::<i32>()
                .collect::<Result<Vec<_>, _>>()
                .context("failed reading 24/32-bit WAV samples")?;
            let f: Vec<f32> = raw.into_iter().map(|x| x as f32 / scale).collect();
            Ok(downmix_to_mono(&f, channels))
        }
        other => bail!("unsupported integer WAV bit depth: {other}"),
    }
}

fn downmix_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    let channels = channels as usize;

    if channels == 1 {
        return samples.to_vec();
    }

    samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32)
        .collect()
}
