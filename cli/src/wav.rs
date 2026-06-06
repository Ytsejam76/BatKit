// Copyright (c) 2026 Elias S. G. Carotti

use std::path::Path;

use anyhow::{Result, bail};

pub fn read_mono_f32_wav(path: &Path) -> Result<(u32, Vec<f32>)> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    if spec.channels == 0 {
        bail!("WAV has zero channels: {}", path.display());
    }

    let samples = match spec.sample_format {
        hound::SampleFormat::Float => read_float_samples(&mut reader, spec.channels)?,
        hound::SampleFormat::Int => {
            read_int_samples(&mut reader, spec.channels, spec.bits_per_sample)?
        }
    };

    Ok((spec.sample_rate, samples))
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
        let sample = sample.clamp(-1.0, 1.0);
        let sample = (sample * i16::MAX as f32) as i16;

        writer.write_sample(sample)?;
    }

    writer.finalize()?;
    Ok(())
}

fn read_float_samples<R: std::io::Read>(
    reader: &mut hound::WavReader<R>,
    channels: u16,
) -> Result<Vec<f32>> {
    let channels = channels as usize;
    let mut out = Vec::new();

    for (i, sample) in reader.samples::<f32>().enumerate() {
        let sample = sample?;

        if i % channels == 0 {
            out.push(sample);
        }
    }

    Ok(out)
}

fn read_int_samples<R: std::io::Read>(
    reader: &mut hound::WavReader<R>,
    channels: u16,
    bits_per_sample: u16,
) -> Result<Vec<f32>> {
    let channels = channels as usize;
    let scale = match bits_per_sample {
        8 => i8::MAX as f32,
        16 => i16::MAX as f32,
        24 | 32 => i32::MAX as f32,
        _ => bail!("unsupported integer WAV bit depth: {bits_per_sample}"),
    };

    let mut out = Vec::new();

    for (i, sample) in reader.samples::<i32>().enumerate() {
        let sample = sample?;

        if i % channels == 0 {
            out.push(sample as f32 / scale);
        }
    }

    Ok(out)
}

//  vim: set ts=4 sw=4 et:
