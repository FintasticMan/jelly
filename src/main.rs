use std::{cmp::Ordering, error::Error};

use itertools::{Itertools, izip};
use rand::distr::{Distribution, Uniform};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum FieldMatch {
    Previous,
    Current,
    Next,
    PreviousOpposite,
    NextOpposite,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Decimation {
    Keep,
    Decimate,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum FrameRate {
    Fps24p,
    Fps30p,
    Fps60i,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Modification {
    Leave,
    AddDecimation,
    RemoveDecimation,
}

#[derive(Debug)]
struct Section {
    start_frame: usize,
    frame_rate: FrameRate,
}

#[derive(Debug)]
struct Clip {
    sections: Vec<Section>,
    pattern: Vec<(FieldMatch, Decimation)>,
}

impl Clip {
    fn random(
        n_sections: usize,
        section_size_range_low: usize,
        section_size_range_high: usize,
    ) -> Result<Self, rand::distr::uniform::Error> {
        let mut rng = rand::rng();
        let section_size_range = Uniform::new(section_size_range_low, section_size_range_high)?;
        let frame_rate_range = Uniform::new(0u8, 10u8)?;
        let pattern_range = Uniform::new_inclusive(0u8, 4u8)?;

        let len_per_section: Vec<_> = section_size_range
            .sample_iter(&mut rng)
            .take(n_sections)
            .collect();
        let start_per_section: Vec<_> = std::iter::once(0)
            .chain(len_per_section.iter().scan(0, |state, len| {
                *state += len;
                Some(*state)
            }))
            .take(n_sections)
            .collect();
        let frame_rate_per_section: Vec<_> = frame_rate_range
            .sample_iter(&mut rng)
            .map(|v| match v {
                0..10 => FrameRate::Fps24p,
                100 => FrameRate::Fps30p,
                _ => FrameRate::Fps60i,
            })
            .take(n_sections)
            .collect();
        let sections: Vec<_> = start_per_section
            .iter()
            .zip(&frame_rate_per_section)
            .map(|(start_frame, frame_rate)| Section {
                start_frame: *start_frame,
                frame_rate: *frame_rate,
            })
            .collect();

        let pattern: Vec<_> = izip!(
            &len_per_section,
            &frame_rate_per_section,
            pattern_range.sample_iter(&mut rng)
        )
        .flat_map(|(len, frame_rate, pattern)| match frame_rate {
            FrameRate::Fps24p => (0..*len)
                .map(|i| match (i + pattern as usize) % 5 {
                    0..=2 => (FieldMatch::Current, Decimation::Keep),
                    3 => (
                        if i < len - 1 {
                            FieldMatch::Next
                        } else {
                            FieldMatch::PreviousOpposite
                        },
                        Decimation::Keep,
                    ),
                    _ => (
                        if i < len - 1 {
                            FieldMatch::Next
                        } else {
                            FieldMatch::PreviousOpposite
                        },
                        Decimation::Decimate,
                    ),
                })
                .collect::<Vec<_>>(),
            _ => (0..*len)
                .map(|_| (FieldMatch::Current, Decimation::Keep))
                .collect::<Vec<_>>(),
        })
        .collect();

        Ok(Self { sections, pattern })
    }
}

/// This calculates the error from a clip with a perfect decimation pattern (a consistent kkkkd)
/// pattern.
///
/// It does this by calculating the offset from that per section, then multiplying that by the
/// length of the section, and squaring it. The sum of those values is the error.
fn calc_error(
    len_per_section: &[usize],
    n_decimations_per_section: &[usize],
    frame_rate_per_section: &[FrameRate],
) -> isize {
    let offsets: Vec<_> = std::iter::once(0)
        .chain(
            izip!(
                len_per_section,
                n_decimations_per_section,
                frame_rate_per_section
            )
            .map(|(len, n_decimations, frame_rate)| match frame_rate {
                FrameRate::Fps24p => *len as isize - (n_decimations * 5) as isize,
                FrameRate::Fps30p => 0,
                FrameRate::Fps60i => 0,
            })
            .scan(0, |state, d| {
                *state += d;
                Some(*state)
            }),
        )
        .collect();

    offsets
        .iter()
        .zip(len_per_section)
        .fold(0, |acc, (o, len)| acc + (o * *len as isize).pow(2))
}

fn optimize_decimations(clip: &Clip) -> Vec<Modification> {
    let patterns_per_section: Vec<_> = clip
        .sections
        .windows(2)
        .map(|s| &clip.pattern[s[0].start_frame..s[1].start_frame])
        .chain(clip.sections.last().map(|s| &clip.pattern[s.start_frame..]))
        .collect();

    let len_per_section: Vec<_> = patterns_per_section.iter().map(|p| p.len()).collect();
    let n_decimations_per_section: Vec<_> = patterns_per_section
        .iter()
        .map(|p| p.iter().filter(|(_, d)| *d == Decimation::Decimate).count())
        .collect();
    let frame_rate_per_section: Vec<_> = clip
        .sections
        .iter()
        .map(|Section { frame_rate, .. }| *frame_rate)
        .collect();

    let mut best_modifications: Vec<_> =
        clip.sections.iter().map(|_| Modification::Leave).collect();
    let mut best_n_decimations_per_section = n_decimations_per_section.clone();
    let mut best_error = isize::MAX;
    for modifications in
        std::iter::repeat_n([false, true], clip.sections.len()).multi_cartesian_product()
    {
        let diff_per_section: Vec<_> = izip!(
            &len_per_section,
            &n_decimations_per_section,
            &frame_rate_per_section
        )
        .map(|(len, n_decimations, frame_rate)| match frame_rate {
            FrameRate::Fps24p => *len as isize - (n_decimations * 5) as isize,
            FrameRate::Fps30p => 0,
            FrameRate::Fps60i => 0,
        })
        .collect();

        let modifications: Vec<_> = modifications
            .iter()
            .zip(&diff_per_section)
            .map(|(p, diff)| {
                if *p {
                    match diff.cmp(&0) {
                        Ordering::Less => Modification::RemoveDecimation,
                        Ordering::Equal => Modification::Leave,
                        Ordering::Greater => Modification::AddDecimation,
                    }
                } else {
                    Modification::Leave
                }
            })
            .collect();
        let n_decimations_per_section: Vec<_> = n_decimations_per_section
            .iter()
            .zip(&modifications)
            .map(|(n_decimations, modification)| match modification {
                Modification::Leave => *n_decimations,
                Modification::AddDecimation => n_decimations + 1,
                Modification::RemoveDecimation => n_decimations - 1,
            })
            .collect();

        let error = calc_error(
            &len_per_section,
            &n_decimations_per_section,
            &frame_rate_per_section,
        );
        if error < best_error {
            best_modifications = modifications;
            best_n_decimations_per_section = n_decimations_per_section;
            best_error = error;
        }
    }

    println!("frames in original: {}", clip.pattern.len());
    println!(
        "frames in default decimation: {} - error: {}%",
        clip.pattern.len() - n_decimations_per_section.iter().sum::<usize>(),
        ((clip.pattern.len() - n_decimations_per_section.iter().sum::<usize>()) as f32
            / (clip.pattern.len() as f32 / 5. * 4.)
            - 1.)
            * 100.
    );
    println!(
        "frames in optimised decimation: {} - error: {}%",
        clip.pattern.len() - best_n_decimations_per_section.iter().sum::<usize>(),
        ((clip.pattern.len() - best_n_decimations_per_section.iter().sum::<usize>()) as f32
            / (clip.pattern.len() as f32 / 5. * 4.)
            - 1.)
            * 100.
    );
    println!(
        "frames in wobbly decimation: {} - error: {}%",
        clip.pattern.len() / 5 * 4,
        ((clip.pattern.len() / 5 * 4) as f32 / (clip.pattern.len() as f32 / 5. * 4.) - 1.) * 100.
    );

    best_modifications
}

fn main() -> Result<(), Box<dyn Error>> {
    let clip = Clip::random(20, 10, 20)?;

    let optimization = optimize_decimations(&clip);
    println!("{optimization:?}");

    Ok(())
}
