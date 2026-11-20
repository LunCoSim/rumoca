//! Compact reverse lookup derived from the checked output placements.

use super::{ScalarOutputSegmentRef, ScalarProgramOutputPlacement};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug)]
struct OrdinalRun {
    start: usize,
    end: usize,
    ordinal: usize,
}

#[derive(Debug)]
pub(super) struct OutputOrdinalIndex(Box<[OrdinalRun]>);

impl OutputOrdinalIndex {
    pub(super) fn new(entries: &[ScalarProgramOutputPlacement]) -> Self {
        let mut boundaries = Vec::new();
        let mut ordinal = 0;
        for segment in entries
            .iter()
            .flat_map(ScalarProgramOutputPlacement::segments)
        {
            match segment {
                ScalarOutputSegmentRef::Range(range) => {
                    if !range.is_empty() {
                        boundaries.push((range.start, true, ordinal, range.start));
                        boundaries.push((range.end, false, ordinal, range.start));
                        ordinal += range.len();
                    }
                }
                ScalarOutputSegmentRef::Explicit(indices) => {
                    for &index in indices {
                        boundaries.push((index, true, ordinal, index));
                        boundaries.push((index + 1, false, ordinal, index));
                        ordinal += 1;
                    }
                }
            }
        }
        boundaries.sort_unstable();
        let mut active = BTreeSet::new();
        let mut runs = Vec::new();
        let mut start = 0;
        for (position, entered, ordinal, origin) in boundaries {
            if start < position
                && let Some(&(first, first_origin)) = active.first()
            {
                Self::push_run(&mut runs, start, position, first + (start - first_origin));
            }
            start = position;
            if entered {
                active.insert((ordinal, origin));
            } else {
                active.remove(&(ordinal, origin));
            }
        }
        Self(runs.into_boxed_slice())
    }

    fn push_run(runs: &mut Vec<OrdinalRun>, start: usize, end: usize, ordinal: usize) {
        if let Some(previous) = runs.last_mut()
            && previous.end == start
            && previous.ordinal + (previous.end - previous.start) == ordinal
        {
            previous.end = end;
        } else {
            runs.push(OrdinalRun {
                start,
                end,
                ordinal,
            });
        }
    }

    pub(super) fn ordinal(&self, output: usize) -> Option<usize> {
        let index = self.0.partition_point(|run| run.end <= output);
        let run = self.0.get(index)?;
        output
            .checked_sub(run.start)
            .map(|offset| run.ordinal + offset)
    }
}
