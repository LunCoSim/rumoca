//! Compact, source-bound ownership of disjoint destination ranges.

use crate::{LinearOp, Reg};

struct DestinationRange {
    start: Reg,
    end: u64,
    position: usize,
}

pub(super) struct UniqueProgram<'a> {
    operations: &'a [LinearOp],
    ranges: Vec<DestinationRange>,
}

impl<'a> UniqueProgram<'a> {
    pub(super) fn new(operations: &'a [LinearOp]) -> Option<Self> {
        let mut ranges = Vec::new();
        for (position, operation) in operations.iter().enumerate() {
            let Some(start) = operation.dst_register() else {
                continue;
            };
            let count = u64::try_from(operation.dst_register_count()).ok()?;
            if count == 0 {
                continue;
            }
            let end = u64::from(start).checked_add(count)?;
            if end > u64::from(Reg::MAX) + 1 {
                return None;
            }
            ranges.push(DestinationRange {
                start,
                end,
                position,
            });
        }
        ranges.sort_unstable_by_key(|range| range.start);
        if ranges
            .windows(2)
            .any(|pair| pair[0].end > u64::from(pair[1].start))
        {
            return None;
        }
        Some(Self { operations, ranges })
    }

    pub(super) fn view(&self) -> ProgramPrefix<'_> {
        ProgramPrefix {
            operations: self.operations,
            ranges: &self.ranges,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct ProgramPrefix<'a> {
    operations: &'a [LinearOp],
    ranges: &'a [DestinationRange],
}

impl<'a> ProgramPrefix<'a> {
    pub(super) fn before(self, position: usize) -> Option<Self> {
        Some(Self {
            operations: self.operations.get(..position)?,
            ranges: self.ranges,
        })
    }

    pub(super) fn len(self) -> usize {
        self.operations.len()
    }

    pub(super) fn operation(self, position: usize) -> Option<&'a LinearOp> {
        self.operations.get(position)
    }

    pub(super) fn producer_position(self, register: Reg) -> Option<usize> {
        let index = self
            .ranges
            .partition_point(|range| range.start <= register)
            .checked_sub(1)?;
        let range = &self.ranges[index];
        (u64::from(register) < range.end && range.position < self.operations.len())
            .then_some(range.position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(start: Reg, count: usize) -> LinearOp {
        LinearOp::TensorLoad {
            dst_start: start,
            input: crate::TensorInputKind::Y,
            input_start: 0,
            count,
            seed_start: None,
            lanes: 1,
        }
    }

    fn reference_producer(operations: &[LinearOp], register: Reg) -> Option<usize> {
        operations.iter().position(|operation| {
            operation.dst_register().is_some_and(|start| {
                let start = u64::from(start);
                let end = start + operation.dst_register_count() as u64;
                (start..end).contains(&u64::from(register))
            })
        })
    }

    #[test]
    fn prefix_index_matches_linear_reference_for_sparse_permuted_ranges() {
        for count in [0, 1, 2, 7, 31, 257] {
            let base = Reg::MAX - (count * 8) as Reg;
            let mut operations = Vec::new();
            let mut registers = vec![0, Reg::MAX, base];
            for i in (0..count).rev() {
                let start = base + (i * 8) as Reg;
                let width = [0, 1, 3, 7][i % 4];
                operations.push(tensor(start, width));
                operations.push(LinearOp::StoreOutput { src: start });
                registers.extend([start - 1, start, start + width as Reg, start + 7]);
            }
            assert_prefixes_match(&operations, &registers);
        }
    }

    fn assert_prefixes_match(operations: &[LinearOp], registers: &[Reg]) {
        let source = UniqueProgram::new(operations).unwrap();
        for limit in 0..=operations.len() {
            let prefix = source.view().before(limit).unwrap();
            assert_eq!(prefix.len(), limit);
            for &register in registers {
                assert_eq!(
                    prefix.producer_position(register),
                    reference_producer(&operations[..limit], register)
                );
            }
        }
        assert!(source.view().before(operations.len() + 1).is_none());
    }

    #[cfg(feature = "diagnostic-benchmarks")]
    fn measure_index(
        operations: &[LinearOp],
        queries: &[(usize, Reg, Option<usize>)],
        iterations: usize,
    ) -> u128 {
        let started = std::time::Instant::now();
        for _ in 0..iterations {
            let source = UniqueProgram::new(std::hint::black_box(operations)).unwrap();
            for &(prefix, register, expected) in queries {
                assert_eq!(
                    source
                        .view()
                        .before(prefix)
                        .unwrap()
                        .producer_position(register),
                    expected
                );
            }
            std::hint::black_box(source);
        }
        started.elapsed().as_nanos()
    }

    #[test]
    #[cfg(feature = "diagnostic-benchmarks")]
    fn producer_index_diagnostic() {
        for count in [4, 64, 1024, 8192] {
            let operations: Vec<_> = (0..count)
                .map(|i| tensor((((i * 37) % count) * 8 + 1024) as Reg, 3))
                .collect();
            let queries: Vec<_> = (0..(count * 8).min(4096))
                .map(|i| {
                    let prefix = (i * 29) % (count + 1);
                    let register = ((i * 13) % (count * 8) + 1024) as Reg;
                    (
                        prefix,
                        register,
                        reference_producer(&operations[..prefix], register),
                    )
                })
                .collect();
            let iterations = 131_072 / count;
            for (with_queries, selected) in [(false, &[][..]), (true, queries.as_slice())] {
                let elapsed = measure_index(&operations, selected, iterations);
                println!(
                    "PRODUCER_INDEX count={count} queries={with_queries} iterations={iterations} elapsed_ns={elapsed}"
                );
            }
        }
    }

    #[test]
    fn tensor_ownership_is_compact_through_the_final_register() {
        let operations = [tensor(1, Reg::MAX as usize)];
        let source = UniqueProgram::new(&operations).unwrap();
        let view = source.view();
        assert_eq!(view.producer_position(0), None);
        assert_eq!(view.producer_position(1), Some(0));
        assert_eq!(view.producer_position(Reg::MAX), Some(0));
        assert_eq!(view.before(0).unwrap().producer_position(1), None);
    }

    #[test]
    fn prefix_ownership_preserves_gaps_and_nonmonotone_destinations() {
        let operations = [tensor(10, 3), tensor(2, 4), tensor(6, 2)];
        let source = UniqueProgram::new(&operations).unwrap();
        for limit in 0..=operations.len() {
            let view = source.view().before(limit).unwrap();
            for register in 0..16 {
                let expected = reference_producer(&operations[..limit], register);
                assert_eq!(view.producer_position(register), expected);
            }
        }
    }

    #[test]
    fn overlaps_and_overflow_cannot_issue_unique_ownership() {
        for operations in [
            [tensor(2, 4), tensor(4, 1)],
            [tensor(4, 1), tensor(2, 4)],
            [tensor(2, 4), tensor(2, 4)],
            [tensor(0, 1), tensor(Reg::MAX, 2)],
        ] {
            assert!(UniqueProgram::new(&operations).is_none());
        }
    }

    #[test]
    fn later_register_overwrite_does_not_invalidate_an_earlier_output() {
        let operations = [
            LinearOp::LoadY { dst: 5, index: 0 },
            LinearOp::LoadP { dst: 1, index: 0 },
            LinearOp::Binary {
                dst: 9,
                op: crate::BinaryOp::Sub,
                lhs: 5,
                rhs: 1,
            },
            LinearOp::StoreOutput { src: 9 },
            LinearOp::Const { dst: 5, value: 1.0 },
            LinearOp::StoreOutput { src: 9 },
        ];
        let shapes = super::super::derive_target_assignment_shapes(&operations);
        assert_eq!(shapes.len(), 1);
        assert_eq!(shapes[0].0, 0);
        assert_eq!(shapes[0].1.target_y_index(), 0);
        assert_eq!(
            super::super::canonical_assignment_shape_for_output(&operations, 0, 0),
            Some(shapes[0].1.clone())
        );
        assert!(super::super::canonical_assignment_shape_for_output(&operations, 1, 0).is_none());
    }
}
