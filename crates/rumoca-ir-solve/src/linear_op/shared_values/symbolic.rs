//! Hash-consed symbolic values of scalar programs: the term table the
//! shared-value construction keys its values by and its checker evaluates
//! both sides over.

use rustc_hash::FxHashMap;

use super::super::*;
use super::shape::{ShapeKey, value_shape};

/// The id of one hash-consed symbolic value.
pub(super) type Term = usize;

/// One symbolic value.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum TermKey {
    /// A solver slot as the sequence found it.
    Slot(usize),
    /// Output `output` of the operation value `key`.
    Value { key: ValueKey, output: usize },
}

/// An operation value interned within one `Terms` table. The builder retains
/// that table across segment boundaries; a check uses one table for both sides.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct ValueKey(usize);

/// One operation's complete structural identity, with operands ordered by
/// field or register offset. Output terms refer to its interned key.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct OperationKey {
    shape: usize,
    operands: Vec<(usize, Term)>,
}

/// The hash-consed terms and interned operation shapes of one derivation or
/// check.
#[derive(Default)]
pub(super) struct Terms {
    ids: FxHashMap<TermKey, Term>,
    shapes: FxHashMap<ShapeKey, usize>,
    values: FxHashMap<OperationKey, ValueKey>,
}

impl Terms {
    fn intern(&mut self, key: TermKey) -> Term {
        let next = self.ids.len();
        *self.ids.entry(key).or_insert(next)
    }

    /// The key of the value of shape `shape` over `operands`.
    pub(super) fn key(&mut self, shape: &ShapeKey, operands: Vec<(usize, Term)>) -> ValueKey {
        let next = self.shapes.len();
        let shape = match self.shapes.get(shape) {
            Some(&id) => id,
            None => *self.shapes.entry(shape.clone()).or_insert(next),
        };
        let next = ValueKey(self.values.len());
        *self
            .values
            .entry(OperationKey { shape, operands })
            .or_insert(next)
    }

    pub(super) fn value(&mut self, key: ValueKey, output: usize) -> Term {
        self.intern(TermKey::Value { key, output })
    }
}

/// Symbolic evaluation of an ordered program sequence: the current term of
/// every solver slot, committed after each program.
#[derive(Default)]
pub(super) struct SymbolicSlots {
    slots: FxHashMap<usize, Term>,
}

impl SymbolicSlots {
    pub(super) fn slot(&self, terms: &mut Terms, index: usize) -> Term {
        match self.slots.get(&index) {
            Some(&term) => term,
            None => terms.intern(TermKey::Slot(index)),
        }
    }

    pub(super) fn commit(&mut self, targets: &[usize], outputs: &[Term]) {
        for (&target, &term) in targets.iter().zip(outputs) {
            self.slots.insert(target, term);
        }
    }

    pub(super) fn final_slots(&self) -> &FxHashMap<usize, Term> {
        &self.slots
    }

    /// The output terms of one program, committed to its targets; `None` when
    /// an operation cannot be evaluated or the output count is not the
    /// target count.
    pub(super) fn run(
        &mut self,
        terms: &mut Terms,
        ops: &[LinearOp],
        targets: &[usize],
    ) -> Option<Vec<Term>> {
        let mut registers: FxHashMap<Reg, Term> = FxHashMap::default();
        let mut outputs = Vec::with_capacity(targets.len());
        for op in ops {
            self.step(terms, op, &mut registers, &mut outputs)?;
        }
        if outputs.len() != targets.len() {
            return None;
        }
        self.commit(targets, &outputs);
        Some(outputs)
    }

    fn step(
        &self,
        terms: &mut Terms,
        op: &LinearOp,
        registers: &mut FxHashMap<Reg, Term>,
        outputs: &mut Vec<Term>,
    ) -> Option<()> {
        match *op {
            LinearOp::LoadY { dst, index } => {
                registers.insert(dst, self.slot(terms, index));
            }
            LinearOp::TensorLoad {
                dst_start,
                input: TensorInputKind::Y,
                input_start,
                count,
                seed_start: None,
                lanes: 1,
            } => {
                for offset in 0..count {
                    registers.insert(
                        dst_start + offset as Reg,
                        self.slot(terms, input_start + offset),
                    );
                }
            }
            LinearOp::Move { dst, src } => {
                let term = *registers.get(&src)?;
                registers.insert(dst, term);
            }
            LinearOp::StoreOutput { src } => outputs.push(*registers.get(&src)?),
            LinearOp::StoreOutputRange {
                start,
                count,
                stride,
            } => {
                for ordinal in 0..count {
                    let register = start + (ordinal * stride) as Reg;
                    outputs.push(*registers.get(&register)?);
                }
            }
            _ => return value_step(terms, op, registers, outputs),
        }
        Some(())
    }
}

/// Any other operation: its outputs are the value terms of its key; fold
/// output stores append that many opaque terms.
fn value_step(
    terms: &mut Terms,
    op: &LinearOp,
    registers: &mut FxHashMap<Reg, Term>,
    outputs: &mut Vec<Term>,
) -> Option<()> {
    let shape = value_shape(op)?;
    let operands = shape
        .operands
        .iter()
        .map(|&(field, register)| Some((field, *registers.get(&register)?)))
        .collect::<Option<Vec<_>>>()?;
    let key = terms.key(&shape.shape, operands);
    let stored = match op {
        LinearOp::StoreOutputFunctionFold { count, .. } => *count,
        LinearOp::StoreOutputFoldTensorUpdate {
            dimensions, lanes, ..
        } => dimensions.iter().fold(*lanes, |count, &extent| {
            count.saturating_mul(extent as usize)
        }),
        _ => 0,
    };
    for ordinal in 0..stored {
        outputs.push(terms.value(key, ordinal));
    }
    if let Some(start) = op.dst_register() {
        for offset in 0..op.dst_register_count() {
            registers.insert(start + offset as Reg, terms.value(key, offset));
        }
    }
    Some(())
}
