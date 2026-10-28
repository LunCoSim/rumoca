//! Cooperative cancellation for a single admitted native solve.
use std::{
    cell::RefCell,
    marker::PhantomData,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

thread_local! {
    static EVALUATION_CANCELLATION: RefCell<Option<Arc<AtomicBool>>> = const { RefCell::new(None) };
}

/// Installs one run's cancellation flag until this thread-local scope ends.
/// Nested solves restore their caller's flag; the guard cannot move threads.
pub struct EvaluationCancellationGuard {
    previous: Option<Arc<AtomicBool>>,
    _thread: PhantomData<Rc<()>>,
}

impl EvaluationCancellationGuard {
    pub fn install(flag: Arc<AtomicBool>) -> Self {
        let previous = EVALUATION_CANCELLATION.with(|slot| slot.replace(Some(flag)));
        Self {
            previous,
            _thread: PhantomData,
        }
    }
}

impl Drop for EvaluationCancellationGuard {
    fn drop(&mut self) {
        EVALUATION_CANCELLATION.with(|slot| {
            slot.replace(self.previous.take());
        });
    }
}

#[inline]
pub fn evaluation_cancellation_requested() -> bool {
    EVALUATION_CANCELLATION.with(|slot| {
        slot.borrow()
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::SeqCst))
    })
}

/// Fail an evaluation at its owner before reading or writing output buffers.
pub(crate) fn check_evaluation_cancellation() -> Result<(), crate::EvalSolveError> {
    if evaluation_cancellation_requested() {
        Err(crate::EvalSolveError::Cancelled)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cancelled_evaluation_is_a_typed_error_and_scope_drop_restores_admission() {
        let guard = EvaluationCancellationGuard::install(Arc::new(AtomicBool::new(true)));
        assert!(matches!(
            check_evaluation_cancellation(),
            Err(crate::EvalSolveError::Cancelled)
        ));
        drop(guard);
        assert!(check_evaluation_cancellation().is_ok());
    }
    #[test]
    fn cancellation_is_thread_local_and_nested_scopes_restore_their_owner() {
        let outer = Arc::new(AtomicBool::new(false));
        let guard = EvaluationCancellationGuard::install(outer.clone());
        outer.store(true, Ordering::SeqCst);
        assert!(evaluation_cancellation_requested());
        std::thread::spawn(|| assert!(!evaluation_cancellation_requested()))
            .join()
            .expect("isolated thread");
        {
            let _inner = EvaluationCancellationGuard::install(Arc::new(AtomicBool::new(false)));
            assert!(!evaluation_cancellation_requested());
        }
        assert!(evaluation_cancellation_requested());
        drop(guard);
        assert!(!evaluation_cancellation_requested());
    }
}
