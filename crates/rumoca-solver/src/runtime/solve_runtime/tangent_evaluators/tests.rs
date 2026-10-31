use super::*;
use std::cell::RefCell;

struct PreparedJacobian;
impl CompiledSolveJacobianExpression for PreparedJacobian {
    fn call(
        &self,
        _: &[f64],
        _: &[f64],
        _: f64,
        _: &[f64],
        _: &[rumoca_core::ExternalTableData],
        _: &mut [f64],
    ) -> Result<(), String> {
        Err("the preparation fixture does not execute whole Jacobians".into())
    }
}

struct RecordingBackend {
    refuse: bool,
    spans: RefCell<Vec<Vec<rumoca_core::Span>>>,
}
impl SolveExecutionBackend for RecordingBackend {
    fn compile_expression(
        &self,
        _: &solve::ScalarProgramBlock,
    ) -> Result<Rc<dyn CompiledSolveExpression>, String> {
        Err("the fixture prepares Jacobians".into())
    }
    fn compile_jacobian_expression(
        &self,
        block: &solve::ScalarProgramBlock,
    ) -> Result<Rc<dyn CompiledSolveJacobianExpression>, String> {
        self.spans.borrow_mut().push(block.program_spans().to_vec());
        if self.refuse {
            return Err("the backend cannot execute this lane width".into());
        }
        Ok(Rc::new(PreparedJacobian))
    }
    fn compile_assignment_schedule(
        &self,
        _: &solve::SharedAssignmentSchedule,
    ) -> Result<Rc<dyn CompiledSolveAssignmentSchedule>, String> {
        Err("the fixture prepares Jacobians".into())
    }
    fn compile_event_transaction(
        &self,
        _: &solve::EventTransactionProgram,
    ) -> Result<Rc<dyn CompiledSolveEventTransaction>, String> {
        Err("the fixture prepares Jacobians".into())
    }
}

fn fixture() -> (solve::ScalarProgramBlock, solve::AlgebraicProjectionPlan) {
    let source = rumoca_core::SourceId::from_source_name("native_torn_preparation.mo");
    let rows = vec![
        vec![
            solve::LinearOp::LoadSeed { dst: 0, index: 0 },
            solve::LinearOp::StoreOutput { src: 0 },
        ],
        vec![
            solve::LinearOp::LoadSeed { dst: 0, index: 1 },
            solve::LinearOp::StoreOutput { src: 0 },
        ],
    ];
    let jvp = solve::ScalarProgramBlock::with_output_indices(
        rows,
        vec![
            rumoca_core::Span::from_offsets(source, 1, 2),
            rumoca_core::Span::from_offsets(source, 3, 4),
        ],
        vec![31, 5],
    )
    .unwrap();
    let plan = solve::AlgebraicProjectionPlan {
        blocks: vec![solve::AlgebraicProjectionBlock {
            rows: vec![5, 31],
            y_indices: vec![0, 1],
            tearing: Some(solve::BlockTearing {
                tear_y_indices: vec![0, 1],
                residual_rows: vec![5, 31],
                causal_steps: vec![],
            }),
            alternate_charts: vec![],
        }],
    };
    (jvp, plan)
}

#[test]
fn native_torn_preparation_preserves_remapped_source_provenance() {
    let (jvp, plan) = fixture();
    let recorded = Rc::new(RecordingBackend {
        refuse: false,
        spans: RefCell::new(Vec::new()),
    });
    let backend: Rc<dyn SolveExecutionBackend> = recorded.clone();
    let original: Rc<dyn CompiledSolveJacobianExpression> = Rc::new(PreparedJacobian);
    let prepared = torn_tangent_evaluators(&plan, &jvp, Some(&backend), Some(&original)).unwrap();
    let tangent = prepared[0].as_ref().unwrap();
    assert!(!tangent.evaluator.plan().directional());
    assert!(tangent.compiled.is_some());
    assert!(!Rc::ptr_eq(tangent.compiled.as_ref().unwrap(), &original));
    assert_eq!(
        *recorded.spans.borrow(),
        [vec![jvp.program_spans()[1], jvp.program_spans()[0]]]
    );
}

#[test]
fn an_unavailable_native_lane_width_retains_the_admitted_directional_program() {
    let (jvp, plan) = fixture();
    let backend: Rc<dyn SolveExecutionBackend> = Rc::new(RecordingBackend {
        refuse: true,
        spans: RefCell::new(Vec::new()),
    });
    let original: Rc<dyn CompiledSolveJacobianExpression> = Rc::new(PreparedJacobian);
    let prepared = torn_tangent_evaluators(&plan, &jvp, Some(&backend), Some(&original)).unwrap();
    let tangent = prepared[0].as_ref().unwrap();
    assert!(tangent.evaluator.plan().directional());
    assert!(Rc::ptr_eq(tangent.compiled.as_ref().unwrap(), &original));
    let reference = torn_tangent_evaluators(&plan, &jvp, None, None).unwrap();
    let tangent = reference[0].as_ref().unwrap();
    assert!(!tangent.evaluator.plan().directional());
    assert!(tangent.compiled.is_none());
}
