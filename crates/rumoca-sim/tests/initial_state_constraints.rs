//! Nonredundant accepted free coordinates acquire given initialization ownership.
//! Identities preserve the checked component and its event program unchanged.
//! MLS §8.6 and SPEC_0043 initialization ownership retain every original row.

#![cfg(any(feature = "solver-diffsol", feature = "solver-rk45"))]

use rumoca_compile::compile::{Session, SessionConfig};
use rumoca_ir_solve::InitialStateConstraint;
use rumoca_ir_solve::fmi::{FmiComponent, FmiInitialStateConstraintError};
use rumoca_sim::{SimOptions, initialize_fmi_component, prepare_fmi_component};

fn component(source: &str, options: &SimOptions) -> FmiComponent {
    let mut session = Session::new(SessionConfig::default());
    session
        .add_document("initial_constraints.mo", source)
        .expect("parse fixture");
    let compiled = session
        .compile_model("InitialConstraints")
        .expect("compile fixture");
    prepare_fmi_component(&compiled.dae, options).expect("prepare fixture")
}

fn constraint(name: &str, value: f64) -> InitialStateConstraint {
    InitialStateConstraint::new(name, value).expect("finite constraint")
}

const FREE_STATE: &str = "model InitialConstraints
    input Real u=0;
    Real x(start=0, min=0, max=10);
    output Real y;
  initial equation u=0;
  equation der(x)=0; y=x;
  end InitialConstraints;";

#[test]
fn accepted_free_state_is_owned_without_mutating_cached_program_or_losing_residuals() {
    let options = SimOptions::default();
    let base = component(FREE_STATE, &options);
    assert_eq!(base.free_initial_state_names(), ["x"]);
    assert!(
        base.initial_state_constraint_features()
            .is_continuous_state_only()
    );
    let constrained = base
        .with_free_initial_state_constraints(&[constraint("x", 1.0)])
        .expect("admit the globally accepted free coordinate");
    assert!(constrained.free_initial_state_names().is_empty());
    assert_eq!(base.free_initial_state_names(), ["x"]);
    assert_eq!(
        constrained
            .problem()
            .initialization
            .residual()
            .len()
            .unwrap(),
        base.problem().initialization.residual().len().unwrap()
    );
    assert_eq!(
        constrained.problem().initialization.row_roles(),
        base.problem().initialization.row_roles()
    );
    let names = vec!["x".into(), "y".into()];
    assert_eq!(
        initialize_fmi_component(&base, &options, &names).unwrap(),
        [0.0, 0.0]
    );
    assert_eq!(
        initialize_fmi_component(&constrained, &options, &names).unwrap(),
        [1.0, 1.0]
    );
    let x = constrained
        .variables()
        .iter()
        .find(|v| v.name() == "x")
        .unwrap();
    assert_eq!(x.start(), Some([1.0].as_slice()));
}

#[test]
fn free_state_constraints_do_not_replace_original_initial_equations() {
    let options = SimOptions {
        initial_inputs: vec![("u".into(), 1.0)],
        ..Default::default()
    };
    let base = component(FREE_STATE, &options);
    let constrained = base
        .with_free_initial_state_constraints(&[constraint("x", 1.0)])
        .unwrap();
    assert!(
        initialize_fmi_component(&constrained, &options, &["x".into()]).is_err(),
        "the original initial equation u=0 must still reject u=1"
    );
}

#[test]
fn free_scalar_constraints_preserve_the_other_coordinates_equation_owner() {
    let options = SimOptions::default();
    let base = component(
        "model InitialConstraints
        Real x(start=0); Real y(start=0);
      initial equation x+y=5;
      equation der(x)=0; der(y)=0;
      end InitialConstraints;",
        &options,
    );
    let free = base.free_initial_state_names();
    assert_eq!(free.len(), 1);
    let constrained = base
        .with_free_initial_state_constraints(&[constraint(&free[0], 1.0)])
        .unwrap();
    let values =
        initialize_fmi_component(&constrained, &options, &["x".into(), "y".into()]).unwrap();
    assert!((values[0] + values[1] - 5.0).abs() < 1.0e-8);
    let free_value = initialize_fmi_component(&constrained, &options, &free).unwrap();
    assert_eq!(free_value, [1.0]);
}

#[test]
fn state_constraints_validate_identity_domain_and_single_ownership() {
    let base = component(FREE_STATE, &SimOptions::default());
    assert!(InitialStateConstraint::new("", 1.0).is_err());
    assert!(InitialStateConstraint::new("x", f64::NAN).is_err());
    assert!(InitialStateConstraint::new("x", f64::INFINITY).is_err());
    for name in ["missing", "y", "u"] {
        assert!(matches!(
            base.with_free_initial_state_constraints(&[constraint(name, 1.0)]),
            Err(FmiInitialStateConstraintError::NotFreeState { .. })
        ));
    }
    assert!(matches!(
        base.with_free_initial_state_constraints(&[constraint("x", 11.0)]),
        Err(FmiInitialStateConstraintError::OutsideDomain { .. })
    ));
    assert!(matches!(
        base.with_free_initial_state_constraints(&[constraint("x", 1.0), constraint("x", 2.0)]),
        Err(FmiInitialStateConstraintError::DuplicateState { .. })
    ));
    let owned = component(
        "model InitialConstraints Real x(start=2, fixed=true);
        equation der(x)=0; end InitialConstraints;",
        &SimOptions::default(),
    );
    assert!(owned.free_initial_state_names().is_empty());
    assert!(matches!(
        owned.with_free_initial_state_constraints(&[constraint("x", 2.0)]),
        Err(FmiInitialStateConstraintError::NotFreeState { .. })
    ));
}

#[test]
fn aggregate_constraints_keep_exact_scalar_identity() {
    let options = SimOptions::default();
    let base = component(
        "model InitialConstraints
        Real x[2](start={0,0}, each fixed=false);
      equation der(x)={0,0}; end InitialConstraints;",
        &options,
    );
    assert_eq!(base.free_initial_state_names(), ["x[1]", "x[2]"]);
    let constrained = base
        .with_free_initial_state_constraints(&[constraint("x[2]", 7.0), constraint("x[1]", 3.0)])
        .unwrap();
    assert_eq!(
        initialize_fmi_component(&constrained, &options, &["x[1]".into(), "x[2]".into()]).unwrap(),
        [3.0, 7.0]
    );
    assert!(
        base.with_free_initial_state_constraints(&[constraint("x[3]", 1.0)])
            .is_err()
    );
}

#[test]
fn hybrid_and_parameter_unknown_state_transfer_is_explicitly_refused() {
    let options = SimOptions::default();
    let hybrid = component(
        "model InitialConstraints Real x(start=0);
        discrete Real d(start=0, fixed=false);
      equation der(x)=0; when initial() then d=2; end when;
      end InitialConstraints;",
        &options,
    );
    assert!(
        hybrid
            .initial_state_constraint_features()
            .event_class
            .is_some()
    );
    assert!(matches!(
        hybrid.with_free_initial_state_constraints(&[constraint("x", 1.0)]),
        Err(FmiInitialStateConstraintError::UnsupportedFeatures { .. })
    ));
    let unchanged = hybrid.with_free_initial_state_constraints(&[]).unwrap();
    assert_eq!(
        initialize_fmi_component(&unchanged, &options, &["d".into()]).unwrap(),
        [2.0]
    );
    let parameter = component(
        "model InitialConstraints
        parameter Real q(start=1, fixed=false); Real x(start=0);
      initial equation q=2;
      equation der(x)=0; end InitialConstraints;",
        &options,
    );
    assert!(
        parameter
            .initial_state_constraint_features()
            .parameter_initialization_unknowns
    );
    assert!(matches!(
        parameter.with_free_initial_state_constraints(&[constraint("x", 1.0)]),
        Err(FmiInitialStateConstraintError::UnsupportedFeatures { .. })
    ));
}

#[test]
fn identity_constraints_preserve_hybrid_initialization_and_validate_before_normalizing() {
    let options = SimOptions::default();
    let hybrid = component(
        "model InitialConstraints Real x(start=0, min=0, max=10);
        discrete Real d(start=0, fixed=false);
      equation der(x)=0; when initial() then d=2; end when;
      end InitialConstraints;",
        &options,
    );
    assert!(
        !hybrid
            .initial_state_constraint_features()
            .is_continuous_state_only()
    );
    let unchanged = hybrid
        .with_free_initial_state_constraints(&[constraint("x", 0.0)])
        .expect("an identical prepared coordinate needs no continuous-state transfer");
    assert!(
        std::ptr::eq(hybrid.problem(), unchanged.problem()),
        "identity admission must preserve the original program owner"
    );
    assert_eq!(
        unchanged.free_initial_state_names(),
        hybrid.free_initial_state_names()
    );
    assert_eq!(
        initialize_fmi_component(&unchanged, &options, &["x".into(), "d".into()]).unwrap(),
        [0.0, 2.0]
    );
    assert!(matches!(
        hybrid.with_free_initial_state_constraints(&[constraint("missing", 0.0)]),
        Err(FmiInitialStateConstraintError::NotFreeState { .. })
    ));
    assert!(matches!(
        hybrid.with_free_initial_state_constraints(&[constraint("x", 0.0), constraint("x", 0.0)]),
        Err(FmiInitialStateConstraintError::DuplicateState { .. })
    ));
    assert!(matches!(
        hybrid.with_free_initial_state_constraints(&[constraint("x", 11.0)]),
        Err(FmiInitialStateConstraintError::OutsideDomain { .. })
    ));
    let error = hybrid
        .with_free_initial_state_constraints(&[constraint("x", 1.0)])
        .expect_err("a changed hybrid coordinate still requires unsupported state transfer");
    let FmiInitialStateConstraintError::UnsupportedFeatures {
        required_changes, ..
    } = error
    else {
        panic!("expected a structural capability diagnostic");
    };
    assert_eq!(required_changes, vec![("x".into(), 1.0, 0.0)]);
}
