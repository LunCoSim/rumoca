//! Numerical and observation contracts retained by the checked Modelica runtime.
use rumoca::Compiler;
use rumoca_sim::{SimOptions, SimResult, simulate_dae};

fn simulate(source: &str, model: &str) -> SimResult {
    let compiled = Compiler::new()
        .model(model)
        .compile_str(source, "contract.mo")
        .unwrap();
    simulate_dae(
        &compiled.dae,
        &SimOptions {
            t_end: 0.1,
            dt: Some(0.05),
            ..Default::default()
        },
    )
    .expect("the admitted initial system must simulate")
}
fn channel<'a>(trace: &'a SimResult, name: &str) -> &'a [f64] {
    let index = trace
        .names
        .iter()
        .position(|item| item == name)
        .expect("declared output is recorded");
    &trace.data[index]
}

#[test]
fn coupled_newton_retains_constant_dependencies_and_dependent_outputs() {
    let trace = simulate(
        r#"
model ConstantFeedback
  Real c; Real quotient; Real a; Real b;
  output Real dependent;
equation
  c = 0.25; quotient = 1/c;
  a = 2*b + 1; b = 3*a + 2;
  dependent = a + b + quotient;
end ConstantFeedback;
"#,
        "ConstantFeedback",
    );
    assert!(
        channel(&trace, "dependent")
            .iter()
            .all(|value| (value - 2.0).abs() < 1e-9)
    );
}

#[test]
fn coupled_square_root_stays_in_its_real_domain() {
    let trace = simulate(
        r#"
model RootDomain
  output Real a(start=0.01); Real b(start=0.0001);
equation
  a = sqrt(b); b = 2-3*a;
end RootDomain;
"#,
        "RootDomain",
    );
    let expected = (17.0_f64.sqrt() - 3.0) / 2.0;
    assert!(
        channel(&trace, "a")
            .iter()
            .all(|value| (value - expected).abs() < 1e-8)
    );
}

#[test]
fn coupled_projection_preserves_the_nearby_branch_with_scaled_output() {
    let trace = simulate(
        r#"
model ScaledFeedback
  output Real x(start=(5+sqrt(5))/2);
  output Real scaled_output;
equation
  x = -x*x + 5*x - 3;
  scaled_output = 1e8*exp(x);
end ScaledFeedback;
"#,
        "ScaledFeedback",
    );
    assert!(
        channel(&trace, "x")
            .iter()
            .all(|value| (value - 3.0).abs() < 1e-8)
    );
    assert!(
        channel(&trace, "scaled_output")
            .iter()
            .all(|value| (value / 1e8 - 3.0_f64.exp()).abs() < 1e-7)
    );
}

#[test]
fn runtime_function_updates_bounds_from_its_current_argument() {
    let trace = simulate(
        r#"
model RuntimeBisection
  function bisect
    input Real u; output Real result;
    protected Real lower; Real upper; Real midpoint;
  algorithm
    lower := 1; upper := 4;
    for i in 1:12 loop
      midpoint := (lower+upper)/2;
      if midpoint < u then lower := midpoint; else upper := midpoint; end if;
    end for;
    result := lower+upper;
  end bisect;
  Real u(start=0); output Real result;
equation
  u = 1.5 + 20*time;
  result = bisect(u);
end RuntimeBisection;
"#,
        "RuntimeBisection",
    );
    for (value, time) in channel(&trace, "result").iter().zip(&trace.times) {
        assert!(
            (value - 2.0 * (1.5 + 20.0 * time)).abs() < 0.001,
            "t={time}, result={value}"
        );
    }
}

#[test]
fn derivative_output_retains_its_runtime_producer() {
    let trace = simulate(
        r#"
model DerivativeOutput
  Real x(start=1, fixed=true); output Real rate;
equation
  der(x) = -2*x;
  rate = der(x);
end DerivativeOutput;
"#,
        "DerivativeOutput",
    );
    for (value, time) in channel(&trace, "rate").iter().zip(&trace.times) {
        assert!((value + 2.0 * (-2.0 * time).exp()).abs() < 1e-6);
    }
}
