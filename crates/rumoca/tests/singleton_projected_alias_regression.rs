//! MLS 10.6.9 / SPEC_0022 ARR-026: eliminating a singleton array alias must
//! not append its [1] to an already scalar matrix element. The compiled DAE
//! is valid; the first bad projection previously arose during structural
//! substitution, before Solve-IR lowering.
use rumoca::Compiler;

#[test]
fn singleton_array_alias_preserves_scalar_matrix_projection() {
    let source = r#"
model SingletonProjectedAlias
  parameter Real matrix[1,1] = [2];
  Real diagonal[1];
  Real x(start=0, fixed=true);
  output Real result;
equation
  for i in 1:1 loop
    diagonal[i] = matrix[i,i];
  end for;
  der(x) = 1;
  result = x + sum(diagonal[i] for i in 1:1);
end SingletonProjectedAlias;
"#;
    let compiled = Compiler::new()
        .model("SingletonProjectedAlias")
        .compile_str(source, "SingletonProjectedAlias.mo")
        .unwrap();
    let matrix = compiled
        .dae
        .variables
        .parameters
        .iter()
        .find(|(name, _)| name.as_str() == "matrix")
        .unwrap()
        .1;
    assert_eq!(matrix.dims, vec![1, 1]);
    let opts = rumoca_sim::SimOptions {
        t_end: 0.1,
        dt: Some(0.1),
        ..Default::default()
    };
    let sim = rumoca_sim::rk45::simulate(&compiled.dae, &opts)
        .expect("singleton alias must remain lowerable after structural elimination");
    let column = sim.names.iter().position(|name| name == "result").unwrap();
    assert!(!sim.data[column].is_empty());
    assert!(
        sim.data[column]
            .iter()
            .zip(&sim.times)
            .all(|(value, time)| (*value - (2.0 + time)).abs() < 1e-10)
    );
}
