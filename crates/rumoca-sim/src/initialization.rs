//! Initialization observations from the common Model Exchange host.

use rumoca_ir_solve::fmi::FmiComponent;

use crate::{SimOptions, SimulationDiagnosticError};

/// Initialize a checked component and read only the requested finite outputs,
/// in request order, without constructing a numerical integration plugin.
///
/// Prepare `component` with the admitted parameter and initial-input overrides
/// through [`crate::prepare_fmi_component`] first. `options` supplies the same
/// numerical and execution policy used by live initialization. Unknown output
/// names, initialization termination, and non-finite observations are errors.
/// The initialized instance is released after observation; this operation does
/// not transfer its state into another component.
pub fn initialize_fmi_component(
    component: &FmiComponent,
    options: &SimOptions,
    requested_outputs: &[String],
) -> Result<Vec<f64>, SimulationDiagnosticError> {
    let backend = crate::fmi_execution_backend(component, options);
    let host =
        crate::me_backend::initialized_host(component.into(), options, backend, "initialization")
            .map_err(|error| SimulationDiagnosticError::Solver(error.to_string()))?;
    host.initial_output_values(requested_outputs)
        .map_err(|error| SimulationDiagnosticError::Solver(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rumoca_compile::compile::{Session, SessionConfig};

    #[test]
    fn nested_tunable_parameters_reach_initial_equations() {
        let source = "model Navigation
            parameter Real initial_y=0;
            output Real y;
          initial equation y=initial_y;
          equation der(y)=0;
          end Navigation;
          model Probe
            parameter Real initial_y=60;
            Navigation navigation(initial_y=initial_y);
          end Probe;";
        let mut session = Session::new(SessionConfig::default());
        session
            .add_document("nested_initialization.mo", source)
            .unwrap();
        session.set_structural_overrides(&[(
            "initial_y".into(),
            rumoca_compile::compile::StructuralOverride::Real(60.0),
        )]);
        let compiled = session.compile_model("Probe").unwrap();
        let root = &compiled.flat.variables[&rumoca_core::VarName::new("initial_y")];
        let nested = &compiled.flat.variables[&rumoca_core::VarName::new("navigation.initial_y")];
        assert!(
            matches!(nested.binding.as_ref(), Some(rumoca_core::Expression::VarRef { name, .. })
            if name.instance_id() == Some(root.instance_id)),
            "the flattened binding must retain the exact enclosing instance dependency"
        );
        for expected in [60.0, 1.0, 2.0, -4.0, 60.0] {
            let options = SimOptions {
                param_overrides: vec![("initial_y".into(), expected)],
                ..Default::default()
            };
            let component = crate::prepare_fmi_component(&compiled.dae, &options).unwrap();
            let values =
                initialize_fmi_component(&component, &options, &["navigation.y".into()]).unwrap();
            assert!(
                (values[0] - expected).abs() < 1e-10,
                "override {expected}: initialized {}",
                values[0]
            );
        }
    }

    #[test]
    fn nested_override_chains_keep_arithmetic_inheritance_and_instance_ownership() {
        let source = "model Leaf
            parameter Real initial_y=0;
            output Real y;
          initial equation y=initial_y;
          equation der(y)=0;
          end Leaf;
          model Settings parameter Real initial_y=0; end Settings;
          model Branch
            extends Settings;
            parameter Real offset=0;
            parameter Real projected=2*initial_y+offset;
            Leaf direct(initial_y=initial_y);
            Leaf derived(initial_y=projected);
          end Branch;
          model Probe
            parameter Real initial_y=60;
            Branch left(initial_y=initial_y,offset=3);
            Branch right(initial_y=-initial_y,offset=5);
            Branch independent(initial_y=7,offset=11);
          end Probe;";
        let mut session = Session::new(SessionConfig::default());
        session.add_document("nested_chains.mo", source).unwrap();
        session.set_structural_overrides(&[(
            "initial_y".into(),
            rumoca_compile::compile::StructuralOverride::Real(60.0),
        )]);
        let compiled = session.compile_model("Probe").unwrap();
        let names = [
            "left.direct.y",
            "left.derived.y",
            "right.direct.y",
            "right.derived.y",
            "independent.direct.y",
            "independent.derived.y",
        ]
        .map(String::from);
        for value in [60.0, 1.0, 2.0, -4.0, 0.0, 60.0] {
            let options = SimOptions {
                param_overrides: vec![("initial_y".into(), value)],
                ..Default::default()
            };
            let component = crate::prepare_fmi_component(&compiled.dae, &options).unwrap();
            let actual = initialize_fmi_component(&component, &options, &names).unwrap();
            let expected = [
                value,
                2.0 * value + 3.0,
                -value,
                -2.0 * value + 5.0,
                7.0,
                25.0,
            ];
            for (name, (actual, expected)) in names.iter().zip(actual.iter().zip(expected)) {
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "{name}: root {value}, expected {expected}, actual {actual}"
                );
            }
        }
        let options = SimOptions {
            param_overrides: vec![("initial_y".into(), 1.0), ("left.initial_y".into(), 9.0)],
            ..Default::default()
        };
        let component = crate::prepare_fmi_component(&compiled.dae, &options).unwrap();
        assert_eq!(
            initialize_fmi_component(&component, &options, &names).unwrap(),
            [9.0, 21.0, -1.0, 3.0, 7.0, 25.0]
        );
    }

    #[test]
    fn nested_overrides_preserve_each_array_and_qualified_dependencies() {
        let source = "model Leaf
            parameter Real p=0;
            output Real y;
          initial equation y=p;
          equation der(y)=0;
          end Leaf;
          model Holder parameter Real p=0; end Holder;
          model Probe
            parameter Real p=60;
            parameter Real values[2]={p,2*p};
            Holder box(p=p+3);
            Leaf repeated[2](each p=p);
            Leaf indexed[2](p=values);
            Leaf qualified(p=box.p);
          end Probe;";
        let mut session = Session::new(SessionConfig::default());
        session
            .add_document("array_initialization.mo", source)
            .unwrap();
        session.set_structural_overrides(&[(
            "p".into(),
            rumoca_compile::compile::StructuralOverride::Real(60.0),
        )]);
        let compiled = session.compile_model("Probe").unwrap();
        let names = [
            "repeated[1].y",
            "repeated[2].y",
            "indexed[1].y",
            "indexed[2].y",
            "qualified.y",
        ]
        .map(String::from);
        for value in [60.0, 1.0, -4.0, 0.0, 60.0] {
            let options = SimOptions {
                param_overrides: vec![("p".into(), value)],
                ..Default::default()
            };
            let component = crate::prepare_fmi_component(&compiled.dae, &options).unwrap();
            assert_eq!(
                initialize_fmi_component(&component, &options, &names).unwrap(),
                [value, value, value, 2.0 * value, value + 3.0]
            );
        }
    }

    fn component(source: &str, options: &SimOptions) -> FmiComponent {
        let mut session = Session::new(SessionConfig::default());
        session
            .add_document("initial_observation.mo", source)
            .expect("parse fixture");
        let compiled = session
            .compile_model("InitialObservation")
            .expect("compile fixture");
        crate::prepare_fmi_component(&compiled.dae, options).expect("prepare fixture")
    }

    #[test]
    fn requested_initial_outputs_include_settled_inputs_and_events() {
        let options = SimOptions {
            initial_inputs: vec![("u".into(), 3.0)],
            ..Default::default()
        };
        let component = component(
            "model InitialObservation
                input Real u;
                Real x(start=0, fixed=false);
                discrete Real d(start=0, fixed=false);
                output Real y[2];
              initial equation
                x=u;
              equation
                der(x)=0;
                when initial() then d=2; end when;
                y={x+d,x};
              end InitialObservation;",
            &options,
        );
        let names = vec!["y[2]".into(), "y[1]".into(), "y[2]".into()];
        let values = initialize_fmi_component(&component, &options, &names)
            .expect("initialize and observe without an integrator");
        assert_eq!(values, vec![3.0, 5.0, 3.0]);
        let error = initialize_fmi_component(&component, &options, &["absent".into()])
            .expect_err("unknown initialization observations must fail");
        assert!(
            error
                .to_string()
                .contains("unknown initialization output `absent`")
        );
    }

    #[test]
    fn terminated_initialization_has_no_observation_result() {
        let options = SimOptions::default();
        let component = component(
            "model InitialObservation
                Real x(start=1, fixed=true);
              equation
                der(x)=0;
                when initial() then terminate(\"initial stop\"); end when;
              end InitialObservation;",
            &options,
        );
        let error = initialize_fmi_component(&component, &options, &["x".into()])
            .expect_err("terminated initialization cannot publish probe values");
        assert!(error.to_string().contains("initialization terminated"));
    }
}
