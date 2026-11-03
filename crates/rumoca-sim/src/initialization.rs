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
