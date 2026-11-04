//! Initialized parameter reads use the checked FMI inventory and current host.
#![cfg(feature = "solver-rk45")]

use rumoca_compile::compile::{Session, SessionConfig};
use rumoca_sim::{SimOptions, prepare_fmi_component};

#[test]
fn initialized_parameter_inventory_preserves_types_arrays_inheritance_and_initial_equations() {
    let source = r#"package ParameterObservation
      model Base
        parameter Real gain=2*3;
        parameter Integer count=2;
        parameter Boolean enabled=true;
        parameter String label="checked";
      end Base;
      model Probe
        extends Base;
        parameter Real dependent[2]={gain,gain+1};
        parameter Real solved(start=1,fixed=false);
        constant Real calibration=99;
        input Real u=4;
        Real x(start=0,fixed=true);
        output Real y;
      initial equation solved=11;
      equation der(x)=0; y=gain+solved+u;
      end Probe;
    end ParameterObservation;"#;
    let mut compiler = Session::new(SessionConfig::default());
    compiler
        .add_document("parameter_observation.mo", source)
        .unwrap();
    let compiled = compiler
        .compile_model("ParameterObservation.Probe")
        .unwrap();
    let options = SimOptions {
        param_overrides: vec![("gain".into(), 8.0)],
        ..Default::default()
    };
    let component = prepare_fmi_component(&compiled.dae, &options).unwrap();
    let mut session =
        rumoca_sim::rk45::SimulationSession::from_fmi_component(&component, options).unwrap();
    let parameters = session.parameter_values().unwrap();
    assert_eq!(parameters["gain"], 8.0);
    assert_eq!(parameters["count"], 2.0);
    assert_eq!(parameters["enabled"], 1.0);
    assert_eq!(parameters["dependent[1]"], 8.0);
    assert_eq!(parameters["dependent[2]"], 9.0);
    assert_eq!(
        parameters["solved"], 11.0,
        "read the initialization solution, not its prepared seed"
    );
    for other in ["u", "y", "x", "calibration", "label"] {
        assert!(
            !parameters.contains_key(other),
            "{other} has a different inventory owner"
        );
    }
    assert!(session.get("gain").unwrap().is_none());
    assert!(
        session.set_input("gain", 4.0).is_err(),
        "a read cannot grant input write authority"
    );
    assert_eq!(session.parameter_values().unwrap(), parameters);
    session.reset(0.0).unwrap();
    assert_eq!(session.parameter_values().unwrap(), parameters);
}
