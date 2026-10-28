//! MLS 11.1.2, 11.2.7 and Appendix B: each instantiated event algorithm
//! retains its ordered assignments and one complete commit owner.
use super::*;

fn last(result: &crate::SimResult, name: &str) -> f64 {
    let index = result
        .names
        .iter()
        .position(|candidate| candidate == name)
        .unwrap();
    *result.data[index].last().unwrap()
}

#[test]
fn nested_event_algorithms_retain_distinct_transaction_owners() {
    let model = compile(
        r#"
package Test
  model Counter
    discrete Integer state(start=0,fixed=true);
    discrete Real timer(start=0,fixed=true);
  algorithm
    when sample(5,5) then
      state:=pre(state); timer:=pre(timer);
      if pre(state)==0 then state:=1;
      elseif pre(state)==1 then state:=2;
      elseif pre(state)==2 then state:=3;
      end if;
      timer:=pre(timer)+5;
    end when;
  end Counter;
  model Network
    Counter first; Counter second;
    Integer aliasState=first.state;
  end Network;
end Test;
"#,
        "Test.Network",
    );
    model.inspect(|view| assert_eq!(view.model_event_transaction_count(), 2));
    for execution_policy in [
        rumoca_solver::SimExecutionPolicy::Interpreter,
        rumoca_solver::SimExecutionPolicy::Auto,
    ] {
        let result = simulate_dae(
            &model,
            &SimOptions {
                t_end: 30.0,
                dt: Some(5.0),
                solver_mode: SimSolverMode::RkLike,
                execution_policy,
                ..Default::default()
            },
        )
        .expect("nested ordered event algorithms must simulate");
        assert_eq!(last(&result, "first.state"), 3.0);
        assert_eq!(last(&result, "second.state"), 3.0);
        assert_eq!(last(&result, "aliasState"), 3.0);
        assert_eq!(last(&result, "first.timer"), 30.0);
        assert_eq!(last(&result, "second.timer"), 30.0);
    }
}

#[test]
fn nested_event_algorithm_evaluates_its_conditional_state_update() {
    let model = compile(
        r#"
package Test
  model Unit
    parameter Real endTime=30;
    discrete Integer state(start=0,fixed=true);
    discrete Integer count(start=0,fixed=true);
  algorithm
    when sample(5,5) then
      state:=pre(state);count:=pre(count);
      if pre(state)==0 and time<=endTime then
        state:=1;count:=pre(count)+1;
      end if;
    end when;
  end Unit;
  model Network
    Unit unit;
    Integer aliasState=unit.state;
  end Network;
end Test;
"#,
        "Test.Network",
    );
    let result = simulate_dae(
        &model,
        &SimOptions {
            t_end: 30.0,
            dt: Some(5.0),
            solver_mode: SimSolverMode::RkLike,
            ..Default::default()
        },
    )
    .expect("one nested ordered event algorithm must simulate");
    assert_eq!(last(&result, "unit.count"), 1.0);
    assert_eq!(last(&result, "unit.state"), 1.0);
}

#[test]
fn nested_event_algorithm_with_vector_and_current_child_outputs_advances() {
    let model = compile(
        r#"
package Test
  model Feeder
    input Real available;
    input Boolean enabled;
    output Real dose;
  equation
    dose=if enabled then min(available,25) else 0;
  end Feeder;
  model Unit
    parameter Real stepTime=5;
    parameter Real target=1000*(1-0.01);
    discrete Real stock(start=30000,fixed=true);
    discrete Real contents[2](each start=0,each fixed=true);
    discrete Integer state(start=0,fixed=true),count(start=0,fixed=true);
    Feeder feeder(available=pre(stock),enabled=pre(state)==1);
  algorithm
    when sample(stepTime,stepTime) then
      stock:=pre(stock);contents:=pre(contents);state:=pre(state);count:=pre(count);
      if pre(state)==0 then
        if time<=28800 and pre(stock)>=target then state:=1;count:=pre(count)+1;end if;
      elseif pre(state)==1 then
        stock:=pre(stock)-feeder.dose;contents[1]:=pre(contents[1])+feeder.dose;
        if pre(contents[1])+feeder.dose>=1000 then state:=2;end if;
      end if;
    end when;
  end Unit;
  model Network
    Unit unit;
    Integer aliasState=unit.state;
    Real clock=time;
    Real aliasTime=clock;
  end Network;
end Test;
"#,
        "Test.Network",
    );
    model.inspect(|view| assert_eq!(view.model_event_transaction_count(), 1));
    let quotient = rumoca_phase_structural::quotient_aliases(&model)
        .unwrap()
        .expect("continuous alias must exercise DAE reconstruction");
    quotient.inspect(|view| assert_eq!(view.model_event_transaction_count(), 1));
    let result = simulate_dae(
        &model,
        &SimOptions {
            t_end: 30.0,
            dt: Some(5.0),
            solver_mode: SimSolverMode::RkLike,
            ..Default::default()
        },
    )
    .expect("nested vector event algorithm must simulate");
    assert_eq!(last(&result, "unit.count"), 1.0);
    assert_eq!(last(&result, "unit.state"), 1.0);
    assert_eq!(last(&result, "unit.stock"), 29875.0);
}

#[cfg(all(feature = "solver-diffsol", feature = "solver-rk45"))]
#[test]
fn incremental_session_keeps_future_periodic_events_beyond_initial_scan_scale() {
    let model = compile(
        r#"
model Counter
  Real x(start=0,fixed=true);
  discrete Integer count(start=0,fixed=true);
equation
  der(x)=1;
  when sample(5,5) then count=pre(count)+1; end when;
end Counter;
"#,
        "Counter",
    );
    for solver_mode in [SimSolverMode::RkLike, SimSolverMode::Bdf] {
        let mut session = SimulationSession::new(
            &model,
            SimOptions {
                t_end: 1.0 / 60.0,
                dt: Some(1.0 / 60.0),
                solver_mode,
                ..Default::default()
            },
        )
        .unwrap();
        for time in [1.0_f64, 4.9, 5.0, 5.1, 16.0, 31.0] {
            session.advance_to(time).unwrap();
            assert!((session.get("x").unwrap().unwrap() - time).abs() < 1e-8);
            assert_eq!(
                session.get("count").unwrap().unwrap(),
                (time / 5.0).floor(),
                "{solver_mode:?} at {time}"
            );
        }
        session.reset(0.0).unwrap();
        session.advance_to(16.0).unwrap();
        assert_eq!(session.get("count").unwrap().unwrap(), 3.0);
    }
}

#[cfg(all(feature = "solver-diffsol", feature = "solver-rk45"))]
#[test]
fn incremental_session_retains_future_deadlines_without_a_terminal_time() {
    let model = compile(
        r#"
model Deadlines
  discrete Integer periodic(start=0,fixed=true),deadlines(start=0,fixed=true);
  discrete Real deadline(start=3,fixed=true);
  discrete Boolean afterSeven(start=false,fixed=true),finished(start=false,fixed=true);
equation
  when sample(5,5) then periodic=pre(periodic)+1; end when;
  when time>=pre(deadline) then
    deadline=pre(deadline)+7;deadlines=pre(deadlines)+1;
  end when;
  when time>7 then afterSeven=true; end when;
  when terminal() then finished=true; end when;
end Deadlines;
"#,
        "Deadlines",
    );
    for solver_mode in [SimSolverMode::RkLike, SimSolverMode::Bdf] {
        let options = SimOptions {
            t_end: 1.0 / 60.0,
            dt: Some(1.0 / 60.0),
            solver_mode,
            ..Default::default()
        };
        let mut session = SimulationSession::new(&model, options.clone()).unwrap();
        for time in [2.0_f64, 3.0, 7.1, 16.0, 31.0] {
            session.advance_to(time).unwrap();
            assert_eq!(
                session.get("periodic").unwrap().unwrap(),
                (time / 5.0).floor()
            );
            assert_eq!(
                session.get("deadlines").unwrap().unwrap(),
                if time < 3.0 {
                    0.0
                } else {
                    ((time - 3.0) / 7.0).floor() + 1.0
                }
            );
            assert_eq!(
                session.get("afterSeven").unwrap().unwrap(),
                if time > 7.0 { 1.0 } else { 0.0 }
            );
            assert_eq!(session.get("finished").unwrap().unwrap(), 0.0);
        }
        let batch = simulate_dae(
            &model,
            &SimOptions {
                t_end: 16.0,
                dt: Some(1.0),
                ..options
            },
        )
        .unwrap();
        assert_eq!(last(&batch, "periodic"), 3.0);
        assert_eq!(last(&batch, "deadlines"), 2.0);
        assert_eq!(last(&batch, "finished"), 1.0);
    }
}

#[test]
fn nested_event_boolean_child_dependency_settles_before_algorithm() {
    let model = compile(
        r#"
package Test
  model Flow
    input Boolean enabled;
    output Real moved;
    output Boolean blocked;
  equation
    moved=if enabled then 25 else 0;
    blocked=enabled and moved<25;
  end Flow;
  model Unit
    discrete Integer state(start=0,fixed=true),blockedCount(start=0,fixed=true);
    Flow transfer(enabled=pre(state)==1);
  algorithm
    when sample(5,5) then
      state:=pre(state);blockedCount:=pre(blockedCount);
      if pre(state)==0 then state:=1;
      elseif pre(state)==1 then
        if transfer.blocked then blockedCount:=pre(blockedCount)+1;end if;
      end if;
    end when;
  end Unit;
  model Network
    Unit unit;
  end Network;
end Test;
"#,
        "Test.Network",
    );
    for execution_policy in [
        rumoca_solver::SimExecutionPolicy::Interpreter,
        rumoca_solver::SimExecutionPolicy::Auto,
    ] {
        let result = simulate_dae(
            &model,
            &SimOptions {
                t_end: 30.0,
                dt: Some(5.0),
                solver_mode: SimSolverMode::RkLike,
                execution_policy,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(last(&result, "unit.blockedCount"), 0.0);
    }
}

#[test]
fn external_host_persisted_component_keeps_future_events_and_reset() {
    let model = compile(
        r#"
model ExternalCounter
  Real x(start=0,fixed=true);
  discrete Integer count(start=0,fixed=true);
equation
  der(x)=1;
  when sample(5,5) then count=pre(count)+1; end when;
end ExternalCounter;
"#,
        "ExternalCounter",
    );
    for execution_policy in [
        rumoca_solver::SimExecutionPolicy::Interpreter,
        rumoca_solver::SimExecutionPolicy::Auto,
    ] {
        let options = SimOptions {
            t_end: 1.0 / 60.0,
            dt: Some(1.0 / 60.0),
            solver_mode: SimSolverMode::RkLike,
            execution_policy,
            ..Default::default()
        };
        let component = crate::prepare_fmi_component(&model, &options).unwrap();
        let declaration_inputs = serde_json::to_value(component.construction_inputs()).unwrap();
        let wire = rumoca_phase_solve::fmi::runtime_fmi_component_wire(&component).unwrap();
        let encoded = serde_json::to_string(&wire).unwrap();
        drop(component);
        let mut deserializer = serde_json::Deserializer::from_str(&encoded);
        let component =
            rumoca_phase_solve::fmi::deserialize_fmi_component(&mut deserializer).unwrap();
        assert_eq!(
            serde_json::to_value(component.construction_inputs()).unwrap(),
            declaration_inputs,
            "runtime persistence preserves admitted declaration facts"
        );
        assert_eq!(
            serde_json::to_string(
                &rumoca_phase_solve::fmi::runtime_fmi_component_wire(&component).unwrap()
            )
            .unwrap(),
            encoded,
            "checked runtime replay has a canonical construction wire"
        );
        let mut session =
            crate::rk45::SimulationSession::from_fmi_component(&component, options).unwrap();
        session.advance_to(16.0).unwrap();
        assert_eq!(session.get("count").unwrap().unwrap(), 3.0);
        assert!((session.get("x").unwrap().unwrap() - 16.0).abs() < 1e-8);
        session.reset(0.0).unwrap();
        session.advance_to(10.0).unwrap();
        assert_eq!(session.get("count").unwrap().unwrap(), 2.0);
    }
}
