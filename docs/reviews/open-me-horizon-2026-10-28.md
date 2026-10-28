# Open FMI Model Exchange session horizons

Anchors: FMI 3 EnterInitializationMode's optional `stopTimeDefined` metadata; SPEC_0038 / SPEC_0044 ME component and common-host ownership, MLS sample scheduling and Appendix B. A live session with no defined experiment end must retain its future time events. A finite initial scan span is not an experiment end.

The public `SimulationSession` counter starts with `SimOptions.t_end=1/60`, then advances past 5 seconds. Before repair its first sample is missing (`count=0`, expected 1), reproducing the native workbench failure. The host lease already carries `stop_time=None`, but the component configuration retained the finite span; `discrete_states_after_update` clipped its FMI next-event output to it.

The component configuration and saved kernel state now retain an optional stop. Live composition supplies None; batch composition supplies the defined end. Stop schedules retain future static events in open sessions and create a terminal event only for a defined stop. The next-event query is lazy and accepts an optional bound for periodic and dynamic deadlines; actual finite host advance queries still use their target. No giant finite horizon or infinite-time integration target is introduced. Right-limit probes use the existing adjacent representable-time owner.

Focused validation covers periodic events beyond the initial scan span, both numerical modes, reset, future static and dynamic deadlines, and absence of a terminal event in an open session. A bounded batch negative control retains its terminal event. Production native workbench acceptance and the fixed twenty-model canary are recorded separately.
