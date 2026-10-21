# Selected projection retention

Griffin's coupled initialization constructs many projection applications over
one large immutable source. Cranelift allocated a retention vector for every
source program for each application, including programs absent from all its
colors. An owned LunCoSim Tracy capture attributes 6.46% of the constructor
thread's leaf samples to allocating/dropping these vectors. This is sampling
attribution, not a wall-time saving estimate.

The emitter now allocates each selected program's retention storage on its
first issued color and reuses it across subsequent colors. Checked source
program identity keys the lookup; program and color iteration remain unchanged.
There is no public API, IR, equation, tolerance, execution-policy, or native
code lifetime change. Storage is released after the one application is emitted.

The regression fails before the fix: 16,398 retention slots are allocated for
13 selected operations. It passes with 13 slots, including repeated colors and
a nonzero source-program index. Native outputs are checked bit for bit across
fresh coordinates, parameters and conditional branches. All 13 focused
projection tests pass with:

```sh
CARGO_BUILD_JOBS=4 RUST_TEST_THREADS=4 RAYON_NUM_THREADS=4 \
  cargo test -p rumoca-exec-cranelift --lib projection -j 4
```

A matched diagnostic compiles the same checked Griffin FMI component from
LunCoSim prepared cache v13, source key `5c7237cb45d93c71`, with automatic native
execution, RK-like integration, 1/180 s communication steps and 1e-6 absolute
and relative tolerances. Three baseline constructions have median 7,589.186 ms;
two qualified candidate constructions have median 7,290.481 ms (3.94% lower).
A third candidate overlaps graphical app startup and is excluded from timing.
All six runs expose the same 6,053 values at initialization and twelve following
steps, with identical f64 bits. Their complete state digest is
`8def1a60ed868e958272691c30a8f0406747498e55e75526b8f88e28836bb409`.

The locally patched production LunCoSim executable has SHA-256
`149adc0ed3e15553ded926c2b3352751fd46af61a69ecc9e52f6c92b76b45f0b`.
Its owned graphical session confirms Griffin touchdown and reaches tick 1,848
with unchanged source hashes, clear readiness and zero retained errors. It exits
and releases port 4758. Its screenshot was inspected. Two foreign simulator
sessions remain active; readiness takes 34.911 s and retained physics samples
have median 70.295 ms. These contention-affected observations establish runtime
completion, not a comparative loading/FPS improvement. The preceding artifact's
startup overlaps the last diagnostic above. A separate baseline run is excluded
after a concurrent test-source edit and a foreign app launch.

The changed crate passes scoped Clippy with `--lib --tests --no-deps -D warnings`.
Clippy including dependencies stops on existing excessive nesting in
`rumoca-core/src/lib.rs:291-295`; that unrelated source is unchanged.

Detailed logs, paired f64 states and source/artifact hashes are retained in the
LunCoSim optimization checkout's `.cache/codex-evidence/performance-20261020/`,
under `projection-construction-*` and `griffin-rumoca-retention-final/`.
The loading, sub-millisecond physics and 200 FPS targets remain unmet; macOS
has not been exercised. No dependency publication is included.
