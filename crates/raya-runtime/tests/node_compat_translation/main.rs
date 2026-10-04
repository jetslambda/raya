//! Test-target root for the Node-compat translation harness.
//!
//! Cargo registers an integration target from `tests/<name>.rs` OR
//! `tests/<name>/main.rs`. This harness lived as `tests/node_compat_translation/mod.rs`
//! with no target root, so `cargo test -p raya-runtime --test
//! node_compat_translation` failed with `no test target named
//! 'node_compat_translation'` before running a single test — which is why the
//! `Node Compat Translation` CI job has been red on `main` as well as on this branch.
//!
//! The path below pulls the existing module in unchanged. Note that the harness's one
//! `#[test]` is a fixture GENERATOR that skips when `RAYA_NODE_TEST_ROOT` does not
//! exist, so a green run of this target means "the harness ran", not "Node
//! compatibility is verified".

#[path = "mod.rs"]
mod node_compat_translation;
