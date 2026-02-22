# Conversion Notes

## Feb 17

### Objective
Move ConvTasNet export from broad reshape failures (`shortcut1/concat1/ew_mult1`) toward a compileable first-block path on Hailo.

### What CAN be converted to ONNX/HAR (translation path works)
1. Encoder + masker + decoder baseline graph (`skip_topology_mode=concat`, `deconv_mode=grouped`) translates to HAR reliably.
2. Export-time topology variants translate to HAR:
- `skip_topology_mode=project`
- `mask_mul_mode=bypass`
- `force_n_src_1=true`
- decoder fallbacks: `ungrouped_blockdiag`, `reduced_deconv_128`, `reduced_deconv_64`, `conv1x1_head`
3. First-block truncated variants (`TRUNCATE_K_BLOCKS=1`) also translate to HAR across all tested fallbacks.

### What currently CANNOT compile to HEF (as of Feb 17 runs)
1. Baseline/full topology cannot compile due to allocator reshape insertion failure:
- `conv5..conv73` + `shortcut1` + `concat1` + `ew_mult1`
- root-cause family: `skip_concat_maskmul`
2. `project + mask_bypass + grouped deconv` fails with:
- `Super deconv is not supported with groups` (unsupported grouped super-deconv kernel)
3. `project + mask_bypass + ungrouped_blockdiag deconv` fails with:
- allocator resource infeasibility (`Memory units capacity exceeded`)
4. `project + mask_bypass + reduced_deconv_128/64` and `project + mask_bypass + conv1x1_head` remove skip/concat/mask failures but still fail on early conv reshape:
- `Reshape is needed for layers: conv2, conv3`

### Architecture-level conclusion
1. The original skip/concat branch is a dominant compile blocker in baseline topology.
2. Replacing skip/concat with projection (`skip_topology_mode=project`) is directionally correct:
- it removes `shortcut1/concat1` from first-block failure signatures.
3. After that fix, bottleneck moves to either:
- decoder kernel/resource constraints, or
- early conv reshape legalization (`conv2`, `conv3`).

### Current best isolation profile (for continued debugging)
`skip_topology_mode=project`, `mask_mul_mode=bypass`, `truncate_k_blocks=1` with decoder fallback sweeps.

### Artifacts reviewed
- `hailo/night_runs2/summary.tsv`
- `hailo/night_runs2/20260217_204039/*.log`
- `hailo/night_runs2/20260217_204039/*.failure.txt(.json)`
- `hailo/TO_MATTHEW_FEB_17.md`
