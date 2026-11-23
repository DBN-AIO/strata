# Attention plan 2026-11-23

Source: the operator, 2026-11-23: "no project should be ignore more than 4 days". Last push before this plan: 2026-11-05.

## State

1. DBN-AIO/strata is a fork of Niko1221/Strata (the local-LLM engine, desktop app and Terminal; version 0.2.10, engine 0.1.56). `main` is the upstream state of 2026-11-02; nothing says it is frozen.
2. Our work is "remote experts": one MoE model across several PCs (docs/REMOTE_EXPERTS.md). A conductor keeps attention, router and its own experts; expert servers answer for a block of layers over TCP (protocol SXP1).
3. It lives on branch `remote-experts` (PR #1, open, mergeable, about 11,500 added lines, 41 commits). Eight stacked PRs (#2-#19, BBXL stages S2-S7, Qwen3.8-Max shapes, DSpark) merged into that branch on 2026-11-03/04, not into `main`.
4. CI on PR #1: Intake, Title area, Linux CPU-only build, Metal build all green at the last run (an earlier Title run failed, since passed).
5. Measured (docs/REMOTE_EXPERTS.md): Qwen3.8-Flash-Next Q2_0, 64 tokens: 13.10 tok/s alone, 17.41 with a localhost server, 2.92 with the slow AI-rig CPU as server; tokens identical.
6. Issues are disabled on the repo; the plan is the "Next steps" list at the end of docs/REMOTE_EXPERTS.md.
7. Gap found today: the layer ranges and the two commands are worked out by hand. Fixed by this slice: `tools/remote_experts_plan.py` plus tests/remote (13 tests) plus a CI job.

## Next five slices (delivery order)

1. Land PR #1 (decision A below). Then the operator can run `strata --remote-experts` from `main`.
2. fp16 hidden states on the wire (Next steps item 4): halves the LAN traffic; the operator sees the tok/s line rise on a 1 GbE pair.
3. Conductor without a local copy of the experts (item 1, open part): a conductor with a small disk can serve the model; the operator starts it with only the dense shard.
4. Stream-all and fused prompt paths through the servers (item 1): long prompts stop reading local experts; the operator sees the prompt time drop in the run table.
5. The planner reads the popularity profile (`data/expert-profile.bin`) so hot experts stay on the fastest host; the operator gets one plan for a mixed GPU/CPU fleet.

## Decisions that are the operator's

A. Merge PR #1 into `main` of the fork? It is 11.5k lines of engine and network code on top of Niko1221's `main`. Recommendation: yes, after one more read of the security section (token required, listen address bound exactly); until then every later slice has to be stacked on the branch.

B. Keep following Niko1221/Strata upstream, or treat the fork as ours? Recommendation: keep following; merge upstream into `main` monthly so PR #1 stays small.

C. Send the remote-experts work upstream as a pull request to Niko1221/Strata? Recommendation: no, not before decision A and a second user has run it; the code opens a network port.
