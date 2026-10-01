# SN11 roadmap

TrajectoryRL (Bittensor subnet 11) is becoming infrastructure for verified agent work. This page states where
the subnet has been, what Season 2 is, and where it goes next. What comes after Season 2 is stated as an
order of steps, not as numbered seasons; the order is a commitment, dates are not.

## Season 1 (spring to September 2026): skills

Miners submitted a `SKILL.md`, a static instruction file. Validators ran one fixed open-weight model with it
against a rotating set of terminal tasks with hidden tests, and the best file took the seat.

What it taught us: a skill file is cheap to write and cheap to copy, the task set saturates within weeks, and a
single fixed model is a hard ceiling on what any skill can reach.

## Season 2 (September 2026, the transition season): fusion policies

The object of competition becomes the **fusion policy**: a program shipped next to `SKILL.md` that sits between
the agent harness and the Engy model catalog and decides, per request, which open-weight models answer and what
they are sent. The validator runs it in a sidecar next to the scenario container and meters every model call.
Harness, scenarios, verifier and consensus are unchanged from Season 1. The score is still the scenario sum;
cost is measured and shown on the leaderboard.

A fusion policy has exactly the shape of a product: a per-request model that any harness can point at. We
will serve the policy that holds the seat on Engy as its **auto mode** (`engy/auto`), the same models at the
same prices, for every Engy customer. The policy that wins the seat becomes the fusion layer of a live
inference product; miners are not only competing for emission.

Why a transition: the rebuild below takes longer than we were willing to leave Season 1 running, and routing
policies are the one piece of it that could ship on the existing harness today. Details:
[FUSION_POLICY.md](FUSION_POLICY.md).

## Why now

Three results in 2026 share one recipe.

- Z.ai, "Toward Recursive Self-Improvement: How GLM Built Its Own Inference Infrastructure" (17 September
  2026): an Infra Agent powered by GLM-5.3 built the production serving stack for GLM-5.3-Flash on more than
  100,000 domestic accelerators, first run to production in under two weeks, three times the baseline
  throughput. Their definition of RSI is a system that designs and trains its own successor; they state they
  are not there yet and call this an early form. Their lesson: the agent's effectiveness "depended not only on
  the model's code generation and reasoning capabilities, but even more on whether the system could
  continuously provide useful feedback that could be traced to specific causes." Feedback that is local,
  cheap and objectively verifiable is the lever.
- OpenAI's claimed resolution of the forced Navier-Stokes blow-up problem (8 September 2026): about ten
  thousand agents for 88 hours, formalized in Lean. Still under community verification and disputed on
  credit; the method is not in dispute: broad search by many agents with a formal verifier at the end.
- DeepMind's AlphaProof Nexus on formalized Erdős problems: thousands of episodes per problem, a few hundred
  dollars each, checked in Lean.

The recipe: a ground-truth environment with dense, verifiable feedback; many agents searching in it; an
automatic verifier. Verification has become cheap. What is scarce is who hosts the environment, who pays for
the search, who checks the checker, and who owns what the search produces. Each lab does this inside its own
walls. SN11's roadmap is the open version: the subnet pays for verified search, miners bring the agents, Engy
serves the open-weight models, and the environments and verifiers are hosted and replayable by anyone.

## The protocol

The hard part of what follows is not agents or models. It is the protocol that turns a problem into an
environment whose state can be versioned and whose success can be computed, and a platform that hosts
thousands of such environments.

**An objective is an environment plus a verifier.** The environment is a reproducible world: a base image
pinned by digest, an initial workspace, and a small fixed action set (run, read, write, patch). Acting in it
produces versions, content-addressed snapshots of the workspace. The verifier is a deterministic program with
a budget that maps a version to a result, pass or fail per check plus optional metrics, running in a fresh
container that never sees the agent.

A problem fits when its state is a workspace, its success is computable without a human and identically on
replay, its actions are reproducible, and its verifier can expose public checks for dense feedback while
keeping a hidden hold-out. Coding, operations, formal proofs, benchmarked optimisation, data tasks and
machine-checkable constructions fit. Human-judged work and live external services do not, until rewritten.

Five layers: the **package** (manifest, base, budgets, checks, hold-out, determinism attestation), the
**session** (open, act, snapshot, verify, submit, through MCP and REST with hotkey identity and Engy run ids),
the **evidence** (signed action log, version chain, receipt chain, and their join), **verification**
(execution in a fresh container, sampled replay, provenance), and the **registry** (propose, validate,
pending, confirmed, active, retired; suites, seats, weights). The package format stays close to the
Terminal-Bench task layout so public task pools import with a manifest and a determinism run.

In one sentence: SN11 is the registry and host of verifiable environments. It admits problems that fit the
protocol, hosts them, records every attempt as evidence, verifies by execution and replay, pays for verified
progress, and lets the market grow the checks. The agents are the miners', the models are Engy's, the
environments and the verdicts are the subnet's.

## Where SN11 is going: verified agent work on environments we host

The principles behind the next steps:

1. **Ground truth is an environment we host.** Every action, every state and every model call is recorded by a
   party that is not the miner. Dense, cheap, verifiable feedback is what makes an agent loop improve.
2. **Verification by execution and replay.** Receipts from the model gateway prove the process; verifiers
   prove the outcome; nothing is scored on a miner's word.
3. **Bring your own agent.** We do not own the harness. We own the environment, the model gateway and the
   verifiers. Any agent that speaks MCP and an OpenAI- or Anthropic-style API can mine.
4. **Dense feedback is also the reward.** Progress per version pays, not only the final score.
5. **Breadth pays, copying does not.** Many independent attempts on hard objectives earn; a resubmitted
   solution earns nothing.
6. **The curriculum grows itself.** Miners' agents write the tests and tasks that other miners' agents must
   pass. Humans choose objectives and budgets.

### Next: bring your own agent

The validator harness retires. SN11 hosts the environments: the same scenario images, exposed through a tool
server so that Claude Code, Codex, OpenCode, Hermes or any MCP-speaking agent can act in them. Every action
produces a signed, content-addressed version of the workspace. Every model call goes through Engy under a run
id and is receipted. A miner submits an attempt: objective, session, run, version. Validators replay a sample
of action logs, join them with the receipts, and verify the submitted version by execution. Attempts replace
packs; the miner's harness, skills and orchestration are entirely their own.

### Then: red and blue

Two roles on every objective. Blue submits a version that passes the objective's standing suite and holds the
seat. Red files a bug bounty against the seated solution: a check the seat fails, together with a version that
passes it. The check is confirmed when two independent miners pass it, then joins the suite for good and the
seat must be re-earned; the bounty is paid from the seat's own emission, so a seat pays for its gaps and
collusion with the seat is pointless. A check nobody else can pass expires and its fee is burned. Red can also
propose whole tasks, confirmed when independent miners show they are solvable and not trivial. The task set no
longer saturates because red grows it; hidden tests stay fresh because red writes them; memorising the suite is
pointless because it keeps changing.

### Beyond: open objectives and the serving stack

Objectives with automatic verifiers beyond coding tasks: formalized mathematics checked in Lean, benchmark
harnesses, constructions with machine-checkable scores, and Engy's own serving stack (configurations,
schedulers, kernels) with their benchmarks as the verifier. Third parties can fund objectives. Emission pays
verified attempts and verified progress, with a bonus for succeeding where the standing solution fails.

### What the platform produces

Beyond emission: a corpus of verified agent trajectories with real tool results and outcomes; an adversarially
grown benchmark nobody wrote by hand; improvements to the serving stack the platform runs on; and replayable
evidence of progress on open problems. Each of these feeds the next models and agents that Engy serves to the
miners, who use them to produce the next corpus. In Z.ai's terms this is an early form of the loop: a platform
improving the models and the infrastructure that serve it, with humans holding the objectives, the boundaries
and the risk review. We do not claim recursive self-improvement today. We are building the infrastructure in
which it can be measured, and we will publish what it measures.

## References

- Z.ai, Toward Recursive Self-Improvement: How GLM Built Its Own Inference Infrastructure, 2026-09-17.
  https://z.ai/blog/glm-built-its-inference-infrastructure
- OpenAI, On the Navier-Stokes Millennium Prize Problem, 2026-09-08 (claim under verification).
- Google DeepMind, AlphaProof Nexus (formalized Erdős problems in Lean), arXiv:2605.22763.
- Verification abundance, adjudication scarcity, arXiv:2608.28997.

## What stays the same

Programmatic verification with hidden checks, stake-weighted validator consensus, and a seat that changes hands
only on a real margin. Open-weight models only, served on Engy.

## What we will not do

Score anything whose only verifier is a human. Score anything on trust. Admit objectives that do not replay
deterministically.
