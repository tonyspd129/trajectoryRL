"""Local dev driver for sandbox_harness.py — shell_verifier scenarios.

Runs the unified single-container shell_verifier eval pipeline against
a single SKILL.md. Calls upstream
``TrajectorySandboxHarness.evaluate_miner()`` — image pulls, scenario
metadata fetching, /app injection, agent ``docker exec``, output
extraction and verifier container orchestration are all handled
upstream.

Architecture (feat/shell-verifier-scenarios, commit 138707e):
  - One sandbox-agent container per session (holds /app + /workspace).
  - Agent runs via ``docker exec -u agent`` (no testee container).
  - One verifier container per episode runs the scenario's
    ``tests/test.sh``; reward = (exit_code == 0). No LLM judge.
  - quality is binary: 0.0 or 1.0 per episode.

Artifact layout per run (scenario-keyed throughout — no "episode"):
    <out>/<label>-<timestamp>/
        SKILL.md
        metadata.json                  # harness scores, salt, costs
        result.json                    # full session payload (added here)
        sandbox.log                    # combined: all scenarios, [scenario] prefix
        sandbox-<scenario>.log         # per-scenario sandbox container stdout
        verifier.log                   # combined: all verifier containers
        verifier-<scenario>.log        # per-scenario verifier container stdout
        scenarios/<name>/
            testee_transcript.txt      # agent docker-exec stdout (harness)
            turns.jsonl                # hermes session export (harness)
            turns_export.err           # hermes export stderr (harness)
            evaluation.json            # {reward, passed, total, correctness,
                                       #  verifier_stdout, ctrf, cost_usd} (harness)
            verifier_stdout.txt        # verifier container stdout (harness)
            ctrf.json                  # pytest CTRF report (harness)
            episode.json               # {instruction_md, agent_output_path} (harness)
            error.txt                  # iff the cell raised (harness)
            workspace/                 # /workspace snapshot (run_eval_local)
                SKILL.md, INSTRUCTION.md, learned/, ...
            app/                       # /app snapshot (run_eval_local)
                <agent_output>, ...
            testee/                    # hermes runtime data (run_eval_local)
                sessions/session_*.json
                logs/agent.log

Usage (from trajectoryRL/):
    python scripts/run_eval_local.py --skill ../room/packs/c1/SKILL.md --out ../room/debug
    python scripts/run_eval_local.py --pack ../packs/0414/pack.json --out ../room/debug
    python scripts/run_eval_local.py --skill <a> --skill <b> --run-num 3 --out ../room/debug

Requires env vars (auto-loaded from trajectoryRL/.env.validator):
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL  (the testee LLM)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import logging
import secrets
import subprocess
import sys
import tarfile
import threading
from datetime import datetime
from pathlib import Path

TRAJECTORYRL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAJECTORYRL_ROOT))

from trajectoryrl.utils.config import ValidatorConfig  # noqa: E402
import trajectoryrl.utils.sandbox_harness as _sandbox_harness_mod  # noqa: E402
from trajectoryrl.utils.sandbox_harness import (         # noqa: E402
    TrajectorySandboxHarness,
    SANDBOX_SCENARIOS,
)


# ---------------------------------------------------------------------------
# Config override — redirect /var/lib paths to a user-writable dir
# ---------------------------------------------------------------------------

class _LocalValidatorConfig(ValidatorConfig):
    """ValidatorConfig that redirects /var/lib/trajectoryrl paths to a
    user-writable dir. LLM credentials still come straight from
    ``LLM_*`` / ``JUDGE_*`` env vars via ``ValidatorConfig.from_env``.
    """

    _local_state_dir: Path = Path.home() / ".trajectoryrl_local"

    def __post_init__(self):
        base = type(self)._local_state_dir
        if str(self.pack_cache_dir).startswith("/var/lib/"):
            self.pack_cache_dir = base / "packs"
        if str(self.eval_state_path).startswith("/var/lib/"):
            self.eval_state_path = base / "eval_state.json"
        if str(self.winner_state_path).startswith("/var/lib/"):
            self.winner_state_path = base / "winner_state.json"
        super().__post_init__()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_skill_md(path: Path) -> str:
    """Load SKILL.md from either a bare ``.md`` file or a pack ``.json``."""
    if path.suffix.lower() == ".json":
        pack = json.loads(path.read_text())
        skill = TrajectorySandboxHarness.extract_skill_md(pack)
        if not skill:
            raise SystemExit(f"No SKILL.md / skill.md found in pack {path}")
        return skill
    return path.read_text()


def _pack_label(path: Path) -> str:
    """Short label for a pack — uses parent dir for generic file names."""
    if path.name.lower() in ("skill.md", "pack.json"):
        return path.parent.name
    return path.stem


# ---------------------------------------------------------------------------
# Container log streaming + workspace/app snapshot
# ---------------------------------------------------------------------------

_STREAM_LOCKS: dict[str, threading.Lock] = {}


def _stream_container_logs(container, log_path: Path, label: str = "") -> None:
    """Spawn a daemon thread that tails container stdout/stderr to log_path."""
    key = str(log_path)
    lock = _STREAM_LOCKS.setdefault(key, threading.Lock())
    prefix = f"[{label}] ".encode() if label else b""
    tag = label or container.name

    def _write(data: bytes) -> None:
        with lock, open(log_path, "ab") as f:
            f.write(data)
            f.flush()

    def _run() -> None:
        _write(f"[stream-start {tag}] container={container.name}\n".encode())
        chunks = 0
        try:
            for chunk in container.logs(
                stream=True, follow=True, stdout=True, stderr=True, tail="all",
            ):
                if not chunk:
                    continue
                chunks += 1
                out = bytearray()
                for line in chunk.splitlines(keepends=True):
                    out += (prefix + line) if prefix else line
                if out and not out.endswith(b"\n"):
                    out += b"\n"
                _write(bytes(out))
            _write(f"[stream-end {tag}] chunks={chunks}\n".encode())
        except Exception as e:
            _write(f"[stream-error {tag}] {type(e).__name__}: {e}\n".encode())

    threading.Thread(target=_run, daemon=True, name=f"log-{tag}").start()


def _extract_tar(container, src_path: str, dest_dir: Path,
                 strip_prefix: str = "") -> bool:
    """Extract a path from a container via get_archive. User-owned, never root."""
    try:
        stream, _ = container.get_archive(src_path)
        raw = b"".join(stream)
    except Exception:
        return False
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
            for member in tar.getmembers():
                if strip_prefix and member.name.startswith(strip_prefix):
                    member.name = member.name[len(strip_prefix):]
                elif strip_prefix and member.name == strip_prefix.rstrip("/"):
                    continue
                if not member.name:
                    continue
                tar.extract(member, dest_dir, filter="data")
        return True
    except Exception:
        return False


def _container_scenario(container) -> str:
    """Extract the scenario name from a ``sandbox_<sid>_<scenario>`` name."""
    name = getattr(container, "name", "") or ""
    parts = name.split("_", 2)
    return parts[2] if len(parts) >= 3 else "unknown"


def _snapshot_sandbox_paths(container, run_dir: Path) -> None:
    """Snapshot per-scenario sandbox state at scenario teardown.

    Writes everything into ``scenarios/<name>/`` (no "episode"
    terminology). The harness's ``write_artifacts`` initially writes to
    ``episodes/scenario_<name>/``; we migrate that to the same
    ``scenarios/<name>/`` dir at the end of each pack run via
    ``_migrate_episodes_to_scenarios``.

    Captures:
      * /workspace        → scenarios/<X>/workspace/
                            (SKILL.md, INSTRUCTION.md, learned/,
                            turns.jsonl)
      * /app              → scenarios/<X>/app/
                            (final agent output state — the file the
                            verifier inspects)
      * /opt/data/sessions → scenarios/<X>/testee/sessions/
                            (hermes session_*.json — model, system prompt,
                            tool list, full message history)
      * /opt/data/logs     → scenarios/<X>/testee/logs/
                            (hermes agent.log — per-API-call latency lines)

    Idempotent: running it twice on the same container/dir is safe (tar
    extraction overwrites existing files). Best-effort: any extraction
    failure is logged and skipped.
    """
    scenario = _container_scenario(container)
    base = run_dir / "scenarios" / scenario
    for src, sub, prefix in (
        ("/workspace",         "workspace",       "workspace/"),
        ("/app",               "app",             "app/"),
        ("/opt/data/sessions", "testee/sessions", "sessions/"),
        ("/opt/data/logs",     "testee/logs",     "logs/"),
    ):
        try:
            ok = _extract_tar(container, src, base / sub, strip_prefix=prefix)
        except Exception as e:
            print(f"[local] snapshot {src} on {scenario} failed: {e}")
            continue
        if ok:
            print(f"[local] snapshotted {src} -> scenarios/{scenario}/{sub}/")


def _migrate_episodes_to_scenarios(run_dir: Path) -> None:
    """Move harness ``episodes/scenario_<X>/`` output into ``scenarios/<X>/``.

    The upstream harness still writes per-cell artifacts (transcript,
    turns.jsonl, evaluation.json, ctrf.json, verifier_stdout.txt,
    episode.json) into ``episodes/scenario_<X>/``. This script normalizes
    everything under ``scenarios/<X>/`` so the analyze server has a
    single layout to read. Merges with any pre-existing scenario dir
    that the snapshot patch already created (sandbox /workspace, /app,
    testee/sessions, testee/logs).

    Idempotent: a second call after migration is a no-op (the source
    ``episodes/`` dir is removed once empty).
    """
    src_root = run_dir / "episodes"
    if not src_root.is_dir():
        return
    dst_root = run_dir / "scenarios"
    dst_root.mkdir(parents=True, exist_ok=True)
    for child in list(src_root.iterdir()):
        if not child.is_dir():
            continue
        # Strip the legacy ``scenario_`` prefix; non-prefixed names
        # (e.g. an ``episode_<i>`` straggler) move as-is.
        new_name = child.name.removeprefix("scenario_")
        dst = dst_root / new_name
        dst.mkdir(parents=True, exist_ok=True)
        for entry in list(child.iterdir()):
            target = dst / entry.name
            if target.exists():
                # Conflict (snapshot wrote here first). Prefer the
                # snapshot for directory-typed paths; let the harness
                # win for files. Tar-extracted dirs (workspace/, app/,
                # testee/) are write-once so this rarely matters.
                if entry.is_dir() and target.is_dir():
                    for sub in entry.rglob("*"):
                        rel = sub.relative_to(entry)
                        out = target / rel
                        if sub.is_dir():
                            out.mkdir(parents=True, exist_ok=True)
                        else:
                            out.parent.mkdir(parents=True, exist_ok=True)
                            sub.replace(out)
                    # entry will be cleaned up below.
                else:
                    target.unlink()
                    entry.replace(target)
            else:
                entry.replace(target)
        # child should now be empty; rmtree if not (defensive).
        try:
            child.rmdir()
        except OSError:
            import shutil
            shutil.rmtree(child, ignore_errors=True)
    try:
        src_root.rmdir()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Docker patches — class-level so they survive harness re-creation
# ---------------------------------------------------------------------------

# Mutable handle to the active per-pack run dir. Updated via
# patch_for_local_logging(run_dir) between sequential pack runs.
_CURRENT_RUN: dict = {"dir": None}


def _current_run_dir() -> Path:
    rd = _CURRENT_RUN["dir"]
    if rd is None:
        raise RuntimeError(
            "patch_for_local_logging(run_dir) must be called before any "
            "container event fires"
        )
    return rd


def patch_for_local_logging(run_dir: Path) -> None:
    """Stream sandbox + verifier logs and snapshot sandbox paths,
    everything keyed by scenario so ``episodes/scenario_<X>/`` is the
    single source for that scenario's evidence.

    Container naming on this branch:
      sandbox_<sid>_<scenario>  → one per scenario (per session)
      verifier_<sid>_ep<i>      → one per episode

    Per-scenario log files written:
      sandbox.log                       — combined, with [scenario] prefix
      sandbox-<scenario>.log            — single scenario's container stdout
      verifier.log                      — combined, with [ep<i>] prefix
      verifier-<scenario>.log           — single scenario's verifier stdout

    Snapshots written on Container.stop() and again as a safety on
    Container.remove() (idempotent — tar extraction overwrites).

    Patches installed once and re-targeted per pack via ``_CURRENT_RUN``.
    No bind mounts. All extraction via ``get_archive`` (user-owned).
    """
    run_dir = run_dir.resolve()
    _CURRENT_RUN["dir"] = run_dir
    for name in ("sandbox.log", "verifier.log"):
        (run_dir / name).touch()

    from docker.models.containers import Container as _Ctr

    # Track verifier→scenario by session_id so verifier_<sid>_ep<i> can
    # be split into the same scenario log file the sandbox wrote to.
    # Rebuilt at every container start so scenarios across separate
    # sessions don't bleed.
    if not hasattr(patch_for_local_logging, "_verifier_scenarios"):
        patch_for_local_logging._verifier_scenarios = {}
    verifier_scenarios: dict[str, list[str]] = (
        patch_for_local_logging._verifier_scenarios
    )

    # --- Container.start: attach log streamer ---
    if not getattr(_Ctr.start, "_trajrl_patched", False):
        orig_start = _Ctr.start

        def patched_start(self, *args, **kwargs):
            result = orig_start(self, *args, **kwargs)
            name = getattr(self, "name", "") or ""
            rd = _current_run_dir()
            if name.startswith("sandbox_"):
                # sandbox_<sid>_<scenario>
                parts = name.split("_", 2)
                sid = parts[1] if len(parts) >= 2 else ""
                scenario = parts[2] if len(parts) >= 3 else parts[-1]
                # Combined run-level log + per-scenario log so both views
                # are available to the analyze server.
                _stream_container_logs(self, rd / "sandbox.log", label=scenario)
                _stream_container_logs(self, rd / f"sandbox-{scenario}.log")
                verifier_scenarios.setdefault(sid, []).append(scenario)
            elif name.startswith("verifier_"):
                # verifier_<sid>_ep<i> — look up which scenario this ep
                # belongs to via the sandbox-startup map. Falls back to
                # ep<i> tag if the lookup misses.
                parts = name.rsplit("_", 1)
                ep_tag = parts[-1]
                pre = parts[0]                                  # verifier_<sid>
                sid = pre[len("verifier_"):]
                ep_idx_str = ep_tag.removeprefix("ep")
                scenario = None
                if ep_idx_str.isdigit():
                    ep_idx = int(ep_idx_str)
                    seen = verifier_scenarios.get(sid, [])
                    if 0 <= ep_idx < len(seen):
                        scenario = seen[ep_idx]
                _stream_container_logs(self, rd / "verifier.log", label=ep_tag)
                if scenario:
                    _stream_container_logs(
                        self, rd / f"verifier-{scenario}.log",
                    )
            return result

        patched_start._trajrl_patched = True  # type: ignore[attr-defined]
        _Ctr.start = patched_start  # type: ignore[assignment]

    # --- Container.stop: primary snapshot trigger (called by harness teardown). ---
    if not getattr(_Ctr.stop, "_trajrl_patched", False):
        orig_stop = _Ctr.stop

        def patched_stop(self, *args, **kwargs):
            name = getattr(self, "name", "") or ""
            if name.startswith("sandbox_"):
                try:
                    _snapshot_sandbox_paths(self, _current_run_dir())
                except Exception as e:
                    print(f"[local] sandbox snapshot on stop failed: {e}")
            return orig_stop(self, *args, **kwargs)

        patched_stop._trajrl_patched = True  # type: ignore[attr-defined]
        _Ctr.stop = patched_stop  # type: ignore[assignment]

    # --- Container.remove: fallback snapshot trigger. Some teardown paths
    # call remove() without stop() (e.g., kill+remove on error). We
    # extract here too; the snapshot is idempotent so double-firing on
    # normal stop+remove just rewrites the same files. ---
    if not getattr(_Ctr.remove, "_trajrl_patched", False):
        orig_remove = _Ctr.remove

        def patched_remove(self, *args, **kwargs):
            name = getattr(self, "name", "") or ""
            if name.startswith("sandbox_"):
                # Only retry the snapshot if the scenario dir doesn't
                # already have evidence — avoids redundant extraction
                # on the normal stop→remove path.
                rd = _current_run_dir()
                scenario = _container_scenario(self)
                marker = rd / "scenarios" / scenario / "workspace"
                if not marker.exists():
                    try:
                        _snapshot_sandbox_paths(self, rd)
                    except Exception as e:
                        print(f"[local] sandbox snapshot on remove failed: {e}")
            return orig_remove(self, *args, **kwargs)

        patched_remove._trajrl_patched = True  # type: ignore[attr-defined]
        _Ctr.remove = patched_remove  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Epoch seed helpers (mirror validator.compute_epoch_seed)
# ---------------------------------------------------------------------------

def _compute_epoch_seed(epoch: int, netuid: int = 11) -> int:
    raw = f"trajectoryrl-{netuid}-epoch-{epoch}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def _fetch_mainnet_epoch_seed(which: str, netuid: int,
                              eval_interval_blocks: int = 7200,
                              network: str = "finney") -> tuple[int, int, int]:
    """Connect to bittensor, read current block, derive epoch seed."""
    import bittensor as bt
    subtensor = bt.Subtensor(network=network)
    try:
        block = subtensor.get_current_block()
    finally:
        for closer in (lambda: subtensor.substrate.close(),
                       lambda: subtensor.close()):
            try:
                closer()
            except Exception:
                pass
    epoch = block // eval_interval_blocks
    if which == "next":
        epoch += 1
    return block, epoch, _compute_epoch_seed(epoch, netuid)


# ---------------------------------------------------------------------------
# Local sandbox-agent build (for unreleased bench changes)
# ---------------------------------------------------------------------------

_LOCAL_BENCH_TAG = "trajrl-bench:local"


def _build_sandbox_image(src_dir: Path, tag: str = _LOCAL_BENCH_TAG) -> None:
    """Build the sandbox-agent image from a trajrl-bench checkout. Live output.

    Note: only rebuilds the sandbox-agent image. Scenario images (e.g.
    alexgshaw/...:20251031) come from their own registries; the local
    build does NOT replace them.
    """
    src = src_dir.resolve()
    dockerfile = src / "docker" / "Dockerfile.sandbox-agent"
    if not dockerfile.is_file():
        # Fall back to legacy filename for older bench checkouts.
        legacy = src / "docker" / "Dockerfile.sandbox"
        dockerfile = legacy if legacy.is_file() else dockerfile
    if not dockerfile.is_file():
        raise SystemExit(f"[local] no sandbox-agent Dockerfile at {dockerfile}")
    print(f"[local] building {tag} from {dockerfile} ...")
    r = subprocess.run(
        ["docker", "build", "-f", str(dockerfile), "-t", tag, str(src)],
        check=False,
    )
    if r.returncode != 0:
        raise SystemExit(f"[local] docker build failed (exit {r.returncode})")
    print(f"[local] built {tag}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pack", type=Path, nargs="+",
                     help="One or more pack.json files. Each is evaluated in "
                          "sequence; per-pack output dirs are named "
                          "<pack-parent-or-stem>-<timestamp>/.")
    src.add_argument("--skill", type=Path, nargs="+",
                     help="One or more SKILL.md files. Same per-pack output "
                          "dir convention as --pack.")
    p.add_argument("--out", type=Path, required=True,
                   help="Host directory for all logs/artifacts")
    p.add_argument("--seed", type=int, default=None,
                   help="Epoch seed. Overridden by --epoch. Defaults to 42.")
    p.add_argument("--epoch", choices=["current", "next"], default=None,
                   help="Derive epoch seed from a live mainnet block. "
                        "Requires bittensor.")
    p.add_argument("--pack-hash", type=str, default="local",
                   help="Pack hash label (cosmetic)")
    p.add_argument("--salt", type=str, default=None,
                   help="Validator salt (16 hex chars). Default: fresh per "
                        "iteration.")
    p.add_argument("--no-pull", action="store_true",
                   help="Skip docker pull (use cached images)")
    p.add_argument("--build-bench", type=Path, nargs="?",
                   default=None,
                   const=TRAJECTORYRL_ROOT.parent / "trajrl-bench",
                   help="Build the sandbox-agent image from a local "
                        "trajrl-bench checkout before running (default path: "
                        f"../trajrl-bench). Tag: '{_LOCAL_BENCH_TAG}'. "
                        "NOTE: only rebuilds sandbox-agent — scenario images "
                        "(e.g. alexgshaw/...) still come from their own "
                        "registries. Implies --no-pull for sandbox.")
    p.add_argument("--run-num", type=int, default=1,
                   help="How many times to evaluate each pack (default 1). "
                        "Iterations are round-robin across all packs.")
    p.add_argument("--scenario", metavar="NAME", action="append", default=None,
                   help="Run only this scenario (repeatable). Must be one of "
                        f"{list(SANDBOX_SCENARIOS)}. "
                        "Default: all scenarios.")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    pack_paths: list[Path] = list(args.pack or args.skill or [])
    for pp in pack_paths:
        if not pp.is_file():
            print(f"[local] ERROR: pack file not found: {pp}")
            return 2

    config = _LocalValidatorConfig.from_env()

    custom_bench = False
    if args.build_bench:
        _build_sandbox_image(args.build_bench)
        config.sandbox_image = _LOCAL_BENCH_TAG
        custom_bench = True
        print(f"[local] sandbox image override: {config.sandbox_image}")

    # Resolve epoch_seed — applied to every pack so comparisons are
    # apples-to-apples.
    if args.epoch:
        try:
            block, epoch, seed = _fetch_mainnet_epoch_seed(
                args.epoch, netuid=config.netuid,
                eval_interval_blocks=config.eval_interval_blocks,
                network=config.network,
            )
        except Exception as e:
            print(f"[local] ERROR: --epoch {args.epoch} failed to query "
                  f"bittensor: {e}")
            return 2
        args.seed = seed
        print(f"[local] --epoch {args.epoch}: block={block} epoch={epoch} "
              f"→ seed={seed}")
    elif args.seed is None:
        args.seed = 42

    pinned_salt: str | None = args.salt
    salt_source = "override (pinned)" if pinned_salt else "generated per iteration"

    if not config.llm_api_key:
        print("[local] ERROR: no LLM_API_KEY in trajectoryRL/.env.validator")
        return 2

    # --scenario filter: override the module-level constant so the harness
    # internals (which read sandbox_harness.SANDBOX_SCENARIOS directly) see
    # only the requested subset.
    if args.scenario:
        unknown = set(args.scenario) - set(SANDBOX_SCENARIOS)
        if unknown:
            print(f"[local] ERROR: unknown scenario(s): {sorted(unknown)}")
            print(f"[local]        valid: {list(SANDBOX_SCENARIOS)}")
            return 2
        _sandbox_harness_mod.SANDBOX_SCENARIOS = tuple(
            s for s in SANDBOX_SCENARIOS if s in set(args.scenario)
        )
        print(f"[local] scenario filter: {list(_sandbox_harness_mod.SANDBOX_SCENARIOS)}")

    print(f"[local] image:     {config.sandbox_image}")
    print(f"[local] scenarios: {list(_sandbox_harness_mod.SANDBOX_SCENARIOS)}")
    print(f"[local] model:     {config.llm_model}")
    print(f"[local] base_url:  {config.llm_base_url}")
    print(f"[local] key:       {config.llm_api_key[:8]}...{config.llm_api_key[-4:]}")
    print(f"[local] per-cell:  {config.sandbox_timeout_per_episode}s timeout "
          f"({len(_sandbox_harness_mod.SANDBOX_SCENARIOS)} cells per session)")
    print(f"[local] seed:     {args.seed}  salt source: {salt_source}  "
          f"pack_hash:{args.pack_hash}")
    print(f"[local] packs:    {len(pack_paths)} "
          f"({', '.join(_pack_label(p) for p in pack_paths)})  "
          f"× {args.run_num} iteration(s) "
          f"= {len(pack_paths) * args.run_num} total run(s)")

    harness = TrajectorySandboxHarness(config)

    if custom_bench:
        # Locally-built image — don't pull it. Query its CLI for version info.
        try:
            from trajectoryrl.utils.sandbox_harness import _docker_run_json
            info = _docker_run_json(
                harness.client, harness._sandbox_image,
                command=["python", "-m", "trajrl_bench.cli", "scenarios"],
            )
            harness.sandbox_version = info.get("version", "unknown")
            harness.sandbox_scenarios = info.get("scenarios", [])
        except Exception as e:
            print(f"[local] failed to query local sandbox version: {e}")
    elif not args.no_pull:
        print("[local] pulling images ...")
        await harness.pull_latest()

    # When skipping pulls (--no-pull or --build-bench), replace _pull_sync with
    # a version that only runs orphan cleanup — no docker pulls — so locally-
    # built images aren't overwritten on every eval call.
    if args.no_pull or custom_bench:
        harness._pull_sync = harness._cleanup_orphans

    print(f"[local] sandbox version: {harness.sandbox_version}")
    print(f"[local] scenarios:       {harness.sandbox_scenarios}")

    summary: list[dict] = []
    total_runs = len(pack_paths) * args.run_num
    run_counter = 0
    for run_idx in range(1, args.run_num + 1):
        iter_salt = pinned_salt or secrets.token_hex(8)
        if args.run_num > 1:
            print()
            print("#" * 70)
            print(f"# iteration {run_idx}/{args.run_num}  salt={iter_salt}")
            print("#" * 70)

        for pack_path in pack_paths:
            run_counter += 1
            label = _pack_label(pack_path)
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            dir_name = (
                f"{label}-{ts}_run{run_idx}" if args.run_num > 1
                else f"{label}-{ts}"
            )
            run_dir = args.out / dir_name
            run_dir.mkdir(parents=True, exist_ok=True)
            patch_for_local_logging(run_dir)

            skill_md = load_skill_md(pack_path)
            print()
            print("=" * 70)
            print(f"[local] [{run_counter}/{total_runs}] "
                  f"pack: {pack_path}  iter={run_idx}/{args.run_num}")
            print(f"[local] run dir:  {run_dir.resolve()}")
            print(f"[local] SKILL.md: {len(skill_md)} chars")
            print(f"[local] salt:     {iter_salt}")
            print("=" * 70)

            result = await harness.evaluate_miner(
                skill_md=skill_md,
                epoch_seed=args.seed,
                pack_hash=args.pack_hash,
                validator_salt=iter_salt,
            )

            # Harness writes per-cell artifacts under episodes/scenario_<X>/;
            # migrate to scenarios/<X>/ so the on-disk layout is
            # episode-free and matches what the analyze server reads.
            try:
                result.write_artifacts(run_dir)
                _migrate_episodes_to_scenarios(run_dir)
                print(f"[local] wrote artifacts to {run_dir}")
            except Exception as e:
                print(f"[local] write_artifacts failed: {e}")

            session_cells = [
                {
                    "scenario": getattr(ep, "scenario", None),
                    "quality": ep.quality,
                    "timed_out": ep.timed_out,
                    "duration_s": ep.duration_s,
                    "error": ep.error,
                    "reward": (ep.judge_result or {}).get("reward"),
                    "cost_usd": getattr(ep, "cost_usd", None),
                }
                for ep in result.session_result.episodes
            ]
            (run_dir / "result.json").write_text(json.dumps({
                "pack_source": str(pack_path),
                "pack_label": label,
                "iteration": run_idx,
                "iteration_total": args.run_num,
                "scenarios": result.scenarios,
                "scenario_qualities": result.scenario_qualities,
                "sandbox_version": harness.sandbox_version,
                "sandbox_scenarios": harness.sandbox_scenarios,
                "epoch_seed": args.seed,
                "validator_salt": iter_salt,
                "score": result.score,
                "success": result.success,
                "error": result.error,
                "mean_quality": result.mean_quality,
                "total_cost_usd": getattr(result, "total_cost_usd", None),
                "mean_cost_usd": getattr(result, "mean_cost_usd", None),
                "session": {"cells": session_cells},
            }, indent=2))

            tag = f"{label} run{run_idx}" if args.run_num > 1 else label
            print()
            print(f"[{tag}] scenarios     : {result.scenarios} "
                  f"(sandbox {harness.sandbox_version})")
            print(f"[{tag}] per-scenario  : "
                  f"{ {s: round(q, 3) for s, q in result.scenario_qualities.items()} }")
            print(f"[{tag}] mean quality  : {result.mean_quality:.3f}")
            print(f"[{tag}] final score   : {result.score:.3f}   "
                  f"(sum across {len(result.scenarios)} scenarios; "
                  f"max = {len(result.scenarios)})  "
                  f"qualified: {'YES' if result.success else 'NO'}")
            if getattr(result, "total_cost_usd", None) is not None:
                print(f"[{tag}] total cost    : ${result.total_cost_usd:.4f}")
            if result.error:
                print(f"[{tag}] ERROR         : {result.error}")

            summary.append({
                "label": label,
                "iteration": run_idx,
                "score": result.score,
                "mean_quality": result.mean_quality,
                "qualified": result.success,
                "error": result.error,
                "run_dir": str(run_dir.resolve()),
            })

    if len(summary) > 1:
        print()
        print("=" * 78)
        print("ALL-RUN SUMMARY")
        print("=" * 78)
        print(f"  {'pack':<20} {'iter':>5} {'score':>8} {'mean_q':>8}  qual")
        for row in summary:
            qual = "YES" if row["qualified"] else "NO "
            err = f"  ERR: {row['error']}" if row["error"] else ""
            print(
                f"  {row['label']:<20} "
                f"{row['iteration']:>5} "
                f"{row['score']:>8.3f} "
                f"{row['mean_quality']:>8.3f}  "
                f"{qual}{err}"
            )
        print("=" * 78)

        if args.run_num > 1:
            print()
            print("PER-PACK AGGREGATE (across iterations)")
            print("-" * 78)
            print(f"  {'pack':<20} {'n':>3} {'mean':>8} {'min':>8} {'max':>8}")
            from collections import defaultdict
            by_pack: dict[str, list[float]] = defaultdict(list)
            for row in summary:
                by_pack[row["label"]].append(row["score"])
            for label, scores in by_pack.items():
                print(
                    f"  {label:<20} "
                    f"{len(scores):>3d} "
                    f"{sum(scores) / len(scores):>8.3f} "
                    f"{min(scores):>8.3f} "
                    f"{max(scores):>8.3f}"
                )
            print("-" * 78)

    return 0 if all(not r["error"] for r in summary) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
