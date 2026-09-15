#!/usr/bin/env python3
"""Batch pack submitter for TrajectoryRL Subnet 11.

Reads a submission config (JSON) and runs in **two phases**:

  PHASE 1 (burn all) — pays every ``recycle_alpha`` fee first (or carries a
  provided receipt), collecting all receipts. Burns run in parallel across
  coldkeys and sequentially within a coldkey (hotkeys sharing one coldkey
  would collide on the nonce if burned concurrently).

  PHASE 2 (submit all) — once every fee is paid, POSTs all packs at once
  (plain HTTP, fully parallel), so every submission lands in the challenger
  queue within the same short window rather than trickling in as each burn
  finishes.

Every item is assumed already vetted (the hotkey holds >= fee alpha for a fresh
burn, or the receipt is a valid unconsumed credit). The script does not re-check
eligibility — it just runs the submissions and reports what the server returned.

Each submitted pack is given a unique, anonymous trailing marker
(``<!-- <random 64-bit hex> -->``) so the byte hashes never collide. The
marker carries no wallet/hotkey info and is inert to the agent.

Usage
-----
    python trajectoryRL/scripts/submit.py

No arguments, and it works from any directory: config is read from
``submission.json`` **next to this script**, and results are written to
``submission_results.json`` next to it too. The ``pack`` path in the config may
be absolute, or relative to the repo parent (the dir holding both
``trajectoryRL/`` and ``room/``), or relative to your current directory.

submission.json
---------------
{
  "pack": "room/packs/v26/pack.json",   # OPTIONAL default pack for items that
                                        #   don't specify their own
  "password": "alpha1234",              # coldkey password for encrypted wallets
  "items": [
    # per-item pack: each hotkey can submit a different pack
    {"wallet": "tony11-4", "hotkey": "tony1", "pack": "room/packs/v26/1.json"},
    {"wallet": "tony11-4", "hotkey": "tony2", "pack": "room/packs/v26/2.json"},

    {"wallet": "tony11-5", "hotkey": "tony1"},                        # uses the default pack
    {"wallet": "tony11-1", "hotkey": "tony", "receipt": "8558488-6"}  # cite existing receipt
  ]
}

Each distinct pack is loaded and validated once. Every submission still gets its
own anonymous marker, so even two hotkeys sharing a pack never collide on hash.

``fee_alpha`` (50), ``netuid`` (11) and ``max_workers`` (8) are fixed defaults;
add them to submission.json only if you ever need to override them.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Fixed defaults — constant across runs; override via submission.json only if needed.
SCRIPT_DIR = Path(__file__).resolve().parent   # .../trajectoryRL/scripts
ROOT = SCRIPT_DIR.parent                        # .../trajectoryRL (repo pkg root)
CONFIG_PATH = SCRIPT_DIR / "submission.json"    # lives next to this script
RESULTS_PATH = SCRIPT_DIR / "submission_results.json"
DEFAULT_FEE_ALPHA = 50.0
DEFAULT_NETUID = 11
DEFAULT_WORKERS = 8

# Make ``trajectoryrl`` importable whether run from repo root or elsewhere.
sys.path.insert(0, str(ROOT))
from trajectoryrl.base.miner import TrajectoryMiner  # noqa: E402


def _resolve_path(p: str) -> Path:
    """Resolve a config path (e.g. the pack) independent of the current dir.

    Tries, in order: as given (absolute / CWD-relative), then relative to the
    repo parent (the dir holding both trajectoryRL/ and room/), then next to
    this script.
    """
    for cand in (Path(p), Path.cwd() / p, ROOT.parent / p, SCRIPT_DIR / p):
        if cand.exists():
            return cand
    return Path(p)  # let the caller report a clean "cannot read" error


def _build_pack(pack: dict) -> dict:
    """Return a byte-unique copy of ``pack``.

    Uniqueness comes from a single trailing HTML comment holding a random
    64-bit token — nothing more. It carries **no identifying information**
    (no wallet or hotkey names) and, being a standard markdown comment with
    no words or instructions, is inert to the agent that reads SKILL.md.
    """
    nonce = os.urandom(8).hex()  # 64 bits — collision-proof, anonymous
    files = dict(pack["files"])
    files["SKILL.md"] = files["SKILL.md"].rstrip("\n") + f"\n<!-- {nonce} -->\n"
    return {"schema_version": pack.get("schema_version", 1), "files": files}


def _burn_one(item: dict, cfg: dict) -> dict:
    """PHASE 1: pay the fee for one item — a fresh ``recycle_alpha`` burn, or
    just carry through a receipt the item already provides. Returns the item
    annotated with ``receipt`` = (block, index), or ``error`` on failure. No
    submission happens here."""
    w, hf = item.get("wallet"), item.get("hotkey")
    # carry the item's pack key through to phase 2
    res = {"wallet": w, "hotkey": hf, "receipt": None, "packkey": item.get("_packkey")}
    try:
        given = item.get("receipt")
        if given:
            parts = str(given).split("-")
            if len(parts) != 2:
                res["error"] = f"malformed receipt {given!r} (want 'block-index')"
                return res
            res["receipt"] = (int(parts[0]), int(parts[1]))
            res["source"] = f"receipt {given}"
            return res
        miner = TrajectoryMiner(
            wallet_name=w, wallet_hotkey=hf,
            netuid=cfg["netuid"], wallet_password=cfg["password"],
        )
        try:
            rec = miner.recycle_alpha_fee(cfg["fee_alpha"])
        finally:
            miner.close()
        if rec is None:
            res["error"] = "recycle_alpha failed (balance / registration / chain)"
        else:
            res["receipt"] = rec
            res["source"] = f"burn {rec[0]}-{rec[1]}"
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
    return res


def _submit_one(burned: dict, cfg: dict) -> dict:
    """PHASE 2: POST one pack citing its already-obtained receipt. This is a
    plain HTTP call (no chain wait), so all submits fire near-simultaneously."""
    w, hf = burned["wallet"], burned["hotkey"]
    rb, ri = burned["receipt"]
    packkey = burned.get("packkey")
    res = {"wallet": w, "hotkey": hf, "ok": False, "source": burned.get("source"),
           "pack": Path(packkey).name if packkey else None}
    try:
        pack = _build_pack(cfg["packs"][packkey])
        miner = TrajectoryMiner(
            wallet_name=w, wallet_hotkey=hf,
            netuid=cfg["netuid"], wallet_password=cfg["password"],
        )
        try:
            resp = miner.submit_pack_via_api(
                pack, recycle_block=rb, recycle_extrinsic_index=ri,
            )
        finally:
            miner.close()
        if resp and resp.get("success"):
            res.update(ok=True, submission_id=resp.get("submission_id"),
                       pack_hash=resp.get("pack_hash"),
                       pre_eval=resp.get("pre_eval_status"))
        else:
            res["detail"] = f"submit rejected: {resp}"
    except Exception as e:  # noqa: BLE001
        res["detail"] = f"{type(e).__name__}: {e}"
    return res


def _coldkey_of(wallet: str) -> str:
    """Resolve a wallet's coldkey ss58 (grouping key). Two wallet dirs can share
    one coldkey — grouping by directory name would let them burn in parallel and
    collide on the shared nonce."""
    try:
        pub = Path.home() / ".bittensor" / "wallets" / wallet / "coldkeypub.txt"
        return json.loads(pub.read_bytes())["ss58Address"]
    except Exception:  # noqa: BLE001
        return f"wallet:{wallet}"  # fall back to dir name


def _burn_lane(coldkey: str, items: list, cfg: dict) -> list:
    """PHASE 1 lane: burn all of one coldkey's items sequentially (they share
    the coldkey nonce, so parallel burns would collide)."""
    out = []
    for it in items:
        try:
            r = _burn_one(it, cfg)
        except Exception as e:  # never lose the lane over one item  # noqa: BLE001
            r = {"wallet": it.get("wallet"), "hotkey": it.get("hotkey"),
                 "receipt": None, "error": f"{type(e).__name__}: {e}"}
        out.append(r)
        tag = f"{r['wallet']}/{r['hotkey']}"
        if r["receipt"]:
            print(f"  burned {tag:<22} {r.get('source','')}", flush=True)
        else:
            print(f"  BURN-FAIL {tag:<22} {r.get('error','')}", flush=True)
    return out


def main() -> int:
    try:
        raw = json.loads(Path(CONFIG_PATH).read_text())
    except FileNotFoundError:
        return _die(f"config not found: {CONFIG_PATH} (run from the dir that contains it)")
    except json.JSONDecodeError as e:
        return _die(f"invalid JSON in {CONFIG_PATH}: {e}")

    default_pack = raw.get("pack")           # optional fallback
    items = raw.get("items") or []
    if not items:
        return _die(f"no 'items' in {CONFIG_PATH}")

    # Each item may carry its own "pack"; otherwise the top-level "pack" is used.
    # Load + validate every distinct pack once, keyed by resolved path.
    packs: dict[str, dict] = {}
    for it in items:
        p = it.get("pack") or default_pack
        if not p:
            return _die(f"item {it.get('wallet')}/{it.get('hotkey')} has no 'pack' "
                        f"and no top-level 'pack' default is set")
        key = str(_resolve_path(p))
        it["_packkey"] = key
        if key in packs:
            continue
        try:
            pk = json.loads(Path(key).read_text())
        except Exception as e:  # noqa: BLE001
            return _die(f"cannot read pack {p}: {e}")
        issues = TrajectoryMiner.validate_s1(pk)
        if issues:
            return _die(f"pack {p} failed validation: " + "; ".join(issues))
        packs[key] = pk

    cfg = {
        "packs": packs,
        "password": raw.get("password") or os.environ.get("WALLET_PASSWORD"),
        "fee_alpha": float(raw.get("fee_alpha", DEFAULT_FEE_ALPHA)),
        "netuid": int(raw.get("netuid", DEFAULT_NETUID)),
    }
    workers = int(raw.get("max_workers", DEFAULT_WORKERS))
    # Phase-2 submit concurrency — capped low: >~20 simultaneous POSTs have
    # disconnected the server. Burns dominate wall-clock anyway.
    submit_conc = int(raw.get("submit_concurrency", 8))

    # Pre-sanitize every involved keyfile once, up front, so the parallel phase
    # never races on a keyfile rewrite (strips btcli's cryptoType, incl. inside
    # encrypted coldkeys).
    wroot = Path.home() / ".bittensor" / "wallets"
    done = set()
    for it in items:
        for p in (wroot / it["wallet"] / "coldkey",
                  wroot / it["wallet"] / "hotkeys" / it["hotkey"]):
            if p not in done:
                done.add(p)
                TrajectoryMiner._sanitize_keyfile(p, cfg["password"])

    # Group by *coldkey ss58* (not wallet dir): parallel across groups,
    # sequential within. Distinct wallet dirs may share one coldkey.
    groups: dict[str, list] = defaultdict(list)
    for it in items:
        groups[_coldkey_of(it["wallet"])].append(it)

    if len(packs) == 1:
        only = next(iter(packs))
        print(f"{len(items)} submission(s), pack {Path(only).name} "
              f"(hash {TrajectoryMiner.compute_pack_hash(packs[only])[:12]}) — "
              f"{len(groups)} coldkeys\n")
    else:
        print(f"{len(items)} submission(s) using {len(packs)} distinct packs — "
              f"{len(groups)} coldkeys")
        for it in items:
            print(f"    {it['wallet']}/{it['hotkey']:<7} <- {Path(it['_packkey']).name}")
        print()

    # ------------------------------------------------------------------
    # PHASE 1 — burn every fee first (parallel across coldkeys, sequential
    # within). Nothing is submitted yet; we only collect receipts.
    # ------------------------------------------------------------------
    print(f"PHASE 1 — burning {len(items)} fees across {len(groups)} coldkeys "
          f"(up to {workers} in parallel)...")
    t0 = time.time()
    burned: list = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_burn_lane, ck, its, cfg): ck
                for ck, its in groups.items()}
        for fut in as_completed(futs):
            try:
                burned.extend(fut.result())
            except Exception as e:  # noqa: BLE001
                print(f"  coldkey group {futs[fut][:12]}.. crashed: {e}", flush=True)
    ready = [b for b in burned if b["receipt"]]
    burn_fails = [b for b in burned if not b["receipt"]]
    print(f"  burned {len(ready)}/{len(burned)} in {time.time()-t0:.0f}s"
          f"{f' | {len(burn_fails)} burn failures' if burn_fails else ''}\n")

    # ------------------------------------------------------------------
    # PHASE 2 — submit ALL burned packs at once (plain HTTP, fully parallel,
    # so they all land in the queue within the same short window).
    # ------------------------------------------------------------------
    print(f"PHASE 2 — submitting all {len(ready)} packs at once...")
    t1 = time.time()
    results: list = []
    with ThreadPoolExecutor(max_workers=min(len(ready), submit_conc) or 1) as ex:
        futs = [ex.submit(_submit_one, b, cfg) for b in ready]
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            tag = f"{r['wallet']}/{r['hotkey']}"
            if r["ok"]:
                print(f"  OK   {tag:<22} {str(r.get('pack','')):<10} sub {r['submission_id']}  ({r.get('source','')})", flush=True)
            else:
                print(f"  FAIL {tag:<22} {str(r.get('pack','')):<10} {r.get('detail','')}", flush=True)
    print(f"  all submits done in {time.time()-t1:.0f}s")

    ok = [r for r in results if r["ok"]]
    print(f"\nDone: {len(ok)}/{len(items)} submitted "
          f"({len(burn_fails)} burn-failed, {len(results)-len(ok)} submit-rejected).")
    for b in burn_fails:
        print(f"  BURN-FAIL {b['wallet']}/{b['hotkey']}: {b.get('error','')}")
    for r in results:
        if not r["ok"]:
            print(f"  SUBMIT-FAIL {r['wallet']}/{r['hotkey']}: {r.get('detail','')}")
    # persist everything, including burn receipts (so a submit-only retry is possible)
    Path(RESULTS_PATH).write_text(json.dumps(
        {"burned": [{**b, "receipt": list(b["receipt"]) if b["receipt"] else None} for b in burned],
         "submitted": results}, indent=2))
    print(f"\nFull results written to {RESULTS_PATH}")
    return 0 if (len(ok) == len(items)) else 1


def _die(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
