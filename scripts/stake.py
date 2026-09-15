#!/usr/bin/env python3
"""Stake ~N alpha into every hotkey under the listed coldkeys (SN11).

Reads ``stake.json`` (next to this script) and, for each listed coldkey, adds
stake to every signable hotkey in that wallet — enough TAO to receive
``target_alpha`` alpha at the current pool price. Runs in parallel across
coldkeys and sequentially within one (they share the coldkey nonce). Prints a
result table and writes ``stake_results.json``.

stake.json
----------
{
  "password": "alpha1234",        # coldkey password (encrypted wallets)
  "target_alpha": 50,             # alpha to receive per hotkey (default 50; +buffer lands ~51, clears the 50a fee)
  "buffer_pct": 2,                # extra % TAO for slippage/fees (default 2)
  "netuid": 11,
  "max_workers": 8,
  "skip_if_alpha_ge": 50,         # OPTIONAL: skip hotkeys already >= this alpha
  "coldkeys": ["tony11-3", "tony11-15", "tony11-16"]   # wallet names
}

Notes
-----
* ``amount`` in add_stake is TAO; the TAO needed for target_alpha is computed
  from the live pool (SubnetTAO / SubnetAlphaIn) read via the btcli env, whose
  newer substrate stack handles the chain's current NetUid type.
* This ADDS target_alpha to each hotkey — it does not "top up to". Run it only
  on coldkeys whose hotkeys need alpha, or set "skip_if_alpha_ge" to skip ones
  already funded.
* safe_staking / MEV shield are OFF (plain add_stake) — small stakes have
  negligible slippage and this avoids the shield-timeout seen with btcli.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
CONFIG_PATH = SCRIPT_DIR / "stake.json"
RESULTS_PATH = SCRIPT_DIR / "stake_results.json"
BTCLI_PY = "/home/chuck/.local/share/uv/tools/bittensor-cli/bin/python3"
WROOT = Path.home() / ".bittensor" / "wallets"

sys.path.insert(0, str(ROOT))
import bittensor as bt  # noqa: E402
from bittensor.utils.balance import Balance  # noqa: E402
from trajectoryrl.base.miner import TrajectoryMiner  # reuse _sanitize_keyfile  # noqa: E402


def _die(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 2


# Reliable-first; entrypoint-finney has been intermittently slow this session.
ENDPOINTS = ["wss://archive.chain.opentensor.ai", "wss://entrypoint-finney.opentensor.ai:443"]


def _btcli_query(body: str, timeout: int = 240):
    """Run a small snippet in the btcli env (newer substrate handles NetUid),
    trying each endpoint with retries. ``body`` uses the connected ``s`` and prints one JSON line."""
    eps = json.dumps(ENDPOINTS)
    code = (
        "import json,time,sys,os\n"
        "from async_substrate_interface.sync_substrate import SubstrateInterface\n"
        f"s=None\n"
        f"for ep in {eps}:\n"
        "  for _ in range(2):\n"
        "    try:\n"
        "      s=SubstrateInterface(url=ep); s.get_block_hash(None); break\n"
        "    except Exception: s=None; time.sleep(2)\n"
        "  if s: break\n"
        "assert s is not None, 'no endpoint reachable'\n"
        f"{body}\n"
        "sys.stdout.flush()\n"
        "os._exit(0)\n"   # hard-exit: the ws client hangs on normal teardown
    )
    out = subprocess.check_output([BTCLI_PY, "-c", code], text=True, timeout=timeout)
    return json.loads(out.strip().splitlines()[-1])


def read_pool(netuid: int):
    """Return (tao_in_rao, alpha_in_rao) from the pool, via the btcli env."""
    d = _btcli_query(
        f"t=s.query('SubtensorModule','SubnetTAO',[{netuid}]);"
        f"a=s.query('SubtensorModule','SubnetAlphaIn',[{netuid}]);"
        "print(json.dumps({'tao':int(getattr(t,'value',t)),'alpha':int(getattr(a,'value',a))}))"
    )
    return d["tao"], d["alpha"]


def read_alphas(ss58_list, netuid: int):
    """Return {ss58: alpha_float} for every hotkey, in one btcli-env call."""
    if not ss58_list:
        return {}
    body = (
        f"hks={json.dumps(list(ss58_list))}\n"
        "out={}\n"
        "for h in hks:\n"
        f"  r=s.query('SubtensorModule','TotalHotkeyAlpha',[h,{netuid}])\n"
        "  out[h]=float(getattr(r,'value',0))/1e9\n"
        "print(json.dumps(out))"
    )
    return _btcli_query(body, timeout=300)


def tao_rao_for_alpha(target_alpha: float, tao_in: int, alpha_in: int, buffer_pct: float) -> int:
    """AMM-exact TAO (rao) to receive target_alpha, plus a buffer for fees/slippage.

    alpha_out = alpha_in * amount / (tao_in + amount)  =>  amount = a*tao_in/(alpha_in - a)
    """
    a = target_alpha * 1e9  # target alpha in rao
    amount = a * tao_in / (alpha_in - a)
    return int(amount * (1 + buffer_pct / 100.0))


def signable_hotkeys(wallet: str):
    """Yield (hotkey_file_name, ss58) for every signable hotkey in a wallet."""
    hkd = WROOT / wallet / "hotkeys"
    if not hkd.is_dir():
        return
    for f in sorted(hkd.iterdir(), key=lambda x: (len(x.name), x.name)):
        if f.name.endswith("pub.txt"):
            continue
        try:
            d = json.loads(f.read_bytes())
            if d.get("privateKey") and d.get("ss58Address"):
                yield f.name, d["ss58Address"]
        except Exception:  # noqa: BLE001
            continue


def stake_one(wallet: str, hf: str, hk_ss58: str, tao_rao: int, cfg: dict) -> dict:
    res = {"wallet": wallet, "hotkey": hf, "ss58": hk_ss58,
           "tao": round(tao_rao / 1e9, 6), "ok": False}
    try:
        st = bt.Subtensor(network="finney")     # own connection per task
        try:
            w = bt.Wallet(name=wallet, hotkey=hf)
            if (WROOT / wallet / "coldkey").read_bytes()[:5] == b"$NACL" and cfg["password"]:
                w.coldkey_file.save_password_to_env(cfg["password"])
            _ = w.coldkey  # force decrypt now (clear error if wrong password)
            # Raw compose_call (like recycle_alpha) — bypasses the SDK add_stake's
            # internal balance/price reads, which break on this chain runtime.
            call = st.substrate.compose_call(
                "SubtensorModule", "add_stake",
                {"hotkey": hk_ss58, "netuid": cfg["netuid"], "amount_staked": int(tao_rao)},
            )
            resp = st.sign_and_send_extrinsic(
                call=call, wallet=w, sign_with="coldkey",
                wait_for_inclusion=True, wait_for_finalization=False,
            )
        finally:
            try: st.substrate.close()
            except Exception: pass
        if getattr(resp, "success", None) is False:
            res["detail"] = getattr(resp, "message", "failed")
        else:
            res["ok"] = True
            res["detail"] = getattr(resp, "message", "ok")
    except Exception as e:  # noqa: BLE001
        res["detail"] = f"{type(e).__name__}: {e}"
    return res


def stake_lane(wallet: str, jobs: list, cfg: dict) -> list:
    """All stakes for one coldkey, sequential (shared nonce)."""
    out = []
    for hf, hk, tao_rao in jobs:
        r = stake_one(wallet, hf, hk, tao_rao, cfg)
        out.append(r)
        tag = f"{r['wallet']}/{r['hotkey']}"
        print((f"  OK   {tag:<22} +{r['tao']:.4f} TAO") if r["ok"]
              else (f"  FAIL {tag:<22} {r.get('detail','')}"), flush=True)
    return out


def main() -> int:
    try:
        raw = json.loads(CONFIG_PATH.read_text())
    except FileNotFoundError:
        return _die(f"config not found: {CONFIG_PATH}")
    except json.JSONDecodeError as e:
        return _die(f"invalid JSON in {CONFIG_PATH}: {e}")

    coldkeys = raw.get("coldkeys") or []
    if not coldkeys:
        return _die("no 'coldkeys' list in stake.json")
    cfg = {
        "password": raw.get("password") or os.environ.get("WALLET_PASSWORD"),
        "netuid": int(raw.get("netuid", 11)),
    }
    target_alpha = float(raw.get("target_alpha", 50))
    buffer_pct = float(raw.get("buffer_pct", 2))
    workers = int(raw.get("max_workers", 8))
    skip_ge = raw.get("skip_if_alpha_ge")   # if set, skip hotkeys already >= this alpha
    skip_ge = float(skip_ge) if skip_ge is not None else None

    # sanitize keyfiles once (cryptoType strip, incl. encrypted coldkeys).
    # add_stake reads coldkeypub.txt too, so that must be sanitized as well.
    seen = set()
    for w in coldkeys:
        seen_paths = [WROOT / w / "coldkey", WROOT / w / "coldkeypub.txt"] \
            + [WROOT / w / "hotkeys" / hf for hf, _ in signable_hotkeys(w)]
        for p in seen_paths:
            if p not in seen:
                seen.add(p)
                TrajectoryMiner._sanitize_keyfile(p, cfg["password"])

    # live pool price -> TAO per hotkey
    try:
        tao_in, alpha_in = read_pool(cfg["netuid"])
    except Exception as e:  # noqa: BLE001
        return _die(f"could not read pool price: {e}")
    tao_rao = tao_rao_for_alpha(target_alpha, tao_in, alpha_in, buffer_pct)
    price = tao_in / alpha_in
    print(f"pool: price {price:.6f} TAO/alpha | target {target_alpha} alpha "
          f"=> {tao_rao/1e9:.4f} TAO/hotkey (incl {buffer_pct}% buffer)\n")

    # build jobs grouped by coldkey
    groups: dict[str, list] = {}
    total = 0
    for w in coldkeys:
        jobs = [(hf, hk, tao_rao) for hf, hk in signable_hotkeys(w)]
        if jobs:
            groups[w] = jobs
            total += len(jobs)
        else:
            print(f"  (no signable hotkeys in {w})")
    if not total:
        return _die("no signable hotkeys found under the listed coldkeys")

    # Optional guard: skip hotkeys already funded >= skip_if_alpha_ge (avoids
    # double-staking already-topped-up hotkeys). One batched read.
    if skip_ge is not None:
        allhk = [hk for jobs in groups.values() for (_, hk, _) in jobs]
        try:
            cur = read_alphas(allhk, cfg["netuid"])
        except Exception as e:  # noqa: BLE001
            return _die(f"skip_if_alpha_ge set but could not read current stakes: {e}")
        kept, skipped = {}, []
        for w, jobs in groups.items():
            keep = []
            for hf, hk, tr in jobs:
                a = cur.get(hk, 0.0)
                (skipped if a >= skip_ge else keep).append((w, hf, a) if a >= skip_ge else (hf, hk, tr))
            if keep:
                kept[w] = keep
        groups, total = kept, sum(len(v) for v in kept.values())
        if skipped:
            print(f"skipping {len(skipped)} hotkey(s) already >= {skip_ge} alpha:")
            for w, hf, a in skipped:
                print(f"   {w}/{hf:<6} ({a:.2f})")
            print()
        if not total:
            return _die(f"all listed hotkeys already have >= {skip_ge} alpha — nothing to stake")

    print(f"Staking ~{target_alpha} alpha to {total} hotkeys across {len(groups)} coldkeys "
          f"(~{tao_rao/1e9*total:.3f} TAO total)...\n")

    results: list = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(stake_lane, w, jobs, cfg): w for w, jobs in groups.items()}
        for fut in as_completed(futs):
            try:
                results.extend(fut.result())
            except Exception as e:  # noqa: BLE001
                print(f"  lane {futs[fut]} crashed: {e}", flush=True)

    ok = [r for r in results if r["ok"]]
    print(f"\nDone: {len(ok)}/{len(results)} staked.")
    for r in results:
        if not r["ok"]:
            print(f"  FAIL {r['wallet']}/{r['hotkey']}: {r.get('detail','')}")
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"results -> {RESULTS_PATH}")
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
