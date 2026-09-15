#!/usr/bin/env python3
"""Monitor real ``recycle_alpha`` fee burns on SN11 — including other miners'.

Every submission fee is a public on-chain ``SubtensorModule.recycle_alpha``
extrinsic. Its ``amount`` **call argument** is attacker-controlled and proves
nothing; the truth is the ``SubtensorModule.AlphaRecycled(coldkey, hotkey,
amount, netuid)`` **event**, which records the alpha actually destroyed. This
script reads both and flags any burn where actual < the required fee.

(That gap is exactly the fee-bypass bug: `recycle_alpha` saturates — an empty
coldkey burns 0 alpha yet the extrinsic still succeeds with `amount=50` in the
args. The server now validates the event, but this lets you verify independently.)

Usage
-----
    python trajectoryRL/scripts/watch_burns.py              # last 300 blocks
    python trajectoryRL/scripts/watch_burns.py 1000         # last 1000 blocks
    python trajectoryRL/scripts/watch_burns.py follow       # live, tail new blocks

Notes
-----
* Uses its own SubstrateInterface (no bittensor SDK), so it is unaffected by the
  SDK/runtime NetUid mismatch that breaks stake queries in the project venv.
* Blocks older than ~a few hours need an archive endpoint (auto-selected).
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

from async_substrate_interface.sync_substrate import SubstrateInterface

NETUID = 11
FEE_RAO = 50_000_000_000          # 50 alpha
LIVE = "wss://entrypoint-finney.opentensor.ai:443"
ARCHIVE = "wss://archive.chain.opentensor.ai"


def _uid_map() -> dict:
    """hotkey -> uid, from the public dashboard API (avoids a metagraph sync)."""
    try:
        with urllib.request.urlopen("https://trajrl.com/api/miners", timeout=20) as r:
            return {m["hotkey"]: m["uid"] for m in json.load(r)["miners"]}
    except Exception:
        return {}


def _own_hotkeys() -> set:
    """Local wallet hotkeys, so we can label ours vs. other miners'."""
    out = set()
    root = Path.home() / ".bittensor" / "wallets"
    if not root.is_dir():
        return out
    for wdir in root.iterdir():
        hkd = wdir / "hotkeys"
        if not hkd.is_dir():
            continue
        for f in hkd.iterdir():
            if f.name.endswith("pub.txt"):
                continue
            try:
                d = json.loads(f.read_bytes())
                if d.get("ss58Address"):
                    out.add(d["ss58Address"])
            except Exception:
                pass
    return out


def _connect(url: str) -> SubstrateInterface:
    last = None
    for _ in range(3):
        try:
            s = SubstrateInterface(url=url)
            s.get_block_hash(None)
            return s
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2)
    raise RuntimeError(f"cannot connect to {url}: {last}")


def scan_block(sub, bn: int) -> list:
    """Return every recycle_alpha in block `bn` with claimed + actual amounts."""
    out = []
    bh = sub.get_block_hash(bn)
    blk = sub.get_block(block_hash=bh)
    hits = []
    for i, ext in enumerate(blk.get("extrinsics", [])):
        v = getattr(ext, "value", {})
        if not isinstance(v, dict):
            continue
        call = v.get("call", {})
        if call.get("call_function") == "recycle_alpha":
            args = {a["name"]: a["value"] for a in call.get("call_args", [])}
            if int(args.get("netuid", -1)) == NETUID:
                hits.append((i, v.get("address"), args))
    if not hits:
        return out
    events = list(sub.get_events(block_hash=bh))       # only when needed
    for idx, signer, args in hits:
        actual, ok = None, None
        for e in events:
            ev = e.value if hasattr(e, "value") else e
            if ev.get("extrinsic_idx") != idx:
                continue
            inner = ev.get("event", {}) or {}
            eid = inner.get("event_id") or ev.get("event_id")
            attrs = inner.get("attributes") if inner else ev.get("attributes")
            if eid == "AlphaRecycled":
                try:
                    actual = attrs[2] if isinstance(attrs, (list, tuple)) else attrs.get("amount")
                except Exception:
                    actual = None
            elif eid == "ExtrinsicSuccess":
                ok = True
            elif eid == "ExtrinsicFailed":
                ok = False
        out.append({"block": bn, "idx": idx, "coldkey": signer,
                    "hotkey": args.get("hotkey"),
                    "claimed": int(args.get("amount") or 0),
                    "actual": int(actual or 0), "success": ok})
    return out


def report(rows: list, uids: dict, mine: set) -> None:
    if not rows:
        print("  (no recycle_alpha burns found in range)")
        return
    print(f"{'block-idx':<14} {'UID':>4} {'hotkey':<14} {'claimed':>9} {'ACTUAL':>9}  verdict")
    for r in sorted(rows, key=lambda x: (x["block"], x["idx"])):
        uid = uids.get(r["hotkey"])
        who = "MINE" if r["hotkey"] in mine else "other"
        verdict = "OK" if r["actual"] >= FEE_RAO else "*** UNDERPAID ***"
        print(f"{str(r['block'])+'-'+str(r['idx']):<14} {str(uid):>4} "
              f"{(r['hotkey'] or '')[:12]+'..':<14} {r['claimed']/1e9:>9.2f} "
              f"{r['actual']/1e9:>9.4f}  {verdict} [{who}]")
    under = [r for r in rows if r["actual"] < FEE_RAO]
    print(f"\n  total burns: {len(rows)} | full 50a: {len(rows)-len(under)} | UNDERPAID: {len(under)}")
    if under:
        print("  underpaid hotkeys:", sorted({(uids.get(r['hotkey']), (r['hotkey'] or '')[:12]) for r in under}))


def main() -> int:
    arg = sys.argv[1] if len(sys.argv) > 1 else "300"
    uids, mine = _uid_map(), _own_hotkeys()

    if arg == "follow":
        sub = _connect(LIVE)
        head = sub.get_block_number(sub.get_chain_head())
        print(f"following new blocks from {head} (Ctrl-C to stop)\n")
        seen = head
        while True:
            cur = sub.get_block_number(sub.get_chain_head())
            for bn in range(seen + 1, cur + 1):
                for r in scan_block(sub, bn):
                    report([r], uids, mine)
            seen = cur
            time.sleep(6)

    span = int(arg)
    sub = _connect(LIVE)
    head = sub.get_block_number(sub.get_chain_head())
    lo, hi = head - span, head
    # older ranges need the archive node
    if span > 250:
        sub = _connect(ARCHIVE)
    print(f"scanning blocks {lo}..{hi} ({span}) for netuid-{NETUID} recycle_alpha "
          f"(~{span*2//60} min)...\n")
    rows = []
    for n, bn in enumerate(range(lo, hi + 1), 1):
        try:
            rows.extend(scan_block(sub, bn))
        except Exception:
            continue
        if n % 100 == 0:
            print(f"  ...{n}/{span} blocks, {len(rows)} burns so far", flush=True)
    report(rows, uids, mine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
