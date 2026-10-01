#!/usr/bin/env python3
"""trajrl_policy: the runtime a miner's fusion policy runs in (SN11 Season 2, "fusion policies").

The validator copies this file into the policy sidecar next to the miner's pack files and runs it as a script.
It serves an OpenAI-compatible ``/v1/chat/completions`` to the Hermes harness and forwards model calls to the
validator's metering proxy (``$UPSTREAM_URL``, bearer ``$EPISODE_TOKEN``), which is the only route out of the
sidecar. Everything between those two ends is the miner's to decide.

Three ways to write a policy, all evaluated the same way:

1. ``policy.json`` only (declarative)::

       {"kind": "advisers", "writer": "kimi-k3", "advisers": ["deepseek-v4.1-flash", "qwen3.8-27b", "glm-5.3-flash"]}
       {"kind": "escalate", "cheap": "glm-5.3-flash", "strong": "kimi-k3", "signals": ["stall", "toolerr"]}
       {"kind": "pin", "model": "glm-5.3-flash"}

2. ``policy.py`` using this SDK::

       from trajrl_policy import Policy, serve
       class Mine(Policy):
           async def handle(self, req, ctx):
               req["model"] = "kimi-k3" if ctx.session["turn"] > 20 else "glm-5.3-flash"
               return await ctx.upstream(req)        # streams the writer's tokens through unchanged
       serve(Mine())

   ``ctx.call(model, messages, **gen)`` makes an extra completed call (advisers, judges, samplers);
   ``ctx.session`` is a dict that persists across the turns of one agent session; ``ctx.signals()`` reads
   stall / tool-error / no-progress signals off the transcript; ``ctx.render(message)`` returns an assembled
   assistant message as a stream; ``ctx.remaining_usd`` is what the meter says is left for this episode.

3. ``policy.py`` that ignores the SDK and serves ``/v1/chat/completions`` + ``/v1/models`` on ``$POLICY_PORT``
   itself, calling ``$UPSTREAM_URL`` with ``Authorization: Bearer $EPISODE_TOKEN``.

Env set by the validator: POLICY_PORT (8800), UPSTREAM_URL, EPISODE_TOKEN, POLICY_DIR (/policy), POLICY_LOG.
"""
from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import os
import runpy
import sys
import time
from typing import Any, Awaitable, Callable, Optional

from aiohttp import ClientSession, ClientTimeout, web

POLICY_PORT = int(os.environ.get("POLICY_PORT", "8800"))
UPSTREAM_URL = os.environ.get("UPSTREAM_URL", "http://meter:8790/v1").rstrip("/")
EPISODE_TOKEN = os.environ.get("EPISODE_TOKEN", "")
POLICY_DIR = os.environ.get("POLICY_DIR", "/policy")
POLICY_LOG = os.environ.get("POLICY_LOG", "")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "qwen3.8-27b")
# UPSTREAM_AUTH=token (default): every upstream call carries EPISODE_TOKEN (validator sidecar).
# UPSTREAM_AUTH=passthrough: the caller's own Authorization header is forwarded instead, so the same runtime
# can serve a policy as a product endpoint (Engy "auto mode"): the customer's key pays, per request.
UPSTREAM_AUTH = os.environ.get("UPSTREAM_AUTH", "token")

_log_fh = open(POLICY_LOG, "a") if POLICY_LOG else None


def log(**kw: Any) -> None:
    """Append one JSON line to the policy log (kept by the validator with the episode artifacts)."""
    kw.setdefault("ts", round(time.time(), 3))
    line = json.dumps(kw, default=str)
    if _log_fh:
        _log_fh.write(line + "\n"); _log_fh.flush()
    else:
        print(line, flush=True)


# ---------------------------------------------------------------------------
# Transcript helpers
# ---------------------------------------------------------------------------

def content_text(c: Any) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return "" if c is None else str(c)


def session_key(msgs: list[dict]) -> str:
    """Hermes sends the whole conversation every turn; its system prompt and first user turn are identical
    across scenarios, so the key is the first five messages (the two tool results that read SKILL.md and
    INSTRUCTION.md make it unique). Sessions shorter than five messages get a temporary key."""
    head = msgs[:5] if len(msgs) >= 5 else msgs[:2]
    blob = "\n".join(f"{m.get('role')}:{content_text(m.get('content'))[:6000]}" for m in head)
    return ("s" if len(msgs) >= 5 else "pre") + hashlib.sha1(blob.encode()).hexdigest()[:12]


def tool_is_error(text: str) -> bool:
    t = text[:400].lower()
    return any(k in t for k in ("traceback", "error:", "command not found", "no such file", "permission denied",
                                "exit code 1", "exit status 1", "syntaxerror", "failed"))


_DELIVERABLE_HINTS = ("write_file", "create_file", "cat >", "tee ", "> /app", ">/app", "applypatch", "git apply", "patch ")


def transcript_signals(msgs: list[dict], sess: dict, stall_s: float = 420.0,
                       k_toolerr: int = 3, n_noprog: int = 8) -> list[str]:
    """Signals a gateway can see: wall-clock stall, last finish was length, K consecutive tool errors,
    N assistant turns without a deliverable-looking action, the same tool call repeated three times."""
    fired: list[str] = []
    if time.time() - sess.get("t0", time.time()) > stall_s:
        fired.append("stall")
    if sess.get("last_finish") == "length":
        fired.append("len")
    tools = [m for m in msgs if m.get("role") == "tool"]
    if len(tools) >= k_toolerr and all(tool_is_error(content_text(m.get("content"))) for m in tools[-k_toolerr:]):
        fired.append("toolerr")
    asst = [m for m in msgs if m.get("role") == "assistant"]
    if len(asst) >= n_noprog:
        recent = asst[-n_noprog:]
        blob = json.dumps([m.get("tool_calls") or m.get("content") for m in recent], default=str).lower()
        if not any(h in blob for h in _DELIVERABLE_HINTS):
            fired.append("noprog")
    calls = [json.dumps(m.get("tool_calls"), sort_keys=True, default=str) for m in asst[-3:] if m.get("tool_calls")]
    if len(calls) == 3 and len(set(calls)) == 1:
        fired.append("repeat")
    return fired


# ---------------------------------------------------------------------------
# Upstream client
# ---------------------------------------------------------------------------

class UpstreamResponse:
    """Result of ``ctx.upstream(req)``: either a streaming pass-through or a buffered JSON body."""

    def __init__(self, status: int, headers: dict, stream: bool, first: bytes = b"", body: bytes = b"",
                 aiter=None, release: Optional[Callable[[], Awaitable[None]]] = None):
        self.status = status; self.headers = headers; self.stream = stream
        self.first = first; self.body = body; self._aiter = aiter; self._release = release
        self.usage: dict | None = None; self.finish: str | None = None

    async def chunks(self):
        yield self.first
        if self._aiter is not None:
            async for c in self._aiter:
                yield c

    async def release(self):
        if self._release:
            await self._release()


class Ctx:
    """Per-request context handed to ``Policy.handle``."""

    def __init__(self, server: "Server", msgs: list[dict], sess: dict, req_id: str, auth: str | None = None):
        self._server = server; self.messages = msgs; self.session = sess; self.req_id = req_id
        self.remaining_usd: float | None = server.remaining_usd
        self._auth = auth   # caller's Authorization header when UPSTREAM_AUTH=passthrough

    @property
    def hdr(self) -> dict:
        return self._server.hdr_for(self._auth)

    def signals(self, **kw) -> list[str]:
        return transcript_signals(self.messages, self.session, **kw)

    async def upstream(self, req: dict) -> UpstreamResponse:
        """Forward ``req`` to the meter. Streaming requests stream; the server pipes the result to Hermes."""
        return await self._server.upstream(req, self.session, self._auth)

    async def call(self, model: str, messages: list[dict], max_tokens: int = 600, temperature: float = 0.7,
                   timeout_s: float = 120.0, **gen) -> dict:
        """One completed (non-streaming) call. Returns {"text", "message", "usage", "finish", "s", "status"}."""
        body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                "stream": False, **gen}
        t0 = time.time()
        try:
            async with ClientSession(timeout=ClientTimeout(total=timeout_s)) as cs:
                async with cs.post(f"{UPSTREAM_URL}/chat/completions", json=body, headers=self.hdr) as up:
                    status = up.status; data = await up.read()
            self._server.note_budget(up.headers)
            d = json.loads(data) if status == 200 else {}
            ch = (d.get("choices") or [{}])[0]; msg = ch.get("message") or {}
            return {"text": content_text(msg.get("content")), "message": msg, "usage": d.get("usage"),
                    "finish": ch.get("finish_reason"), "s": round(time.time() - t0, 2), "status": status,
                    "error": None if status == 200 else data[:300].decode("utf-8", "replace")}
        except Exception as e:  # noqa: BLE001
            return {"text": "", "message": {}, "usage": None, "finish": None, "s": round(time.time() - t0, 2),
                    "status": 0, "error": str(e)}

    def render(self, message: dict, model: str = "auto", finish: str = "stop", usage: dict | None = None,
               stream: bool = True) -> "Rendered":
        """Return an assembled assistant message to Hermes, as SSE when it asked for a stream."""
        return Rendered(message, model, finish, usage, stream, self.req_id)


class Rendered:
    def __init__(self, message: dict, model: str, finish: str, usage: dict | None, stream: bool, req_id: str):
        self.message = message; self.model = model; self.finish = finish; self.usage = usage
        self.stream = stream; self.req_id = req_id

    def sse(self) -> bytes:
        base = {"id": self.req_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": self.model}
        out = []
        content = self.message.get("content") or ""
        if content:
            out.append({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}]})
        if self.message.get("tool_calls"):
            tcs = []
            for i, tc in enumerate(self.message["tool_calls"]):
                tcs.append({"index": i, "id": tc.get("id"), "type": "function",
                            "function": {"name": tc["function"]["name"], "arguments": tc["function"].get("arguments", "")}})
            out.append({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": tcs}, "finish_reason": None}]})
        out.append({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": self.finish}], "usage": self.usage})
        return b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in out) + b"data: [DONE]\n\n"

    def json(self) -> bytes:
        return json.dumps({"id": self.req_id, "object": "chat.completion", "created": int(time.time()), "model": self.model,
                           "choices": [{"index": 0, "message": {"role": "assistant", **self.message}, "finish_reason": self.finish}],
                           "usage": self.usage}).encode()


# ---------------------------------------------------------------------------
# Policy base class and reference policies
# ---------------------------------------------------------------------------

class Policy:
    """Subclass and implement ``handle``. Return ``await ctx.upstream(req)`` to stream a model's answer, or
    ``ctx.render(message)`` to answer with a message you assembled."""

    name = "policy"

    async def handle(self, req: dict, ctx: Ctx):
        req["model"] = DEFAULT_MODEL
        return await ctx.upstream(req)


class PinPolicy(Policy):
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model; self.name = f"pin:{model}"

    async def handle(self, req, ctx):
        req["model"] = self.model
        return await ctx.upstream(req)


class EscalatePolicy(Policy):
    """Start cheap; on any of the configured signals switch to the strong model and stay there."""

    def __init__(self, cheap: str, strong: str, signals: list[str] | None = None, stall_s: float = 420.0):
        self.cheap, self.strong = cheap, strong
        self.signals = set(signals or ["stall", "toolerr", "repeat"]); self.stall_s = stall_s
        self.name = f"escalate:{cheap}->{strong}"

    async def handle(self, req, ctx):
        sess = ctx.session
        if not sess.get("escalated"):
            fired = [s for s in ctx.signals(stall_s=self.stall_s) if s in self.signals]
            if fired:
                sess["escalated"] = True; sess["why"] = fired
        req["model"] = self.strong if sess.get("escalated") else self.cheap
        return await ctx.upstream(req)


ADVISER_PROMPT = ("You are an adviser watching another agent solve a terminal task. Read the conversation so far and "
                  "write at most 150 words of concrete, specific advice for its NEXT step only: what to check, what "
                  "to run, what mistake to avoid, or what the deliverable must contain. No preamble. If the agent is "
                  "on track, say so in one line.")


class AdvisersPolicy(Policy):
    """N advisers read the conversation and write short notes; one writer answers with the notes appended as a
    trailing system message. Advisers engage only once tool results exist (the agent has started acting)."""

    def __init__(self, writer: str, advisers: list[str], max_words: int = 150, temperature: float = 0.7,
                 max_tokens: int = 400, every: int = 1):
        self.writer, self.advisers = writer, list(advisers)
        self.temperature, self.max_tokens, self.every = temperature, max_tokens, max(1, every)
        self.name = f"advisers:{writer}|{','.join(advisers)}"

    async def handle(self, req, ctx):
        msgs = req.get("messages") or []
        turn = ctx.session.get("turn", 0)
        if any(m.get("role") == "tool" for m in msgs) and turn % self.every == 0:
            convo = [{"role": "system", "content": ADVISER_PROMPT}] + [
                {"role": "user", "content": "Conversation so far (JSON):\n" + json.dumps(msgs[1:], default=str)[-60000:]}]
            notes = await asyncio.gather(*[ctx.call(m, convo, max_tokens=self.max_tokens, temperature=self.temperature)
                                           for m in self.advisers])
            good = [(m, n["text"].strip()) for m, n in zip(self.advisers, notes) if n["text"].strip()]
            if good:
                block = "\n\n".join(f"Adviser {i + 1} ({m}):\n{t}" for i, (m, t) in enumerate(good))
                req["messages"] = msgs + [{"role": "system", "content":
                    "Notes from independent advisers who read the conversation so far. They may be wrong; weigh them "
                    "with your own judgment, then take the next step.\n\n" + block}]
            log(event="advisers", n=len(good), s=[n["s"] for n in notes], status=[n["status"] for n in notes])
        req["model"] = self.writer
        return await ctx.upstream(req)


def policy_from_config(cfg: dict) -> Policy:
    kind = (cfg.get("kind") or "pin").lower()
    if kind == "pin":
        return PinPolicy(cfg.get("model", DEFAULT_MODEL))
    if kind == "escalate":
        return EscalatePolicy(cfg["cheap"], cfg["strong"], cfg.get("signals"), float(cfg.get("stall_s", 420)))
    if kind == "advisers":
        return AdvisersPolicy(cfg["writer"], cfg["advisers"], temperature=float(cfg.get("temperature", 0.7)),
                              max_tokens=int(cfg.get("max_tokens", 400)), every=int(cfg.get("every", 1)))
    raise ValueError(f"unknown policy kind {kind!r}")


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class Server:
    def __init__(self, policy: Policy):
        self.policy = policy
        self.hdr = {"Authorization": f"Bearer {EPISODE_TOKEN}", "Content-Type": "application/json"}
        self.sessions: dict[str, dict] = {}
        self.remaining_usd: float | None = None
        self.app = web.Application(client_max_size=64 * 1024 * 1024)
        self.app.router.add_post("/v1/chat/completions", self.chat)
        self.app.router.add_get("/v1/models", self.models)
        self.app.router.add_get("/health", self.health)

    def hdr_for(self, auth: str | None) -> dict:
        """Upstream headers: the episode token, or the caller's own Authorization in passthrough mode."""
        if UPSTREAM_AUTH == "passthrough" and auth:
            return {"Authorization": auth, "Content-Type": "application/json"}
        return self.hdr

    def note_budget(self, headers) -> None:
        v = headers.get("x-trajrl-budget-remaining-usd")
        if v is not None:
            try:
                self.remaining_usd = float(v)
            except ValueError:
                pass

    async def refresh_budget(self) -> None:
        """Ask the meter for the authoritative remaining budget (a streamed
        response's header is computed before the call)."""
        if UPSTREAM_AUTH == "passthrough":
            return   # no per-episode budget in product mode
        try:
            async with ClientSession(timeout=ClientTimeout(total=5)) as cs:
                async with cs.get(f"{UPSTREAM_URL}/budget", headers=self.hdr) as r:
                    if r.status == 200:
                        self.remaining_usd = float((await r.json()).get("remaining_usd"))
        except Exception:  # noqa: BLE001
            pass

    async def upstream(self, req: dict, sess: dict, auth: str | None = None) -> UpstreamResponse:
        stream = bool(req.get("stream"))
        if stream:
            req.setdefault("stream_options", {})["include_usage"] = True
        retries = 0
        hdr = self.hdr_for(auth)
        cs = ClientSession(timeout=ClientTimeout(total=900))
        while True:
            up = await cs.post(f"{UPSTREAM_URL}/chat/completions", json=req, headers=hdr)
            self.note_budget(up.headers)
            if up.status in (429, 503) and retries < 6:
                await up.read(); up.release(); retries += 1; await asyncio.sleep(1.5 * retries); continue
            if up.status != 200 or not stream:
                body = await up.read(); await cs.close()
                r = UpstreamResponse(up.status, dict(up.headers), False, body=body)
                if up.status == 200:
                    try:
                        d = json.loads(body); r.usage = d.get("usage")
                        r.finish = (d.get("choices") or [{}])[0].get("finish_reason")
                    except Exception:  # noqa: BLE001
                        pass
                return r
            first = await up.content.readany()
            if (b'"error"' in first[:400] or b"concurrency" in first[:400]) and retries < 8:
                up.close(); retries += 1; await asyncio.sleep(2.0 * retries); continue   # drop the pooled connection

            async def _release(up=up, cs=cs):
                up.release(); await cs.close()

            return UpstreamResponse(200, dict(up.headers), True, first=first, aiter=up.content.iter_any(), release=_release)

    async def chat(self, request: web.Request):
        t0 = time.time()
        body = await request.json(); msgs = body.get("messages") or []
        sk = session_key(msgs); sess = self.sessions.setdefault(sk, {"t0": time.time(), "turn": 0})
        sess["turn"] += 1
        stream = bool(body.get("stream")); req_id = f"chatcmpl-{sk}-{sess['turn']}"
        ctx = Ctx(self, msgs, sess, req_id, auth=request.headers.get("Authorization"))
        try:
            result = await self.policy.handle(body, ctx)
        except Exception as e:  # noqa: BLE001
            log(event="policy_error", session=sk, turn=sess["turn"], error=repr(e))
            return web.json_response({"error": {"message": f"policy error: {e!r}", "type": "policy"}}, status=500)
        model = body.get("model")
        if isinstance(result, Rendered):
            sess["last_finish"] = result.finish
            log(event="turn", session=sk, turn=sess["turn"], model=result.model, rendered=True, s=round(time.time() - t0, 2))
            if stream:
                resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
                await resp.prepare(request); await resp.write(result.sse()); await resp.write_eof(); return resp
            return web.Response(body=result.json(), content_type="application/json")
        if not isinstance(result, UpstreamResponse):
            return web.json_response({"error": {"message": "policy returned neither upstream() nor render()", "type": "policy"}}, status=500)
        if not result.stream:
            sess["last_finish"] = result.finish
            log(event="turn", session=sk, turn=sess["turn"], model=model, status=result.status, finish=result.finish,
                usage=result.usage, s=round(time.time() - t0, 2))
            return web.Response(body=result.body, status=result.status,
                                content_type=(result.headers.get("Content-Type") or "application/json").split(";")[0])
        resp = web.StreamResponse(status=200, headers={"Content-Type": result.headers.get("Content-Type", "text/event-stream"),
                                                       "Cache-Control": "no-cache"})
        await resp.prepare(request)
        buf = b""; usage = None; finish = None
        try:
            async for chunk in result.chunks():
                await resp.write(chunk); buf += chunk
                while b"\n\n" in buf:
                    ev, buf = buf.split(b"\n\n", 1)
                    for line in ev.split(b"\n"):
                        if line.startswith(b"data: ") and line != b"data: [DONE]":
                            try:
                                j = json.loads(line[6:])
                                if j.get("usage"):
                                    usage = j["usage"]
                                for ch in j.get("choices") or []:
                                    if ch.get("finish_reason"):
                                        finish = ch["finish_reason"]
                            except Exception:  # noqa: BLE001
                                pass
            await resp.write_eof()
        finally:
            await result.release()
        await self.refresh_budget()
        sess["last_finish"] = finish
        log(event="turn", session=sk, turn=sess["turn"], model=model, status=200, finish=finish, usage=usage,
            s=round(time.time() - t0, 2), remaining_usd=self.remaining_usd)
        return resp

    async def models(self, request):
        return web.json_response({"object": "list", "data": [{"id": "auto", "object": "model", "owned_by": "policy",
                                                              "policy": self.policy.name}]})

    async def health(self, request):
        return web.json_response({"ok": True, "policy": self.policy.name})


def serve(policy: Policy, port: int = POLICY_PORT) -> None:
    """Run the policy server (blocking). Call this at the end of policy.py."""
    srv = Server(policy)
    log(event="start", policy=policy.name, port=port, upstream=UPSTREAM_URL)
    web.run_app(srv.app, host="0.0.0.0", port=port, print=None, access_log=None)


def main() -> None:
    """Sidecar entry: run the miner's policy.py if present, else the declarative policy.json, else the default pin."""
    sys.path.insert(0, POLICY_DIR); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    py = os.path.join(POLICY_DIR, "policy.py"); js = os.path.join(POLICY_DIR, "policy.json")
    if os.path.exists(py):
        runpy.run_path(py, run_name="__main__")   # the policy calls serve() itself (or runs its own server)
        return
    if os.path.exists(js):
        with open(js) as fh:
            serve(policy_from_config(json.load(fh)))
        return
    serve(PinPolicy(DEFAULT_MODEL))


if __name__ == "__main__":
    main()
