"""Example fusion policy written against the trajrl_policy SDK.

Cheap writer by default; once the agent has taken more than 12 turns, or a stall / tool-error signal fires,
switch the session to the strong model and ask one cheap adviser for a note on every strong turn.
"""
from trajrl_policy import Policy, serve

CHEAP, STRONG, ADVISER = "glm-5.3-flash", "kimi-k3", "deepseek-v4.1-flash"

ADVICE = ("Read the conversation and write at most 120 words of concrete advice for the agent's NEXT step "
          "only. No preamble.")


class Custom(Policy):
    name = "example:cheap-then-strong+adviser"

    async def handle(self, req, ctx):
        sess = ctx.session                      # persists across the turns of one agent session
        msgs = req.get("messages") or []
        if not sess.get("strong"):
            if sess["turn"] > 12 or any(s in ("stall", "toolerr", "repeat") for s in ctx.signals()):
                sess["strong"] = True
        if sess.get("strong"):
            note = await ctx.call(ADVISER, [{"role": "system", "content": ADVICE},
                                            {"role": "user", "content": str(msgs[1:])[-40000:]}], max_tokens=300)
            if note["text"]:
                req["messages"] = msgs + [{"role": "system", "content": "Adviser note (may be wrong): " + note["text"]}]
            req["model"] = STRONG
        else:
            req["model"] = CHEAP
        return await ctx.upstream(req)          # streams the writer's tokens straight through to Hermes


serve(Custom())
