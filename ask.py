"""Answers a viewer's question about a compiled page. Used by serve.py for POST /ask.

The page keeps the conversation and runs the tools (they act on the page). This
side holds what must not be in the page: the API key, the standing rules, the
tool list, the model and its limits. One call is one round of the conversation:

    request   {"page": "<text describing the page>", "messages": [...]}
    response  one JSON object per line, as they happen:
              {"type": "text", "delta": "..."}             words of the answer
              {"type": "message", "content": [...], "stop_reason": "..."}
                                                           the finished round, to append to the history
              {"type": "error", "code": "...", "message": "..."}

When the round ends with stop_reason "tool_use", the page runs the tools, appends
their results and calls again.

Settings, from the environment or, for the first three, from the settings file:
    ANTHROPIC_API_KEY   the key (or any credential the Anthropic SDK finds by itself)
    ASK_MODEL           default claude-opus-5-5
    ASK_EFFORT          low | medium | high | xhigh | max, default medium
    ASK_FAKE=1          no model is called: a stand-in follows directives in the
                        question, so the page can be tested without a key
    ASK_ENV_FILE        where the settings file is, default ~/.config/transformer-3d/.env

The settings file holds lines like ANTHROPIC_API_KEY=sk-ant-... . It lives outside
the project folder on purpose: a file inside it would be handed out by any plain
file server started there (python3 -m http.server serves dotfiles, to the whole
local network). Keeping the key in this file, not in the shell's environment,
also keeps other programs from picking it up: Claude Code, for one, bills a key
it finds in ANTHROPIC_API_KEY instead of using the Claude plan.
"""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV_FILE = Path(os.environ.get("ASK_ENV_FILE", "~/.config/transformer-3d/.env")).expanduser()


def _load_settings():
    """Reads the settings file, if there is one. What the environment already sets wins."""
    if not ENV_FILE.is_file():
        return
    if ENV_FILE.stat().st_mode & 0o077:
        print("warning: %s can be read by other users of this computer; run: chmod 600 %s" % (ENV_FILE, ENV_FILE), file=sys.stderr)
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        name, eq, value = line.strip().removeprefix("export ").partition("=")
        name, value = name.strip(), value.strip().strip("'\"")
        if eq and value and name in ("ANTHROPIC_API_KEY", "ASK_MODEL", "ASK_EFFORT") and not os.environ.get(name):
            os.environ[name] = value


_load_settings()
RULES = (HERE / "runtime" / "ask-system.md").read_text(encoding="utf-8")
TOOLS = json.loads((HERE / "runtime" / "ask-tools.json").read_text(encoding="utf-8"))

MODEL = os.environ.get("ASK_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("ASK_EFFORT", "medium")
FAKE = os.environ.get("ASK_FAKE") == "1"
MAX_TOKENS = 16000
MAX_MESSAGES = 80  # a long conversation with tool rounds; beyond it the page must start afresh
MAX_PAGE_CHARS = 200_000

_client = None


class BadRequest(Exception):
    pass


def _error(code, message):
    return {"type": "error", "code": code, "message": message}


def check(body):
    """The parts of a request, or BadRequest saying what is wrong with it."""
    if not isinstance(body, dict):
        raise BadRequest("the request is not a JSON object")
    page, messages = body.get("page"), body.get("messages")
    if not isinstance(page, str) or not page.strip() or len(page) > MAX_PAGE_CHARS:
        raise BadRequest("'page' must be a non-empty string of at most %d characters" % MAX_PAGE_CHARS)
    if not isinstance(messages, list) or not messages or len(messages) > MAX_MESSAGES:
        raise BadRequest("'messages' must be a list of 1 to %d messages" % MAX_MESSAGES)
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            raise BadRequest("every message needs a role of 'user' or 'assistant'")
        content = m.get("content")
        blocks_ok = isinstance(content, list) and content and all(isinstance(b, dict) and isinstance(b.get("type"), str) for b in content)
        if not (isinstance(content, str) and content) and not blocks_ok:
            raise BadRequest("every message needs text or a list of content blocks")
    if messages[0]["role"] != "user" or messages[-1]["role"] != "user":
        raise BadRequest("the conversation must start and end with the viewer's side")
    return page, messages


def system_for(page):
    # The rules never change and the page's description changes only with the
    # page, so both are cached: later questions about the same page re-read them cheaply.
    return [
        {"type": "text", "text": RULES},
        {"type": "text", "text": "<page>\n%s\n</page>" % page, "cache_control": {"type": "ephemeral"}},
    ]


def echoable(content):
    """The blocks of a finished round, as they must be sent back in the history.

    After a refusal that another model took over part-way (a "fallback" block),
    thinking and tool calls from before the last such block are left out; the
    block itself stays where it is.
    """
    last = max((i for i, b in enumerate(content) if b.get("type") == "fallback"), default=-1)
    dropped = ("thinking", "redacted_thinking", "tool_use")
    return [b for i, b in enumerate(content) if i >= last or b.get("type") not in dropped]


def events(body):
    """Yields the response's events for one round."""
    try:
        page, messages = check(body)
    except BadRequest as e:
        yield _error("bad_request", str(e))
        return
    if FAKE:
        yield from _stand_in(messages)
        return

    try:
        import anthropic
    except ImportError:
        yield _error(
            "no_sdk",
            "The server's Python has no 'anthropic' package. Questions are optional and need that package and an API key: "
            "see \"Asking about a page\" in README.md. The rest of the page works without them. "
            "If you have set this up already, start the server with .venv/bin/python serve.py",
        )
        return

    global _client
    try:
        if _client is None:
            _client = anthropic.Anthropic()  # finds ANTHROPIC_API_KEY, or another credential, by itself
        if not any(getattr(_client, name, None) for name in ("api_key", "auth_token", "credentials")):
            _client = None  # look again next time: a sign-in profile can appear while the server runs
            yield _error("no_key", "The server has no API key. Put ANTHROPIC_API_KEY=... in %s, or set it where the server is started, then restart it." % ENV_FILE)
            return
        request = dict(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_for(page),
            messages=messages,
            # Inputs stream as they are written and are not checked by the API: the page checks them before running a tool.
            tools=[{**t, "eager_input_streaming": True} for t in TOOLS],
            thinking={"type": "adaptive"},
            output_config={"effort": EFFORT},
        )
        if not MODEL.startswith("claude-haiku"):
            # If the model's safety checks decline a request, the API retries it on the model Anthropic recommends for that case.
            request.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        with _client.beta.messages.stream(**request) as stream:
            for event in stream:
                if event.type == "content_block_delta" and event.delta.type == "text_delta":
                    yield {"type": "text", "delta": event.delta.text}
            final = stream.get_final_message()
    except ValueError as e:
        # A tool call whose input could not be read at all. Nothing can be answered: the viewer asks again.
        yield _error("bad_tool_input", "The model wrote a tool call that could not be read (%s)." % e)
        return
    except anthropic.AuthenticationError:
        yield _error("no_key", "The API key was rejected. Check ANTHROPIC_API_KEY (in %s, or where the server is started) and restart the server." % ENV_FILE)
        return
    except anthropic.PermissionDeniedError as e:
        yield _error("not_allowed", "This API key may not use %s: %s" % (MODEL, e.message))
        return
    except anthropic.NotFoundError:
        yield _error("no_model", "The API does not know the model '%s' (ASK_MODEL)." % MODEL)
        return
    except anthropic.RateLimitError:
        yield _error("rate_limited", "The API is limiting requests for this key right now, or its credits are used up.")
        return
    except anthropic.BadRequestError as e:
        yield _error("api_bad_request", "The API rejected the request: %s" % e.message)
        return
    except anthropic.APIStatusError as e:
        yield _error("api_error", "The API answered with an error (%d). Try again in a moment." % e.status_code)
        return
    except anthropic.APIConnectionError:
        yield _error("no_connection", "The server could not reach the API. Check the network connection.")
        return
    except anthropic.AnthropicError as e:
        # Raised before any request when no credential is found.
        yield _error("no_key", "The server has no usable API key (%s). Set ANTHROPIC_API_KEY and restart it." % e)
        return

    data = final.to_dict()
    usage = data.get("usage") or {}
    yield {
        "type": "message",
        "content": echoable(data.get("content") or []),
        "stop_reason": data.get("stop_reason"),
        "model": data.get("model"),
        "usage": {k: usage.get(k) for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")},
    }


def _stand_in(messages):
    """ASK_FAKE=1: no model. For testing the page and the round trip without a key.

    A question of the form   /tool <name> <json input>   becomes that tool call;
    after tool results it reports what it received; anything else is echoed.
    """
    last = messages[-1]["content"]
    usage = {"input_tokens": 0, "output_tokens": 0}

    def say(text, stop="end_turn", extra=()):
        words = text.split(" ")
        for i, w in enumerate(words):
            yield {"type": "text", "delta": w + (" " if i < len(words) - 1 else "")}
        yield {"type": "message", "content": [{"type": "text", "text": text}, *extra], "stop_reason": stop, "model": "stand-in", "usage": usage}

    if isinstance(last, list):
        results = [b for b in last if b.get("type") == "tool_result"]
        told = "; ".join(("ERROR " if r.get("is_error") else "") + str(r.get("content"))[:300] for r in results)
        yield from say("**Stand-in answer.** The page returned %d result(s): %s" % (len(results), told))
        return
    question = last.rsplit("<question>", 1)[-1].split("</question>")[0].strip()
    if question.startswith("/tool "):
        _, name, *rest = question.split(" ", 2)
        try:
            args = json.loads(rest[0]) if rest else {}
        except ValueError:
            args = "not json"
        call = {"type": "tool_use", "id": "toolu_standin_%d" % len(messages), "name": name, "input": args}
        yield from say("Calling `%s`." % name, "tool_use", [call])
        return
    seen = "a view block" if "<view>" in last else "no view block"
    yield from say("**Stand-in answer.** No model was called. I received %s and the question: “%s”" % (seen, question))
