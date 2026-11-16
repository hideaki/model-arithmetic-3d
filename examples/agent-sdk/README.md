# The add-model skill outside the Claude desktop app

An example, not a product. It shows that the authoring loop of this project (research, proposal, wait for the owner, program, tests, a look at the page) does not depend on the Claude desktop app. One file, `add-model.mjs`, runs the project's own skill through the Claude Agent SDK from a terminal.

Nothing here is used by the compiler, the pages or the tests. Deleting this folder changes nothing else.

## Run it

```bash
cd examples/agent-sdk
npm install
node add-model.mjs --check
node add-model.mjs "graph attention network (Veličković et al., 2018), one layer"
```

- `--check` makes one small call and prints whether the key, the skill and the browser were found.
- With a model description, the program sends `/add-model <description>`. The agent researches, prints its proposal and stops. Type your answer at `you>`; one line, Enter to send. `quit` ends the run.
- `--say "<text>"` sends any first message instead, for example to continue work on an existing page.

It needs Node 18.14 or later, Google Chrome (for the agent's look at the page), and an Anthropic API key with credit. The key is taken from `ANTHROPIC_API_KEY`, or from the same settings file the question panel uses (`~/.config/transformer-3d/.env`). Calls are billed to that key, per token. The program prints a running cost estimate after every turn.

## What the agent may do

Without asking:

- read files inside the project, search the web and fetch pages;
- write under `programs/` and `test/`, and edit `HANDOFF.md`;
- run `node tools/describe.mjs`, `npm test` and `node --test`;
- drive a headless Chrome (to look at `http://localhost:5174/compiled.html?program=…`).

Shell commands that only read (`ls`, `wc`, `grep` and the like) inside the project also run without asking; that is the SDK's own rule.

Everything else is shown to you first: any other shell command, and any change to another file. So an edit under `lang/` or `runtime/`, which the skill allows only with the owner's approval, is shown to you and waits for your `y`. (This covers the agent's edit tools. See the first limit below for what it does not cover.) Your choices at the prompt are `y`, `a` (yes, and do not ask again for this kind of call during this run), `n`, or a sentence saying what to do instead, which is passed to the agent.

Never, whatever you answer: reading the settings file that holds the key, any call that names the key, and a shell command that prints the whole environment. A hook refuses these before any other rule is looked at.

## What differs from the desktop app

- **The page server.** The program uses the server on port 5174 if one is running. Otherwise it starts one and stops it at exit. It never touches a server it did not start.
- **The look at the page.** The skill's step 6 was written for the app's browser pane. Here the agent gets Playwright's browser tools, and the program tells it so in a few lines added to its instructions (`SETTING` in the source).
- **Waiting for the owner.** The agent ends its turn with the proposal; your next line is the answer.

## Limits

- The approval prompt is not a sandbox. The agent may write files under `programs/` and `test/` and run them (`npm test`, `describe`) without asking, and code that runs can change any file or read the environment. An agent following the skill has no reason to do that, but nothing here prevents it. `git status` after a run shows every file that changed.
- Shell commands the agent runs inherit this program's environment, which holds the key. The hook above is a guard against the plain ways of printing it, not a guarantee; read the commands you are asked to approve.
- Started from a terminal inside Claude Code or the Claude desktop app, the program leaves that session's `CLAUDE…` variables out of what it hands to the agent. One of them, `CLAUDE_CODE_ENTRYPOINT=claude-desktop`, makes the API refuse every request made with an API key: HTTP 400, "Your credit balance is too low", whatever the balance is. Found on 2026-11-15 by trying each variable alone; the SDK's own value (`sdk-ts`) and `cli` are accepted, any other value tried was refused.
- One line per answer. There is no way to paste a multi-line reply.
- A conversation cannot be resumed after the program exits.
- Each run prints one warning from the SDK, `CLAUDE_SDK_CAN_USE_TOOL_SHADOWED`. It says that the tools in the allowed list are not shown to you, which is intended.

## State (2026-11-15)

Checked, with three small runs against the real API (about $0.35 together):

- it starts, finds the key and the skill, and connects the browser;
- it starts and stops its own page server on a free port, and leaves a running one alone;
- a write outside the allowed list is shown for approval, and a typed refusal reaches the agent;
- a command that names the key, and one that prints the environment, are refused;
- the agent opens a page of this project in headless Chrome, takes a screenshot and reads the console.

Not checked: a run of the skill itself. No model has been added through this program yet; that needs the owner at the terminal to answer the proposal. The `a` answer and the agent's own multiple-choice questions have not come up in a run either.
