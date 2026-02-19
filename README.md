# Bitdoze Bot (Agno + Discord)

A Discord-first AI agent powered by [Agno](https://github.com/agno-agi/agno), with streaming responses, vector knowledge base, self-improvement, team orchestration, and a rich toolkit ecosystem.

## Features

### Core
- **Streaming responses** — real-time message updates in Discord as the agent thinks
- **Agno toolkits** — WebSearch, HackerNews, Website, GitHub, YouTube, File, Shell, Reddit, Discord
- **Agno Teams** — native team orchestration with member delegation
- **Team observability** — logs selected members, delegation paths, run timing, and metrics
- **Mention-first** — responds when tagged with `@Bot`

### Memory & Knowledge
- **Memory** — SQLite-backed with automatic updates, session summaries, and chat history
- **Cognee memory (optional)** — long-term external memory with conversation chunking, metadata-rich ingestion, and retrieval fallback across payload/path variants
- **Learning** — Agno LearningMachine per agent (user profile, memory, session context, entity, learned knowledge)
- **Knowledge base** — vector search with LanceDb (file-based) or PgVector (PostgreSQL) backends
- **Self-improvement** — discovery tools let the agent save and search its own learnings over time
- **Soul + Heartbeat** — persona via `workspace/SOUL.md`, proactive heartbeat via `workspace/HEARTBEAT.md`

### Infrastructure
- **Docker support** — optional PostgreSQL + PgVector and Cognee via `docker-compose.yml`
- **Structured logging** — config-driven stdout + optional rotating file logs
- **Tool permissions** — runtime allow/deny rules with JSONL audit logging
- **Extensible** — add agents via config or folder-based workspace agents

## Quick Start

### 1. Install dependencies

```bash
# Using uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### 2. Run the setup wizard

```bash
python scripts/setup_bot.py
```

The wizard creates `~/.bitdoze-bot/` (or `$BITDOZE_BOT_HOME`) with:
- `config.yaml` — main configuration
- `.env` — secrets (tokens, API keys)
- `workspace/` — SOUL.md, AGENTS.md, USER.md, HEARTBEAT.md, CRON.yaml
- `workspace/agents/` — folder-based agent definitions
- `workspace/knowledge/` — documents for the knowledge base
- `skills/`, `logs/`, `data/`
- `bitdoze-bot.service` — systemd unit file

Wizard prompts:
- **Required**: Discord bot token
- **Optional**: GitHub token, model/API settings, systemd service install
- **Optional**: PostgreSQL + PgVector setup via Docker (for knowledge base)
- **Optional**: Knowledge backend selection (LanceDb or PgVector)

### 3. (Optional) Start PgVector + Cognee

If you chose PgVector/Cognee during setup, or want to start them manually:

```bash
docker compose up -d
```

This starts:
- PostgreSQL 17 with pgvector on port `5532`, storing data at `~/.bitdoze-bot/data/pgdata`
- Cognee API on `127.0.0.1:8000` (host-loopback only), storing data at `~/.bitdoze-bot/data/cognee`

### 4. (Optional) Setup knowledge base

```bash
# Initialize and load documents from workspace/knowledge/
python scripts/setup_knowledge.py

# Or with custom options
python scripts/setup_knowledge.py --backend pgvector --docs-dir /path/to/docs
```

Add `.md`, `.txt`, or `.pdf` files to `~/.bitdoze-bot/workspace/knowledge/` and re-run the script to ingest them.

### 5. (Optional) Generate SOUL.md

```bash
python scripts/generate_soul.py
# or force overwrite: python scripts/generate_soul.py --force
```

Generates a comprehensive personality and self-improvement template at `~/.bitdoze-bot/workspace/SOUL.md`.

### 6. Review and run

```bash
$EDITOR ~/.bitdoze-bot/.env
$EDITOR ~/.bitdoze-bot/config.yaml
python main.py
```

## Discord Requirements
- Enable the **Message Content Intent** in the Discord Developer Portal for your bot.
- Invite the bot with the right permissions to post messages in your target channels.

## Configuration Overview
Edit `~/.bitdoze-bot/config.yaml` (or `config.yml` if that is your active file):
- Relative paths are resolved from the config file location.
- Override home location with `BITDOZE_BOT_HOME=/custom/path`.
- Explicit overrides still work: `python main.py --config /path/config.yaml --env-file /path/.env`.

### Model
- `model`: provider, model id, base URL, and API key env var
- `model.structured_outputs`: set `false` for providers that reject response_format (e.g., StepFun/OpenRouter)

### Discord & Streaming
- `discord`: bot token env var
- `discord.access_control`: optional ingress allowlists (`allowed_user_ids`, `allowed_channel_ids`, `allowed_guild_ids`, `allowed_role_ids`)
- `runtime.streaming_enabled`: stream responses to Discord with real-time message edits (default: `true`)
- `runtime.streaming_edit_interval`: seconds between message edits during streaming (default: `1.5`)

Streaming progressively edits the Discord message as the agent generates content. It falls back to non-streaming for team runs and research mode.

### Runtime
- `runtime`: timeouts for agent runs, cron, heartbeat, and max concurrency
- `runtime.slow_run_threshold_seconds`: sends an interim "still working" reply when complex runs take longer than expected
- `runtime.session_id_strategy`: session partitioning for runs/history (`channel`, `user`, `channel_user`; default: `channel_user`)

### Knowledge Base
- `knowledge.enabled`: activate vector search knowledge base
- `knowledge.backend`: `lancedb` (file-based, zero setup) or `pgvector` (requires PostgreSQL)
- `knowledge.embedder`: embedding model (default: `text-embedding-3-small`)
- LanceDb settings: `lance_uri`, `table_name`, `learnings_table_name`
- PgVector settings: `db_url` (or `PGVECTOR_DB_URL` env var)

```yaml
knowledge:
  enabled: true
  backend: lancedb         # or pgvector
  embedder: text-embedding-3-small
  lance_uri: data/lancedb
  table_name: bitdoze_knowledge
  learnings_table_name: bitdoze_learnings
  # db_url: postgresql+psycopg://bitdoze:secret@localhost:5532/bitdoze  # for pgvector
```

### Self-Improvement (Discovery Tools)
The agent can build and query its own knowledge base using the `discoveries` toolkit:
- `save_discovery`: save a reusable learning (corrections, preferences, patterns)
- `search_discoveries`: search past learnings before answering

Add `discoveries` to an agent's tools list to enable:

```yaml
agents:
  definitions:
    - name: main
      tools: [web_search, website, file, discoveries]
```

Combined with `learned_knowledge: agentic` in learning config, the agent decides when to save and recall learnings automatically.

### Memory
- `memory`: `mode: automatic` (best capture), SQLite db path, history + summaries (custom prompt supported)
- `memory.summary_prompt`: customizable session summary with support for `decisions` and `unresolved` keys
- `memory.cognee`: optional external long-term memory (Cognee API)
- `memory.cognee.auto_sync_conversations`: when true, each successful Discord user/assistant turn is stored in Cognee as a compact summary plus chunked user/assistant entries (better semantic recall on long turns)
- `memory.cognee.auto_recall_enabled`: when true, each incoming message performs Cognee recall and injects top matches into system context
- Cognee ingestion includes metadata (user/session/agent/channel/guild/timestamps), plus duplicate suppression for recently repeated content
- Cognee retrieval tries multiple compatible payload/path shapes and skips empty 2xx responses to reduce false misses

Recommended Cognee config:

```yaml
memory:
  cognee:
    enabled: true
    base_url: http://localhost:8000
    user: bitdoze-bot@example.com
    dataset: bitdoze-user-profile
    auto_sync_conversations: true
    auto_recall_enabled: true
    auto_recall_limit: 5
    auto_recall_timeout_seconds: 3
    auto_recall_max_chars: 2000
    auto_recall_inject_all: false
    timeout_seconds: 8
    max_turn_chars: 6000
    auth_token_env: COGNEE_API_TOKEN
```

#### Cognee Troubleshooting (Ingest + Recall)

Use this quick checklist when memory does not appear to work:

1. Verify config is enabled:
   - `memory.cognee.enabled: true`
   - `memory.cognee.auto_sync_conversations: true`
   - `memory.cognee.auto_recall_enabled: true`

2. Verify Cognee API is reachable:
```bash
curl -sS http://localhost:8000/api/v1/datasets
```
Expected: JSON response (list/object), not connection refused.

3. Verify ingest is happening:
   - Send one Discord message to the bot and wait for the reply.
   - Check logs for success lines containing:
     - `Cognee add_memory success`
     - `Cognee save_conversation_turn status=ok`

4. Verify recall is happening:
   - Ask a follow-up question that references the previous message.
   - Check logs for:
     - `Cognee search success`
     - `Cognee auto-recall injected items=...`

5. If recall is weak:
   - Increase `memory.cognee.auto_recall_limit` (for example `7` to `10`).
   - Increase `memory.cognee.auto_recall_timeout_seconds` if Cognee search is timing out.
   - Increase `memory.cognee.auto_recall_max_chars` (for example `2500` to `4000`).
   - Set `memory.cognee.auto_recall_inject_all: true` to inject full matched items (no per-item truncation and no total char cap).
   - Ensure your question includes clear keywords from the earlier memory.

Notes:
- Dataset creation retries automatically after transient failures.
- Conversation turns are stored as a summary plus chunks for better semantic retrieval on long messages.
- Recent identical memory payloads are deduplicated to reduce noise.

### Learning
- `learning`: enable Agno LearningMachine and set learning modes (`always`, `agentic`, `propose`, `hitl`)
- `learning.stores.learned_knowledge: agentic` — agent decides when to save/search learnings (recommended)

### Other
- `monitoring`: JSONL telemetry for runs + heartbeat watchdog alerts for long-running active tasks
- `tool_fallback.denied_tools`: tool names blocked during XML-style fallback execution (default: `shell`, `discord`)
- `logging`: set level, format, and rotating file settings from YAML
- `tool_permissions`: runtime allow/deny rules for tool use plus JSONL audit logging
- `toolkits`: enable/disable web search, hackernews, website, github, youtube, file, shell, reddit, discord, cognee tools
- `agents.workspace_dir`: folder-based agent loading from `workspace/agents/<name>/`
- `teams`: native Agno Team definitions, delegation behavior, team memory options, and default team
- `heartbeat`: 30-min cadence, optional channel override, `session_scope` (`isolated` to avoid heartbeat history growth), and optional dedicated `agent`
- `cron`: schedule jobs via `workspace/CRON.yaml`
- `agents`: define multiple agents, per-agent tool selections, and routing rules
- `skills`: optional skill packs loaded from `skills/`
- `context`: enable datetime injection and set timezone
- `context.use_workspace_context_files`: when `false`, skip `USER.md`/daily logs/`MEMORY.md` injection and rely on Agno DB memory+learning only
- `context.agents_path`: load workspace instructions into system context
- `context.user_path` + `context.memory_dir` + `context.long_memory_path`: load USER + daily memory + long memory
- `context.main_session_scope`: `dm_only` (default) or `always` for long memory
- `context.scope_workspace_context_by_tenant`: isolate workspace context files by guild/user (default: `true`)
- `context.scoped_context_dir`: root folder for tenant-isolated context files (default: `workspace/context`)
- `context.allow_global_context_in_guilds`: if `false`, global USER/MEMORY files are not injected in guild messages when tenant scoping is disabled

### Learning Modes (Agno)
Configure learning in `config.yaml`:

```yaml
learning:
  enabled: true
  mode: always # default mode for enabled stores
  stores:
    user_profile: true
    user_memory: true
    session_context: always
```

For each store, you can use:
- `true` / `false`
- a mode string: `always`, `agentic`, `propose`, `hitl`
- an object with `{ enabled, mode, ... }` for advanced per-store options

## Routing Rules (Agent Selection)
By default, the bot replies with the `default` agent. You can add routing rules:

```yaml
agents:
  routing:
    rules:
      - agent: research
        channel_ids: [123456789012345678]
        contains: ["research:"]
```

Rules match on `channel_ids`, `user_ids`, `guild_ids`, `contains`, and `starts_with`.
All specified conditions in a rule must match.

## Tool Permissions + Audit
- Configure runtime tool access with `tool_permissions`.
- Supported selectors per rule:
  - `channel_ids`
  - `role_ids`
  - `user_ids`
  - `guild_ids`
  - `agents`
  - `tools`
- Rule resolution is deterministic:
  - `deny` overrides `allow`
  - if no rule matches, `default_effect` is applied
- Blocked tool calls return a clear user-facing message.
- Every tool event is written to append-only JSONL audit logs with outcomes:
  - `allowed`
  - `blocked`
  - `executed`
  - `failed`
- Audit applies to normal Discord runs, fallback tool-call execution, cron, and heartbeat runs.
- Argument logging is off by default. If enabled, configured sensitive keys are redacted.

Example:

```yaml
tool_permissions:
  enabled: true
  default_effect: allow
  rules:
    - effect: deny
      tools: [shell]
    - effect: allow
      tools: [shell]
      role_ids: [123456789012345678]
      channel_ids: [234567890123456789]
  audit:
    enabled: true
    path: logs/tool-audit.jsonl
    include_arguments: false
    redacted_keys: [token, secret, password, api_key, authorization]
```

## Workspace Agents
Agents can be added without code changes using:

- `workspace/agents/<agent-name>/agent.yaml`
- `workspace/agents/<agent-name>/AGENTS.md`

With the default home setup, these paths resolve under `~/.bitdoze-bot/workspace/agents/`.

Example `agent.yaml`:

```yaml
name: software-engineer
enabled: true
model:
  id: glm-4.7
  base_url: https://api.z.ai/api/coding/paas/v4
  api_key_env: GLM_API_KEY
tools: [web_search, website, github, file, shell]
skills: []
```

Folder agents are merged with config-defined agents by name. If names collide, folder definitions win.

## Teams
Example:

```yaml
teams:
  default: delivery-team
  definitions:
    - name: delivery-team
      members: [architect, software-engineer]
      respond_directly: true
      determine_input_for_members: true
      delegate_to_all_members: true
      add_team_history_to_members: true
      num_team_history_runs: 5
```

The team and members share the configured SQLite DB. Team memory/history is handled by Agno Team settings, while member learning is handled by each member's LearningMachine config.

## Team Debugging
- Each team/agent run logs:
  - target kind (`agent` or `team`) and target name
  - selected team members
  - elapsed runtime
  - run id + model
  - token/latency metrics when provided by Agno
  - delegation paths extracted from `member_responses`

## Logging
- Logging is configured from `config.yaml` under `logging`.
- Defaults: `level: INFO`, `format: detailed`, file logging enabled at `logs/bitdoze-bot.log`.
- Invalid log levels safely fall back to `INFO`.

Example:

```yaml
logging:
  level: DEBUG
  format: detailed # detailed | simple | custom format string
  file:
    enabled: true
    path: logs/bitdoze-bot.log
    max_bytes: 10485760
    backup_count: 5
```

- Live tail:

```bash
tail -f ~/.bitdoze-bot/logs/bitdoze-bot.log
```

## Runtime Timeouts + Concurrency
Each `agent.run()` call is guarded by a configurable timeout. This applies per call, not per
full flow — a research mode request that retries once gets the timeout applied to each attempt
independently.

```yaml
runtime:
  agent_timeout: 600       # seconds per agent.run for Discord messages and research
  cron_timeout: 600        # seconds per agent.run for cron jobs
  heartbeat_timeout: 120   # seconds per agent.run for heartbeat
  max_concurrent_runs: 4   # max parallel agent.run calls across all sources
```

- If a run exceeds its timeout the bot replies with a timeout message (Discord) or logs a warning (cron/heartbeat) and moves on.
- `max_concurrent_runs` limits how many `agent.run` calls can execute in parallel across Discord messages, cron, and heartbeat combined.
- All values are optional and fall back to the defaults shown above when omitted.

## How It Works
Runtime flow:
- On startup, the bot auto-loads `~/.bitdoze-bot/config.yaml` (legacy fallback: `config.yml`, then repo `config.yaml`), then builds global toolkits from `toolkits`.
- It loads agents from:
  - `agents.definitions` in config
  - `workspace/agents/<name>/agent.yaml` (folder-based agents)
- For folder-based agents, `workspace/agents/<name>/AGENTS.md` is added to that agent's instructions.
- It builds Agno `Agent` members (with memory + learning), then Agno `Team` objects from `teams.definitions`.
- The runtime registry can resolve both agents and teams by name, including aliases.

Message handling:
- Discord message arrives -> routing rules in `agents.routing.rules` choose a target name.
- The target can be either a single agent or a team.
- The bot calls `.run(...)` on the selected target.
- If target is a team, Agno handles delegation and synthesis natively.

Memory and learning:
- Shared DB: `memory.db_file` (SQLite).
- Member learning: configured via `learning` (LearningMachine stores like `user_profile`, `user_memory`).
- Team memory/history: configured in `teams.definitions[]` via options such as:
  - `add_team_history_to_members`
  - `num_team_history_runs`
  - `add_history_to_context`
  - session summary settings inherited from memory config.

Add a new teammate:
1. Create folder: `workspace/agents/<new-agent>/`
2. Add `agent.yaml` with model settings (`id`, `base_url`, `api_key_env`)
3. Add `AGENTS.md` with role-specific instructions
4. Add the agent name to `teams.definitions[].members` in `config.yaml`
5. Restart the bot

## Skills Format (Agno)
Skills follow Agno's skill structure (see Agno docs). Each skill lives in its own folder
under `skills/` with a `SKILL.md` that includes YAML frontmatter (name/description).
The agent loads skills via `LocalSkills`.

With the default home setup, this resolves under `~/.bitdoze-bot/skills/`.

To target specific skills per agent, set:

```yaml
agents:
  definitions:
    - name: research
      skills: [web-research]
```

Skill names must be lowercase and use hyphens, and must match the folder name.

## Docker (PgVector + Cognee)

The project includes a `docker-compose.yml` for PostgreSQL 17 with pgvector and a Cognee API service. This is **optional** — LanceDb works without any external services.

```bash
# Start services
docker compose up -d

# Check status
docker compose ps

# View logs (examples)
docker compose logs -f pgvector
docker compose logs -f cognee

# Stop
docker compose down
```

PgVector configuration:
| Setting | Default | Env Var |
|---------|---------|---------|
| Port | 5532 | `PGVECTOR_PORT` |
| Database | bitdoze | — |
| User | bitdoze | — |
| Password | bitdoze_secret | `POSTGRES_PASSWORD` |
| Data dir | `~/.bitdoze-bot/data/pgdata` | `BITDOZE_BOT_HOME` |
| Connection URL | `postgresql+psycopg://bitdoze:bitdoze_secret@localhost:5532/bitdoze` | `PGVECTOR_DB_URL` |

Port 5532 is used to avoid conflicts with any system PostgreSQL on 5432.

Cognee configuration:
| Setting | Default | Env Var |
|---------|---------|---------|
| Host bind | `127.0.0.1:8000` | — |
| Require auth | `false` | — |
| Backend access control | `false` | — |
| LLM key passthrough | empty | `LLM_API_KEY` |
| Data dir | `~/.bitdoze-bot/data/cognee` | `BITDOZE_BOT_HOME` |

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/setup_bot.py` | Interactive setup wizard (config, env, service, Docker) |
| `scripts/generate_soul.py` | Generate/update SOUL.md with self-improvement template |
| `scripts/setup_knowledge.py` | Initialize knowledge base and load documents |

```bash
# Setup wizard
python scripts/setup_bot.py

# Generate SOUL.md (supports --force, --dry-run, --home-dir)
python scripts/generate_soul.py

# Setup knowledge base (supports --backend, --docs-dir, --config)
python scripts/setup_knowledge.py
```

## Notes
- **Heartbeat** sends a proactive update every 30 minutes. If it returns `HEARTBEAT_OK`, the message is suppressed.
- For lower token usage, keep `heartbeat.session_scope: isolated` and optionally point `heartbeat.agent` to a lightweight agent with minimal tools/memory.
- `tools: []` on an agent/team now means "no tools" (explicitly empty), not "all tools".
- **FileTools** is sandboxed to `workspace/` by default.

## Cron Jobs
Enable cron jobs in `workspace/CRON.yaml`:

```yaml
enabled: true
timezone: Europe/Bucharest
channel_id: 123456789012345678
jobs:
  - name: daily-status
    cron: "0 9 * * *"
    agent: main
    message: "Send a daily status update."
    deliver: true
    session_scope: isolated
```

## Tests
Run:

```bash
uv run pytest -q
```

Current coverage includes:
- workspace agent loading + team registry wiring
- alias resolution
- routing rule selection
- delegation path extraction helper
- config loading and validation
- setup wizard answer generation

## Project Structure

```
bitdoze-bot/
├── main.py                    # Entry point
├── config.example.yaml        # Reference configuration
├── docker-compose.yml         # PgVector + Cognee (optional)
├── pyproject.toml             # Dependencies (managed with uv)
├── bitdoze_bot/
│   ├── agents.py              # Agent/team construction + knowledge base
│   ├── config.py              # Config loading and resolution
│   ├── cron.py                # Scheduled job runner
│   ├── discord_bot.py         # Discord client + streaming handler
│   ├── discovery_tools.py     # Self-improvement tools (save/search discoveries)
│   ├── heartbeat.py           # Periodic health checks
│   ├── logging_setup.py       # Structured logging
│   ├── run_monitor.py         # Run monitoring + telemetry
│   ├── setup_wizard.py        # Interactive setup
│   ├── tool_permissions.py    # Tool access control + audit
│   └── utils.py               # Shared utilities
├── scripts/
│   ├── setup_bot.py           # Setup wizard entry point
│   ├── generate_soul.py       # SOUL.md generator
│   └── setup_knowledge.py     # Knowledge base setup
└── tests/                     # pytest test suite
```
