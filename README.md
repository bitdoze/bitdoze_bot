# Bitdoze Bot (Agno + Discord)

A Discord-first agent powered by Agno, with Web Search, Website Scraping, Local File tools, memory, and a periodic heartbeat.

## Features
- **Agno toolkits**: WebSearchTools, WebsiteTools, FileTools, DiscordTools
- **Agno toolkits**: WebSearchTools, HackerNewsTools, WebsiteTools, GithubTools, YouTubeTools, FileTools, ShellTools, DiscordTools
- **Memory**: SQLite-backed, automatic memory updates by default
- **Soul + Heartbeat**: persona via `workspace/SOUL.md`, heartbeat checks via `workspace/HEARTBEAT.md`
- **Mention-first**: respond when tagged with `@Bot`
- **Extensible**: add more agents and future skills in config

## Setup
1) Create a config:

```bash
cp config.example.yaml config.yaml
```

2) Create a `.env` file:

```bash
cp .env.example .env
# edit .env with your values
```

3) Install dependencies:

```bash
pip install agno discord.py pyyaml ddgs beautifulsoup4
```

4) Run (auto-loads `.env` by default):

```bash
python main.py --config config.yaml
```

## Discord Requirements
- Enable the **Message Content Intent** in the Discord Developer Portal for your bot.
- Invite the bot with the right permissions to post messages in your target channels.

## Configuration Overview
Edit `config.yaml`:
- `model`: provider, model id, base URL, and API key env var
- `model.structured_outputs`: set `false` for providers that reject response_format (e.g., StepFun/OpenRouter)
- `discord`: bot token env var
- `memory`: `mode: automatic` (best capture), SQLite db path, history + summaries (custom prompt supported)
- `learning`: enable Agno LearningMachine and set learning modes (`always`, `agentic`, `propose`, `hitl`)
- `toolkits`: enable/disable web search, hackernews, website, github, youtube, file, shell, discord, tasks, collaboration tools
- `heartbeat`: 30-min cadence, optional channel override
- `cron`: schedule jobs via `workspace/CRON.yaml`
- `agents`: define multiple agents, per-agent tool selections, and routing rules
- `agents.team_directory`: auto-load extra agents from subfolders (`agent-config.yml` + `agents.md`)
- `teams`: define Agno Teams that coordinate multiple agents (ex: `build-team`)
- `skills`: optional skill packs loaded from `skills/`
- `context`: enable datetime injection and set timezone
- `context.agents_path`: load workspace instructions into system context
- `context.tasks_dir` + `context.tasks_path`: shared task board folder/instructions for team agents
- `context.specs_dir` + `context.handoffs_dir`: shared architect/engineer collaboration paths
- `context.user_path` + `context.memory_dir` + `context.long_memory_path`: load USER + daily memory + long memory
- `context.main_session_scope`: `dm_only` (default) or `always` for long memory

### Learning Modes (Agno)
Configure learning in `config.yaml`:

```yaml
learning:
  enabled: true
  mode: always # default mode for enabled stores
  stores:
    user_profile: true
    user_memory: true
    session_context: false
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

## Skills Format (Agno)
Skills follow Agno's skill structure (see Agno docs). Each skill lives in its own folder
under `skills/` with a `SKILL.md` that includes YAML frontmatter (name/description).
The agent loads skills via `LocalSkills`.

To target specific skills per agent, set:

```yaml
agents:
  definitions:
    - name: research
      skills: [web-research]
```

Skill names must be lowercase and use hyphens, and must match the folder name.

## External Team Agents
You can auto-discover additional agents from a team folder:

```text
workspace/agents/
  architect/
    agent-config.yml
    agents.md
    workspace/
  software-engineer/
    agent-config.yml
    agents.md
    workspace/
```

`agent-config.yml` supports:
- `name`
- `model` (or top-level fields): `provider`, `id`, `base_url`, `api_key` or `api_key_env`
- `tools`
- `skills`
- `learning`
- `workspace` (defaults to `workspace` inside the agent folder)
- `instructions`

`agents.md` is optional and gets injected into that agent's system instructions.
Use `tools: [bridge]` for a simpler setup with one toolkit exposing file/shell/task/spec/handoff
helpers plus web/github/youtube wrappers. To share the main project workspace, set `workspace: ../../..`.

## Teams (Agno)
You can define coordinated teams in `config.yaml`:

```yaml
teams:
  enabled: true
  default: build-team
  definitions:
    - name: build-team
      members: [architect, software-engineer]
```

Use in Discord with `agent:build-team <your request>`.

## Included Team Setup
This repo now includes a ready-to-use team setup:

- `workspace/agents/architect/agent-config.yml`
- `workspace/agents/architect/agents.md`
- `workspace/agents/software-engineer/agent-config.yml`
- `workspace/agents/software-engineer/agents.md`
- `workspace/tasks/board.yaml`
- `workspace/tasks/README.md`

Default behavior in `config.yaml`:
- `agents.team_directory: workspace/agents`
- `agents.default: build-team`
- `teams.enabled: true`
- `teams.default: build-team`

So by default, the bot runs as a team (`architect` + `software-engineer`) and can coordinate work through `workspace/tasks/board.yaml`.

## Shared Tasks Workspace
The bot now keeps a shared tasks board at `workspace/tasks/`:
- `workspace/tasks/board.yaml`
- `workspace/tasks/board.v2.schema.json`
- `workspace/tasks/events.log`
- `workspace/tasks/README.md`
- `workspace/tasks/notes/`

On startup, these are auto-created if missing. The board is versioned and migrated
automatically, validated on startup, and validated on every write.

Task mutations use a lock-backed store and append an audit record to `events.log`.
The `tasks` toolkit exposes safe operations (`create_task`, `update_status`, `assign_owner`,
`set_dependencies`, `add_note`) so agents do not need to edit YAML manually.
Each agent gets a tasks tool with its own default actor identity, so it can update tasks
without manually passing `actor` each time.
Bridge also exposes these task operations as compatibility aliases.

## Shared Team Workspace
The bot also keeps a shared collaboration workspace:
- `workspace/team/specs/` (architect writes specifications)
- `workspace/team/handoffs/` (engineer writes implementation handoffs)
- `workspace/team/README.md`

These locations are shared and accessible to both architect and software-engineer, independent
of their own per-agent file/shell workspace settings.

## Notes
- **Heartbeat** sends a proactive update every 30 minutes. If it returns `HEARTBEAT_OK`, the message is suppressed.
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
