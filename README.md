# Bitdoze Bot (Agno + Discord)

A Discord-first agent powered by Agno, with Web Search, Website Scraping, Local File tools, memory, and a periodic heartbeat.

## Features
- **Agno toolkits**: WebSearchTools, WebsiteTools, FileTools, DiscordTools
- **Agno toolkits**: WebSearchTools, HackerNewsTools, WebsiteTools, GithubTools, YouTubeTools, FileTools, ShellTools, DiscordTools
- **Agno Teams**: native team orchestration with delegation options
- **Team observability**: logs selected members, delegation paths, run timing, and metrics
- **Persistent logging**: stdout + rotating file logs at `logs/bitdoze-bot.log` (10MB x 5 backups)
- **Memory**: SQLite-backed, automatic memory updates by default
- **Learning**: Agno LearningMachine per member agent (user profile + memory stores)
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
- `toolkits`: enable/disable web search, hackernews, website, github, youtube, file, shell, discord tools
- `agents.workspace_dir`: folder-based agent loading from `workspace/agents/<name>/`
- `teams`: native Agno Team definitions, delegation behavior, team memory options, and default team
- `heartbeat`: 30-min cadence, optional channel override, `session_scope` (`isolated` to avoid heartbeat history growth), and optional dedicated `agent`
- `cron`: schedule jobs via `workspace/CRON.yaml`
- `agents`: define multiple agents, per-agent tool selections, and routing rules
- `skills`: optional skill packs loaded from `skills/`
- `context`: enable datetime injection and set timezone
- `context.agents_path`: load workspace instructions into system context
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

## Workspace Agents
Agents can be added without code changes using:

- `workspace/agents/<agent-name>/agent.yaml`
- `workspace/agents/<agent-name>/AGENTS.md`

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
- The bot logs to both stdout and `logs/bitdoze-bot.log`.
- File logs rotate automatically (`10MB`, `5` backups).
- `logs/` is git-ignored.
- Live tail:

```bash
tail -f logs/bitdoze-bot.log
```

## How It Works
Runtime flow:
- On startup, the bot loads `config.yaml`, then builds global toolkits from `toolkits`.
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

To target specific skills per agent, set:

```yaml
agents:
  definitions:
    - name: research
      skills: [web-research]
```

Skill names must be lowercase and use hyphens, and must match the folder name.

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
