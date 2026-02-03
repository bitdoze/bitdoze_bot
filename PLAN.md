# Bitdoze-Bot Plan (Agno-based Discord Agent)

## 1) Goals (MVP)
- Discord-first agent that can chat in a server and call Agno toolkits.
- Config file to set model + API endpoint (OpenAI-compatible) and other runtime knobs.
- Built-in toolkits: WebSearchTools, DiscordTools, WebsiteTools, FileTools.
- Memory across sessions, with option to switch between automatic vs agentic memory.
- “Soul” (persona) + “Heartbeat” (proactive check-ins) inspired by OpenClaw.
- Extensible: add new agents and “skills” later without major refactor.

## 2) Research notes (key references)
Agno:
- Toolkits overview lists Web Search, Web Scrape (WebsiteTools), Local File toolkit, etc.
- WebSearchTools uses `ddgs` and supports multiple backends; recommended for general web search.
- WebsiteTools requires `beautifulsoup4` and supports `read_url` and `add_website_to_knowledge_base`.
- DiscordTools provides send/read/history/channel management functions via a Discord bot token.
- Memory: Agno supports automatic and agentic memory; memory persists via DB (SQLite/Postgres). Multiple agents can share memories by using the same DB.

OpenClaw inspiration:
- Heartbeat is a periodic “proactive” agent run; default prompt can read `HEARTBEAT.md` and suppress output if `HEARTBEAT_OK`.
- “Soul” is a dedicated file (`SOUL.md`) defining identity/tone/boundaries.
- Workspace-centric design with instruction files (AGENTS.md, SOUL.md, etc.).

Nanobot:
- Public documentation is minimal/closed; plan only draws high-level inspiration (extensible agent + MCP style).

## 3) Architecture Overview
### 3.1 Runtime components
- **Discord Bot Runner**
  - `discord.py` client handling message events and commands.
  - Routes messages to a selected Agno Agent instance.
  - Supports per-channel or per-user session IDs.
- **Agno Agent Layer**
  - One “primary” agent + optional additional agents.
  - Each agent is configured with model + toolkits + memory.
- **Toolkits**
  - WebSearchTools (requires `ddgs`)
  - WebsiteTools (requires `beautifulsoup4`)
  - FileTools (sandboxed to workspace directory)
  - DiscordTools (optional; for agent-initiated actions)
- **Memory**
  - SQLite DB by default.
  - Option to toggle `update_memory_on_run` (automatic) or `enable_agentic_memory` (agent-managed).
- **Heartbeat Scheduler**
  - Background asyncio task that triggers a heartbeat run at a configured cadence.
  - Reads `HEARTBEAT.md` (if present) into the prompt.
  - Suppresses output if response is `HEARTBEAT_OK` (configurable).
- **Soul/Persona Loader**
  - Loads `SOUL.md` content and injects into system prompt.

### 3.2 Config file
Proposed `config.toml` (format to confirm). Example sections:
- `model`: provider, model ID, base URL, API key env var name.
- `discord`: bot token env var name, guild/channel defaults, command prefix.
- `memory`: db type (sqlite/postgres), db path/url, mode (automatic/agentic).
- `toolkits`: enable/disable + per-toolkit settings (backend, base_dir).
- `heartbeat`: enabled, interval, target channel, quiet_acks.
- `agents`: list of agents with model overrides, tools, and routing rules.
- `skills`: registry of skill folders (future). Each skill bundles prompts + tool configs.

## 4) File/Folder Layout (proposed)
- `main.py` – app entrypoint (loads config, starts bot).
- `bitdoze_bot/`
  - `config.py` – config schema + loader.
  - `agents.py` – Agno agent factory + registry.
  - `discord_bot.py` – discord client + message routing.
  - `heartbeat.py` – scheduler + heartbeat prompt logic.
  - `memory.py` – DB initialization + memory mode wiring.
  - `skills/` – pluggable “skills” (future) e.g. `skills/web_research/skill.yaml`.
- `workspace/`
  - `SOUL.md` – persona/voice
  - `HEARTBEAT.md` – proactive checklist
  - `AGENTS.md` – operator instructions for the runtime
- `config.example.toml`

## 5) MVP Flow
1. Load config and build toolkits.
2. Initialize memory DB and create the primary Agent with toolkits + soul prompt.
3. Connect to Discord and listen for messages.
4. For each message: create session id → call agent → send response.
5. Heartbeat task runs periodically and posts to last active channel if not `HEARTBEAT_OK`.

## 6) Extensibility (future)
- **Additional Agents**: define in config; select agent by channel or command.
- **Agno Skills support**: map “skill packs” to toolkits + prompts + optional MCP servers.
- **MCP tools**: allow running remote tool servers for heavy tasks.
- **Knowledge Base**: enable WebsiteTools + vector store for site ingestion.

## 7) Risks / Open Questions
- Config format (TOML vs YAML) — decide early to avoid breaking changes.
- Discord rate limits, message length constraints, and intent permissions.
- Memory strategy (automatic vs agentic) and DB choice for multi-agent scale.
- Safe defaults for FileTools base_dir to avoid risky file access.
- Heartbeat spam prevention and “quiet ack” rules.

## 8) Implementation Phases
- **Phase 1: Skeleton**
  - Config loader + basic agent + Discord bot message loop.
- **Phase 2: Toolkits**
  - Add WebSearchTools, WebsiteTools, FileTools, DiscordTools.
- **Phase 3: Memory**
  - SQLite memory, session IDs, and mode toggle.
- **Phase 4: Soul/Heartbeat**
  - SOUL.md ingestion + heartbeat scheduler.
- **Phase 5: Multi-agent**
  - Agent registry, routing rules, and per-agent tool overrides.

## 9) Minimal “Soul” + “Heartbeat” Behavior
- **SOUL.md**: appended to system prompt on every run.
- **HEARTBEAT.md**: if present and non-empty, included in heartbeat prompt.
- **HEARTBEAT_OK**: if response starts with this token, suppress message.


## 10) References
- https://docs.agno.com/tools/toolkits/overview
- https://docs.agno.com/tools/toolkits/search/websearch
- https://docs.agno.com/tools/toolkits/social/discord
- https://docs.agno.com/integrations/discord/overview
- https://docs.agno.com/tools/toolkits/web-scrape/website
- https://docs-v1.agno.com/tools/toolkits/local/file
- https://docs.agno.com/memory
- https://docs.openclaw.ai/getting-started
- https://docs.openclaw.ai/claude/heartbeat
- https://docs.openclaw.ai/claude/soul
- https://www.nanobot.ai/
- https://github.com/nanobot-ai/nanobot
