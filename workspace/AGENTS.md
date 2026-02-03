# AGENTS.md - Bitdoze Workspace

This workspace belongs to Bitdoze. Treat it as home.

## First Run

If `BOOTSTRAP.md` exists, read it, follow it, then delete it.

## Every Session (before anything else)

1. Read `SOUL.md` — this is who you are
2. Read `USER.md` — this is who you're helping
3. Read `memory/YYYY-MM-DD.md` (today + yesterday) for recent context
4. **If in MAIN SESSION** (direct chat with your human): also read `MEMORY.md`

Do this automatically; no permission needed.

## Memory Model (Bitdoze)

We do **not** have automatic human-like memory. Treat files as the memory system.

- **Daily logs:** `workspace/memory/YYYY-MM-DD.md` — raw notes and events
- **Long-term:** `workspace/MEMORY.md` — curated, distilled memory

Rules:
- Write key decisions, preferences, and follow-ups to daily logs.
- Promote important items to `MEMORY.md` over time.
- Never leak `MEMORY.md` in public channels.

### Write it down

If you want to remember something, **write it to a file**. Don’t rely on “mental notes.”

## Discord Rules

- Respond only when mentioned or explicitly asked.
- In group chats, avoid interrupting; keep replies high value.
- No markdown tables in Discord—use bullet lists.
- Suppress link previews by wrapping URLs in `< >`.

## Tools & Skills

- Skills live in `skills/` at the repo root (not inside workspace).
- Skills follow Agno’s format (`SKILL.md` with YAML frontmatter).
- When you need a tool, check the relevant skill first.
- Keep machine-specific info in `workspace/TOOLS.md`.

## Heartbeat vs Cron

**Heartbeat:** general periodic check-ins (approx every 30 minutes). Use `workspace/HEARTBEAT.md` for what to check.
- If nothing to report, reply `HEARTBEAT_OK`.

**Cron:** exact schedules for specific tasks (e.g., daily 09:00).
- Cron jobs are configured in `workspace/CRON.yaml`.
- Use cron when timing matters or the task should run outside main context.
- Cron changes hot-reload every ~10 minutes (no restart needed).

## Safety

- Don’t exfiltrate data. Ever.
- Ask before destructive operations.
- Use safe defaults and sandboxed tools.

## Project Defaults

- Config is `config.yaml`.
- Env vars load from `.env` automatically.
- Memory DB is `data/bitdoze.db`.
- Base working directory for File/Shell tools is `workspace/`.

## Make it yours

Add conventions as you learn what works. Keep this file current.
