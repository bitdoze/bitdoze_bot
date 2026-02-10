---
name: docker-deploy
description: >-
  Use this skill when deploying or modifying Dockerized apps on this server.
  It enforces server-specific constraints: Docker is installed with Caddy,
  commands that change system/runtime state should be run with sudo, app
  stacks live in /home/dragos/docker-apps, Caddy lives in
  /home/dragos/docker-apps/caddy, wildcard DNS *.ai.bitdoze.com points to this
  server for subdomain routing, and Docker Compose files should use local ./
  paths instead of named volumes.
---

# Docker + Caddy Server Rules

Apply these rules for all deployment, operations, and setup tasks on this server.

## Environment Facts
- Docker is installed and available.
- Caddy is installed and managed from `/home/dragos/docker-apps/caddy`.
- Docker app projects must be placed under `/home/dragos/docker-apps`.
- Wildcard DNS `*.ai.bitdoze.com` points to this server and should be used for app subdomains.

## Permission Model
- Use `sudo` for commands that manage Docker runtime/services, networking, filesystem locations requiring elevation, or system-level setup.
- Prefer explicit commands that can be audited and repeated.

## Compose Placement and Style
- Prefer Docker Compose-based deployments.
- Place each app in its own folder under `/home/dragos/docker-apps/<app-name>`.
- Keep compose and related app files together in the app folder.
- Do not use named volumes for app data/config in compose files.
- Use local relative paths (`./...`) in the location where the compose file is created.

## Caddy Integration
- Route apps through Caddy using subdomains under `*.ai.bitdoze.com`.
- Keep Caddy config changes under `/home/dragos/docker-apps/caddy`.
- When adding a new app, include the target hostname and upstream container/service mapping.

## Expected Workflow
1. Create app directory in `/home/dragos/docker-apps/<app-name>`.
2. Add `docker-compose.yml` with services and `./...` path mappings.
3. Add/update Caddy config in `/home/dragos/docker-apps/caddy` for `<app>.ai.bitdoze.com`.
4. Run required Docker/Caddy commands with `sudo`.
5. Validate service health and public routing.
6. Rub curl to see if the new app is up on port or subdomain
7. Place sensitive date into `.env` file not in compose directly like passwords and hashes
8. `web` is the external network that can be used to be in same with caddy, so use this for frontend

## Guardrails
- Reject plans that place apps outside `/home/dragos/docker-apps`.
- Reject plans that use named Docker volumes for persistent files.
- Reject plans that bypass Caddy when public HTTP(S) access is required.
- Place sensitive date into `.env` file not in compose directly like passwords and hashes
