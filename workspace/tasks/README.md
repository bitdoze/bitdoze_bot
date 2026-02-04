# Team Tasks

This folder is the shared task board for all agents and teams.

## Files
- `board.yaml`: canonical task board
- `board.v2.schema.json`: schema used by YAML language server
- `events.log`: append-only audit trail of task mutations
- `notes/`: optional detailed task notes

## Task Lifecycle
- `todo` -> `in_progress` -> `done`
- Any open task can go to `blocked`, but must include `blocked_reason`.

## Transition Rules
- `todo` -> `todo|in_progress|blocked`
- `in_progress` -> `in_progress|blocked|done|todo`
- `blocked` -> `blocked|in_progress|todo`
- `done` -> `done|in_progress` (re-open allowed)

## Team Policy
- `architect`: planning, decomposition, assignment, review.
- `software-engineer`: implementation and validation tasks.
- `software-engineer` can only mutate tasks owned by `software-engineer`.

## Suggested Task Shape
- `id`: unique id (ex: `task-001`)
- `title`: short summary
- `status`: `todo|in_progress|blocked|done`
- `owner`: agent name (`architect`, `software-engineer`, etc.)
- `depends_on`: list of task ids
- `updated_at`: ISO date-time
- `notes`: short progress updates
- `blocked_reason`: empty string unless `status: blocked`, then a non-empty reason
