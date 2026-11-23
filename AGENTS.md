# AGENTS.md

## Project rules

- The Shopify admin API token never lives in this repo. It is read from the
  Shopify CLI session on the host, mounted read-only into the container. Never
  hardcode it, never commit it, never print any part of it.
- The whole image flow is one command, `docker compose run --rm pipeline`,
  and everything runs inside Docker. Do not add host-side installs, virtual
  environments or system dependencies to the host.
