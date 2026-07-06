# Tool Setup

This repo assumes the long-lived tool environment lives under `/storage`, not inside `/notebooks`.

## Files

- Template: [`env/env_tools.example.sh`](/notebooks/env/env_tools.example.sh:1)
- Local runtime file: `/storage/env_tools.sh`

## Recommended setup

1. Copy the template to `/storage/env_tools.sh`.
2. Replace the Git identity values with your own.
3. Load it from your shell before using Codex or other local CLIs.

Example:

```bash
cp /notebooks/env/env_tools.example.sh /storage/env_tools.sh
$EDITOR /storage/env_tools.sh
source /storage/env_tools.sh
```

## What it configures

- `HOME=/storage`
- `OPENCODE_CONFIG_DIR=/storage/opencode-config`
- `PATH` entries for `~/.opencode/bin` and `~/.local/bin`
- global Git user name / email
- optional `nvm` initialization from `/storage/.nvm`

## Notes

- Keep `/storage/env_tools.sh` local. Do not commit machine-specific edits.
- If you need secrets later, put them in a separate local file and source that from `/storage/env_tools.sh`.
