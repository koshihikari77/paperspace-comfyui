#!/usr/bin/env bash

export HOME=/storage
export OPENCODE_CONFIG_DIR=/storage/opencode-config
export PATH="$HOME/.opencode/bin:$HOME/.local/bin:$PATH"

# Git identity for this environment.
# Replace these values in your local copy.
git config --global user.email "your-email@example.com"
git config --global user.name "Your Name"

# nvm (optional but recommended)
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
