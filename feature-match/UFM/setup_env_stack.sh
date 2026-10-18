#!/usr/bin/env bash
set -e

echo "🚀 Initializing Unified Pixi Workspace..."

# 1. Clean up any stale environment lockfiles
if [ -d ".pixi" ]; then
    echo "🧹 Removing old virtual environments..."
    rm -rf .pixi pixi.lock
fi

# 2. Make sure the script is executable and submodules are present
if [ ! -d "UniCeption" ]; then
    echo "❌ Error: UniCeption directory not found. Did you forget to clone submodules?"
    exit 1
fi

echo "📦 Resolving UniCeption submodule and downloading package ecosystem..."
# Pixi automatically evaluates UniCeption first, hooks its dependencies, 
# and sets up your environment in one unified step.
pixi install

echo "✨ Success! Workspace is ready."