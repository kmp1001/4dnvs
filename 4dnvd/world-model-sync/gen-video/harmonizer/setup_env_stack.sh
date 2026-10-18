#!/usr/bin/env bash
set -e

echo "🚀 Initializing Unified Pixi Workspace..."

# 1. Clean up any stale environment lockfiles
if [ -d ".pixi" ]; then
    echo "🧹 Removing old virtual environments..."
    rm -rf .pixi pixi.lock
fi

echo "📦 Resolving submodule and downloading package ecosystem..."
pixi install

echo "✨ Success! Workspace is ready."