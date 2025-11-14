"""
Setup script for Phase-1 Edge Discovery System
Creates all necessary directories before running.
"""

import os
from pathlib import Path

# Directories to create
directories = [
    'logs',
    'data/cache',
    'results/trade_logs',
    'results/equity_curves',
    'results/reports',
    'models'
]

print("Setting up Phase-1 Edge Discovery System...")
print("-" * 60)

for directory in directories:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created: {directory}")

print("-" * 60)
print("✓ Setup complete!")
print("\nYou can now run:")
print("  python orchestrator_phase1.py")
