# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`cpm_torch` is a PyTorch implementation of the Cellular Potts Model (CPM) for biological cell simulation. The project aims to make CPM learnable through machine learning, particularly reinforcement learning. It provides GPU-accelerated cellular simulations with energy-based dynamics and integrates with machine learning frameworks for training neural networks to control cell behavior.

## Development Commands

The project uses Poetry for dependency management. Common commands:

```bash
# Install dependencies
poetry install

# Install PyTorch with CUDA support
poetry add torch torchvision torchaudio --source torch_cu124

# Install requirements from requirements.txt
poetry add $(cat requirements.txt)

# Build and publish package
poetry publish --build
```

## Core Architecture

### Main Components

- **`CPM` class** (`cpm_torch/CPM.py`): Core Cellular Potts Model implementation with energy calculations (area, perimeter), GPU acceleration, and checkerboard update patterns
- **`CPMEnv`** (`cpm_torch/CPMEnv.py`): Gymnasium environment for reinforcement learning with various reward functions (image matching, directional movement)  
- **`CPMDiffusionEnv`** (`cpm_torch/CPMDiffusionEnv.py`): Extended environment with molecular diffusion simulation
- **Training modules** (`cpm_torch/Training/`): RL training components including custom PPO implementation and Neural Hamiltonian networks

### Key Design Patterns

- **Energy-based simulation**: CPM uses Hamiltonian energy functions (area energy, perimeter energy) to drive cell dynamics
- **Checkerboard updates**: Parallel updates using spatial offsets to avoid conflicts
- **Patch-based processing**: Operations performed on 3x3 patches for efficiency
- **Batched computation**: Support for batch processing multiple simulations simultaneously

### Neural Network Integration

- Neural networks can modify energy functions (`dH_NN` parameters) to influence cell behavior
- Reinforcement learning agents learn to control cellular dynamics through energy modifications
- Support for both traditional CPM physics and learned behaviors

## Code Organization

- `cpm_torch/CPM*.py`: Core simulation classes
- `cpm_torch/Training/`: ML training components (PPO, Neural Hamiltonian, policies)
- `examples/`: Jupyter notebooks demonstrating usage patterns
- Configuration through `CPM_config` class for simulation parameters

## GPU Acceleration

The codebase is designed for CUDA acceleration with fallback to CPU. Most tensor operations are performed on GPU when available, with explicit device management throughout the codebase.

## Development Notes

- The project uses Japanese comments in many places, reflecting its research origins
- Jupyter notebooks in `examples/` directory provide the primary interface for experimentation
- Uses stable-baselines3 for reinforcement learning integration
- Custom gym environments follow standard OpenAI Gym patterns