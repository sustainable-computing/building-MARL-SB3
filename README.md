# Building MARL - SB3

This repository contains the re-implementation of the [building-MARL](https://github.com/sustainable-computing/building-MARL) repository, where the entire pipeline is built as a CLI. It has all the functionality of the parent repository along with some additional features.

## Installation Instructions

### Prerequisites

1. Python (v >= 3.10.0)
2. [EnergyPlus 9.3.0](https://github.com/NREL/EnergyPlus/releases/tag/v9.3.0)
3. A new virtual environment

### Installation

1. Activating your virtual environment
```bash
source path/to/your/env/bin/activate
```
2. Installing dependencies
```bash
pip install -r requirements.txt
```
3. Installing [zr-obp](https://github.com/st-tech/zr-obp)
   1. Clone this repository
   2. Edit the `pyproject.toml` and change `python = ">=3.7.1,<3.10"` to `python = ">=3.7.1,<3.11"`
   3. Run `python setup.py install`
4. Verifying installation
   1. Run `python main.py --help`
   2. If it displays the available commands on the terminal, without any errors you are good to go
