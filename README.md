# 🚀 RISC-V UDB to LLVM TableGen Generator

**Automates RISC-V compiler backend generation from UDB specifications**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![RISC-V](https://img.shields.io/badge/RISC--V-Compatible-green.svg)](https://riscv.org)
[![LLVM](https://img.shields.io/badge/LLVM-TableGen-orange.svg)](https://llvm.org)

## 🎯 Overview

This tool reads RISC-V Unified Database (UDB) YAML specifications and automatically generates production-ready LLVM TableGen definitions for:

- ✅ **Instruction definitions** with proper encoding
- ✅ **Register classes** and allocation rules  
- ✅ **Selection patterns** for code generation
- ✅ **Scheduling models** for pipeline optimization
- ✅ **Complete LLVM backend** structure

**Impact**: Reduces 2-3 weeks of manual TableGen development to minutes of automated generation.

## 🚀 Quick Start

```bash
# 1. Create sample UDB file
python3 udb_tablegen_gen.py --create-sample

# 2. Generate TableGen files
python3 udb_tablegen_gen.py -i sample_udb.yaml -o generated/

# 3. View results
ls generated/