# IDA 9.0 Migration Guide for Dataset Generation Scripts

DISCLAIMER: This migration guide as well as the described changes are mostly authored by Claude, so take them with a grain of salt. They are tailored to macOS but I expect them to mostly translate to other platforms as well.

## Problem Summary
The dataset generation scripts were encountering "Fatal error before kernel init" when running IDA Pro 9.0 in headless mode on macOS. This was due to:

1. **Command-line argument incompatibility**: The `-t elf64` processor type specification is not valid in IDA 9.0
2. **Environment issues**: Python virtual environment variables interfering with IDA's embedded Python
3. **Missing HOME environment variable**: IDA 9.0 on macOS requires HOME to be set
4. **Import path issues**: Scripts couldn't find local module imports

## Applied Fixes

### 1. Updated `generate.py`
- **Removed `-t elf64` argument**: IDA 9.0 auto-detects the processor type
- **Added HOME environment variable**: Required for IDA 9.0 on macOS
- **Preserved environment cleanup**: Ensures Python venv doesn't interfere

copy over the updated version to `DIRTY/dataset-gen`: [generate.py](./generate.py)

### 2. Updated `debug.py` and `dump_trees.py`
- **Added sys.path manipulation**: Ensures local modules can be imported
- **Added error handling**: Better error reporting with stack traces
- **Updated plugin loading**: Try multiple plugin names for compatibility
- **Added typing imports**: Fixed missing type hints

apply the [Patch](./decompiler_scripts.diff)

## Usage Instructions

### Step 1: Initialize IDA (First Time Only)
```bash
python dataset-gen/initialize_ida.py "/Applications/IDA Professional 9.0.app/Contents/MacOS/idat"
```

### Step 2: Run Dataset Generation
```bash
cd dataset-gen
python3 generate.py --ida '/Applications/IDA Professional 9.0.app/Contents/MacOS/idat' -t 1 -v -b /path/to/HyRES/binaries/coreutils/ -o ../baseline_dataset/
```

## Troubleshooting

If "Fatal error before kernel init" still occurs:

1. **Run IDA GUI once**: Open IDA Pro GUI at least once after an update to complete initial setup
   ```bash
   open "/Applications/IDA Professional 9.0.app"
   ```

2. **Check IDA licensing**: Ensure IDA is properly licensed and activated

3. **Clear IDA settings**: Remove existing IDA configuration
   ```bash
   rm -rf ~/.idapro
   ```
   Then re-run `initialize_ida.py`

4. **Check system permissions**: Ensure IDA has necessary permissions
   - System Preferences > Security & Privacy > Privacy tab
   - Add Terminal/iTerm to "Full Disk Access"

5. **Disable Gatekeeper temporarily** (if needed):
   ```bash
   sudo spctl --master-disable
   # Run your scripts
   sudo spctl --master-enable
   ```

### If Python import errors occur:

1. **Check script paths**: Ensure all scripts are in correct directories
2. **Verify module files exist**: Check that `collect.py`, `function.py`, etc. are present
3. **Clear Python cache**:
   ```bash
   find dataset-gen -name "*.pyc" -delete
   find dataset-gen -name "__pycache__" -type d -exec rm -rf {} +
   ```

## API Compatibility Notes

### IDA 9.0 API Changes
The following changes were made for IDA 9.0 compatibility:

1. **Removed modules**: `ida_struct` and `ida_enum` → use `ida_typeinf` instead
2. **Function changes**: 
   - `idaapi.get_inf_structure()` → `ida_ida.inf_get_procname()`
   - Structure/enum functions moved to `tinfo_t` methods

### Current Implementation
The scripts use these IDA API functions which remain compatible:
- `ida.get_func(ea)` - Get function at address
- `ida.decompile(f)` - Decompile function
- `ida.get_func_name(ea)` - Get function name
- `ida.auto_wait()` - Wait for auto-analysis
- `ida.init_hexrays_plugin()` - Initialize Hex-Rays decompiler
- `ida.qexit(code)` - Exit IDA

## Environment Variables

The scripts use these environment variables:

- `OUTPUT_DIR`: Where to save generated data
- `PREFIX`: Filename prefix for output files
- `FUNCTIONS`: Temporary file for function data exchange
- `IDAUSR`: IDA user directory (set to temp dir to avoid venv conflicts)
- `HOME`: User home directory (required for IDA 9.0 on macOS)
