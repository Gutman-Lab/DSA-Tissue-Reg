# Activating uv Environment

## Quick Start

```bash
cd manualDebug
uv venv                    # Create virtual environment (first time only)
source .venv/bin/activate  # Activate (Linux/Mac)
# OR on Windows:
.venv\Scripts\activate     # Activate (Windows)
uv pip install -e .        # Install dependencies (first time only)
python3 app.py             # Run the app
```

## Using the Run Script

The `run.sh` script automatically handles activation:

```bash
cd manualDebug
./run.sh
```

This will:
1. Check if uv is installed
2. Create `.venv` if it doesn't exist
3. Activate the environment
4. Install dependencies if needed
5. Run the app

## Manual Activation

If you prefer to activate manually:

### Linux/Mac:
```bash
source .venv/bin/activate
```

### Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

### Windows (CMD):
```cmd
.venv\Scripts\activate.bat
```

## Verifying Activation

When activated, you should see `(.venv)` in your terminal prompt:
```bash
(.venv) user@machine:~/manualDebug$
```

## Deactivating

Simply run:
```bash
deactivate
```
