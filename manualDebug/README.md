# Manual Registration Debug Tool

A minimal web service for quickly testing transformation matrices from other UIs.

## Setup with uv

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Create virtual environment and install dependencies**:
   ```bash
   cd manualDebug
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e .
   ```

   Or use the run script which does this automatically:
   ```bash
   ./run.sh
   ```

3. **Set up DSA authentication** (required):
   ```bash
   cp .env.example .env
   # Edit .env and add your DSAKEY
   ```

   Or set environment variables:
   ```bash
   export DSAKEY="your-api-key-here"
   export DSA_BASE_URL="http://bdsa.pathology.emory.edu:8080/api/v1"
   ```

## Running

### Option 1: Using the run script (recommended)
```bash
cd manualDebug
./run.sh
```

### Option 2: Manual activation
```bash
cd manualDebug
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python3 app.py
```

### Option 3: With uvicorn directly
```bash
cd manualDebug
source .venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8002 --reload
```

Then open: http://localhost:8002

**Note:** Default port is 8002 (to avoid conflict with main backend on 8000). Change it by setting `PORT` environment variable:
```bash
PORT=8003 python3 app.py
```

## Usage

1. Enter the **Fixed Image ID** (target/reference image)
2. Enter the **Moving Image ID** (image to transform)
3. Select the matrix type (Homography, H1, H2, or Fundamental)
4. Paste your JSON transformation matrix into the text area
5. Click "Apply Transform"

The tool will:
- Load both images from DSA
- Apply the transformation matrix
- Display all four views: Fixed, Moving (original), Moving (transformed), and Overlay
- Show extracted transform parameters (rotation, scale, offset)

## JSON Format

Paste your JSON in this format:

```json
{
  "geom_info": {
    "Homography": [
      [0.8036541512898507, -0.026251719324804954, 7.991064800476521],
      [0.010849798858044617, 0.8128097314434242, 1.9440023459021754],
      [-0.00007160226364392854, -0.000026090723081625177, 1]
    ],
    "H1": [...],
    "H2": [...],
    "Fundamental": [...]
  }
}
```

Or just the `geom_info` object directly:

```json
{
  "Homography": [
    [0.8036541512898507, -0.026251719324804954, 7.991064800476521],
    [0.010849798858044617, 0.8128097314434242, 1.9440023459021754],
    [-0.00007160226364392854, -0.000026090723081625177, 1]
  ]
}
```

## Features

- Quick visual feedback
- Supports all matrix types (Homography, H1, H2, Fundamental)
- Shows overlay for easy comparison
- Displays extracted transform parameters
- Uses DSA authentication from `.env` file or environment variables

## Dependencies

The tool uses dependencies from `viteReg/backend` for DSA client and image processing. Make sure the backend directory is accessible for imports.

## Quick Reference

**Common package installs:**
- OpenCV: `uv pip install opencv-python`
- All deps: `uv pip install -e .`

See `QUICK_REF.md` for more shortcuts.
