# Quick Reference

## Common Commands

### OpenCV (cv2)
```bash
uv pip install opencv-python
# or
pip install opencv-python
```

### Activate Virtual Environment
```bash
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### Install All Dependencies
```bash
uv pip install -e .
```

### Run the App
```bash
python3 app.py          # Default port: 8002
PORT=8003 python3 app.py  # Custom port
# or
./run.sh               # Default port: 8002
PORT=8003 ./run.sh     # Custom port
```

**Default port:** 8002 (to avoid conflict with main backend on 8000)

## Package Names Cheat Sheet

| What you want | Package name |
|--------------|--------------|
| OpenCV (cv2) | `opencv-python` |
| NumPy | `numpy` |
| PIL/Pillow | `Pillow` |
| FastAPI | `fastapi` |
| Uvicorn | `uvicorn[standard]` |
| Pydantic | `pydantic` |

## Common Issues

**"ModuleNotFoundError: No module named 'cv2'"**
```bash
uv pip install opencv-python
```

**"Command 'uv' not found"**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Virtual environment not activated?**
Look for `(.venv)` in your prompt. If missing:
```bash
source .venv/bin/activate
```
