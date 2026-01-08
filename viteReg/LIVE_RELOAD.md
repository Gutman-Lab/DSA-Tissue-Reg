# Live Reload Configuration

This document explains how live reload works in the Docker development setup.

## How It Works

### Frontend (Vite)

The frontend uses **Vite's HMR (Hot Module Replacement)** for instant updates:

1. **File Watching**: Vite watches all files in `src/` and `public/` directories
2. **Polling**: Enabled via `CHOKIDAR_USEPOLLING=true` for Docker compatibility
3. **HMR**: Changes to React components update without full page reload
4. **Volume Mounts**: Source files are mounted as volumes, so changes on your host are immediately visible in the container

**Configuration:**
- `vite.config.ts` has `watch.usePolling: true` enabled
- `docker-compose.dev.yml` mounts all source directories
- `node_modules` is excluded to use the container's installed packages

### Backend (FastAPI/Uvicorn)

The backend uses **Uvicorn's auto-reload** feature:

1. **File Watching**: Uvicorn watches Python files in `/app` and `/app/app` directories
2. **Auto-restart**: When Python files change, Uvicorn automatically restarts the server
3. **Volume Mounts**: The entire `backend/` directory is mounted, so changes are immediately visible

**Configuration:**
- Command: `uvicorn main:app --reload --reload-dir /app/app --reload-dir /app`
- `PYTHONUNBUFFERED=1` ensures logs appear immediately
- `__pycache__` is excluded to prevent sync issues

## Troubleshooting

### Changes Not Detected

1. **Check volume mounts**: Ensure files are being mounted correctly
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec frontend ls -la /app/src
   ```

2. **Check polling**: If on WSL or certain systems, polling might be required
   - Already enabled via `CHOKIDAR_USEPOLLING=true`
   - Can increase polling interval in `vite.config.ts` if needed

3. **Check file permissions**: Ensure files are readable
   ```bash
   ls -la src/
   ```

### Backend Not Reloading

1. **Check Uvicorn logs**: Look for reload messages
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.dev.yml logs backend
   ```

2. **Verify reload directories**: Check that `--reload-dir` includes your code paths
   - Current: `--reload-dir /app/app --reload-dir /app`

3. **Check Python file changes**: Uvicorn only reloads on `.py` file changes
   - Changes to `.txt`, `.yaml`, etc. won't trigger reload
   - Restart manually if needed: `docker-compose restart backend`

### Frontend HMR Not Working

1. **Check browser console**: Look for HMR connection errors
2. **Check Vite logs**: 
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.dev.yml logs frontend
   ```

3. **Try hard refresh**: Sometimes HMR needs a manual refresh (Ctrl+Shift+R)

## Performance Notes

- **Polling**: File watching via polling uses more CPU than native file events
- **Volume Mounts**: Large `node_modules` directories can slow down file operations
  - This is why `node_modules` is excluded and uses the container's version
- **Backend Reload**: Uvicorn restart takes ~1-2 seconds, which is normal

## Disabling Live Reload

If you need to disable live reload:

1. **Frontend**: Remove `CHOKIDAR_USEPOLLING` and `watch.usePolling` settings
2. **Backend**: Change command to remove `--reload` flag
3. **Or**: Use production compose file: `docker-compose up` (without dev override)

