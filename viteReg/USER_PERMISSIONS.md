# User Permissions in Docker

This document explains how user ID mapping works to ensure proper file permissions when using bind mounts.

## Problem

When Docker containers create files in bind-mounted directories, they're often owned by root or a different user, causing permission issues on the host system.

## Solution

We use **user ID mapping** to ensure the container runs as the same UID/GID as the host user, so files created in the container have the correct ownership on the host.

## How It Works

### 1. Build Arguments

Both Dockerfiles accept build arguments for `USER_ID` and `GROUP_ID`:

```dockerfile
ARG USER_ID=1000
ARG GROUP_ID=1000
```

### 2. User Creation

During the build, a user is created with matching UID/GID:

**Backend (Python):**
```dockerfile
RUN groupadd -g ${GROUP_ID} appuser || true && \
    useradd -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash appuser || true
```

**Frontend (Node):**
```dockerfile
RUN addgroup -g ${GROUP_ID} appuser || true && \
    adduser -u ${USER_ID} -G appuser -D -s /bin/sh appuser || true
```

### 3. Runtime User

Docker Compose sets the `user` directive to run as the mapped user:

```yaml
user: "${USER_ID}:${GROUP_ID}"
```

### 4. Automatic Detection

The `run-docker.sh` script automatically detects your user ID:

```bash
export USER_ID=$(id -u)
export GROUP_ID=$(id -g)
```

## Usage

### Automatic (Recommended)

Just use the startup script - it handles everything:

```bash
./run-docker.sh dev
```

### Manual

If running docker-compose directly, set the environment variables:

```bash
export USER_ID=$(id -u)
export GROUP_ID=$(id -g)
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

### Custom User IDs

You can override the user IDs if needed:

```bash
USER_ID=1001 GROUP_ID=1001 docker-compose up --build
```

## Verification

Check that files created in the container have correct ownership:

```bash
# Create a file in the container
docker-compose exec backend touch /app/test.txt

# Check ownership on host
ls -la backend/test.txt
# Should show your username, not root
```

## Troubleshooting

### Permission Denied Errors

1. **Check user ID mapping:**
   ```bash
   docker-compose exec backend id
   # Should match: uid=1000(appuser) gid=1000(appuser)
   ```

2. **Check file ownership:**
   ```bash
   ls -la backend/
   # Files should be owned by your user
   ```

3. **Fix existing files:**
   ```bash
   sudo chown -R $(id -u):$(id -g) backend/ src/
   ```

### User Already Exists

The `|| true` in user creation commands handles cases where the user/group already exists. This is safe and won't cause issues.

### Port Binding Issues

If you get "permission denied" when binding to ports < 1024, you may need to:
- Use ports >= 1024 (already done: 5173, 8000)
- Or run with `sudo` (not recommended)

## Production

In production mode, user mapping is less critical since:
- No bind mounts are used (files are copied into images)
- nginx runs as root (standard practice)
- Files are read-only

However, the build args are still passed for consistency.

## Notes

- **Windows/WSL**: User ID mapping works the same way
- **macOS**: User IDs typically start at 501
- **Linux**: User IDs typically start at 1000
- The `|| true` ensures the build doesn't fail if user/group already exists

