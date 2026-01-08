# DSA Tissue Registration - React/Vite Application

This is the React/Vite frontend and FastAPI backend for the DSA Tissue Registration application, running in a Docker stack.

## Architecture

- **Frontend**: React + TypeScript + Vite
- **Backend**: FastAPI (Python)
- **Components**: bdsa-react-components library for DSA integration
- **Containerization**: Docker Compose

## Quick Start

### Prerequisites

- Docker and Docker Compose

### Docker Setup (Recommended)

1. **Set up environment variables**:
   ```bash
   cp .env.docker .env
   # Edit .env with your DSA credentials (DSAKEY)
   ```

2. **Start the Docker stack**:
   ```bash
   # Using the convenience script (recommended)
   ./run-docker.sh dev    # Development mode with live reload
   ./run-docker.sh prod   # Production mode
   
   # Or manually:
   docker-compose up --build                    # Production
   docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build  # Development with live reload
   ```

   **Live Reload Features:**
   - **Frontend**: Vite HMR (Hot Module Replacement) - changes to React components update instantly
   - **Backend**: Uvicorn auto-reload - changes to Python files restart the server automatically
   - Both services watch for file changes and update in real-time

3. **Access the application**:
   - Frontend: http://localhost (production) or http://localhost:5173 (development)
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Local Development (Outside Docker)

If you prefer to develop locally without Docker:

1. **Install dependencies**:
   ```bash
   npm install
   ```
   This will install all dependencies including `bdsa-react-components` from npm.

2. **Start development servers**:
   ```bash
   # Terminal 1: Frontend
   npm run dev
   
   # Terminal 2: Backend (from backend directory)
   cd backend
   uvicorn main:app --reload
   ```

## Project Structure

```
viteReg/
├── src/                    # React frontend source
│   ├── components/         # React components
│   ├── services/           # API client and services
│   ├── types/              # TypeScript type definitions
│   └── App.tsx             # Main app component
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/            # API route handlers
│   │   └── core/           # Core configuration
│   └── main.py             # FastAPI application entry point
├── Dockerfile              # Production frontend build
├── Dockerfile.dev          # Development frontend
├── docker-compose.yml      # Production Docker Compose
└── docker-compose.dev.yml  # Development overrides
```

## Development

### Frontend Development

The frontend uses Vite for fast development with HMR. In development mode, the source code is mounted as a volume for hot reload.

### Backend Development

The backend uses FastAPI with Uvicorn and auto-reload enabled in development mode. API endpoints are defined in `backend/app/api/`.

### Building for Production

```bash
docker-compose build
docker-compose up
```

The frontend will be built and served via nginx, with API requests proxied to the FastAPI backend.

## API Endpoints

See the FastAPI documentation at http://localhost:8000/docs for interactive API documentation.

Main endpoints:
- `/api/cases` - Case and slide management
- `/api/register` - Image registration
- `/api/images` - Image viewing and metadata
- `/api/annotations` - Annotation management

## Notes

- **User Permissions**: The Docker setup automatically uses your host user's UID/GID to ensure proper file permissions. See `USER_PERMISSIONS.md` for details.
- **bdsa-react-components**: Installed from npm automatically during `npm install`
- Environment variables are configured via `.env` file or docker-compose environment section
- Backend cache is persisted in a Docker volume
- Model files are pre-downloaded during Docker build
