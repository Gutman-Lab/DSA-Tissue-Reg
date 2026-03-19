"""
Celery application for background task processing
"""
import os
from celery import Celery

# Use environment variable for Redis host, default to 'redis' (Docker service name)
redis_host = os.getenv("REDIS_HOST", "redis")

# Create Celery app
celery_app = Celery(
    "registration_worker",
    broker=f"redis://{redis_host}:6379/0",
    backend=f"redis://{redis_host}:6379/0",
    include=["app.tasks.registration_tasks"],
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes max per task
    task_soft_time_limit=25 * 60,  # 25 minutes soft limit
    worker_prefetch_multiplier=1,  # Process one task at a time
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks to prevent memory leaks
    broker_connection_retry_on_startup=True,  # Celery 6.0+: retry broker connection at startup
)

