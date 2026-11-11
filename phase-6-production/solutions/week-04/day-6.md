# Solutions — Day 6 (Saturday): Containerizing Multi-service App

---

## Exercise 1: Add `/metrics` Endpoint + Prometheus Scrape Config

### Solution — FastAPI app with prometheus-client

```python
# app/main.py
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    REGISTRY,
)

logger = logging.getLogger(__name__)

# --- Metrics definitions -------------------------------------------------------
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
TASKS_IN_QUEUE = Gauge("tasks_in_queue", "Number of tasks currently in the queue")
# -------------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Application starting up")
    yield
    logger.info("Application shutting down")


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Record request count and latency for every endpoint."""
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code,
    ).inc()
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path,
    ).observe(duration)

    return response


@app.get("/health", status_code=status.HTTP_200_OK)
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus scrape endpoint — returns text exposition format."""
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/tasks")
async def list_tasks() -> dict[str, list]:
    # Placeholder — real implementation would query a DB/queue
    return {"tasks": []}
```

### Solution — `requirements.txt` addition

```
prometheus-client>=0.20.0
```

### Solution — `prometheus.yml`

```yaml
# prometheus.yml
global:
  scrape_interval: 15s      # how often to scrape targets
  evaluation_interval: 15s  # how often to evaluate alerting rules

scrape_configs:
  - job_name: "fastapi-app"
    static_configs:
      - targets: ["api:8000"]   # service name from compose, port 8000
    metrics_path: /metrics
    scrape_interval: 10s        # override global for this job
```

### Solution — Prometheus service in `docker-compose.yml`

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    healthcheck:
      test: ["CMD", "python", "-c",
             "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 5s
      start_period: 10s
      retries: 3
    networks:
      - app_net

  prometheus:
    image: prom/prometheus:v2.51.0
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro   # config (read-only)
      - prometheus_data:/prometheus                          # persistent metrics storage
    ports:
      - "9090:9090"
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.path=/prometheus"
      - "--storage.tsdb.retention.time=15d"
    depends_on:
      api:
        condition: service_healthy
    networks:
      - app_net

volumes:
  prometheus_data:

networks:
  app_net:
    driver: bridge
```

### How it works end-to-end

1. FastAPI exposes `/metrics` returning Prometheus text format.
2. `prometheus.yml` tells Prometheus to scrape `api:8000/metrics` every 10 seconds.
3. Prometheus stores time-series data in `prometheus_data` volume.
4. Browse `http://localhost:9090` → query `http_requests_total` to see counts by endpoint.

---

## Exercise 2: Graceful Shutdown for the Worker

### Solution — worker with SIGTERM handling

```python
# worker/main.py
import asyncio
import logging
import signal
import os
from typing import Any

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


class Worker:
    """Async task worker that drains the current job on SIGTERM."""

    def __init__(self) -> None:
        self._shutdown_event = asyncio.Event()
        self._current_job: str | None = None

    def _handle_sigterm(self, signum: int, frame: Any) -> None:
        """Called by the OS when Docker sends SIGTERM (e.g. docker stop)."""
        logger.info("SIGTERM received — finishing current job then stopping")
        # asyncio.Event.set() is thread-safe when called from a signal handler
        # but we need the loop to schedule it properly:
        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(self._shutdown_event.set)

    async def process_job(self, job_id: str) -> None:
        """Simulate processing a single job. Replace with real logic."""
        self._current_job = job_id
        logger.info("Processing job %s", job_id)
        # Simulate work — replace with real async I/O (DB query, HTTP call, etc.)
        await asyncio.sleep(5)
        logger.info("Finished job %s", job_id)
        self._current_job = None

    async def fetch_next_job(self) -> str | None:
        """Poll the queue for the next job. Returns None when queue is empty."""
        # Replace with real queue logic (e.g. Redis BLPOP, SQS receive, etc.)
        await asyncio.sleep(1)
        # Simulated: return a job ID, or None when there's nothing to do
        import random
        return f"job-{random.randint(1000, 9999)}" if random.random() > 0.3 else None

    async def run(self) -> None:
        """Main worker loop. Exits cleanly after SIGTERM."""
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        signal.signal(signal.SIGINT, self._handle_sigterm)   # Ctrl+C in dev

        logger.info("Worker started (PID %d)", os.getpid())

        while not self._shutdown_event.is_set():
            job_id = await self.fetch_next_job()

            if job_id is None:
                # No work — wait briefly, but also watch for shutdown signal
                try:
                    await asyncio.wait_for(
                        asyncio.shield(self._shutdown_event.wait()), timeout=2.0
                    )
                except asyncio.TimeoutError:
                    pass  # normal — loop back and poll again
                continue

            # Process the job. If shutdown arrives mid-job, we finish it first
            # because we check _shutdown_event only at the top of the while loop.
            await self.process_job(job_id)

        logger.info("Shutdown event set — worker exiting gracefully")


def main() -> None:
    worker = Worker()
    asyncio.run(worker.run())


if __name__ == "__main__":
    main()
```

### Why this works

- `signal.signal(SIGTERM, handler)` registers a callback for when Docker sends `SIGTERM` (triggered by `docker stop` or compose stopping the service).
- `asyncio.Event` is the coordination primitive: the signal handler sets it, the main loop checks it at the **top** of each iteration.
- This guarantees the **current job always completes** before the process exits. A job started just before SIGTERM will finish; no new job is picked up after the signal.
- Docker waits `stop_grace_period` seconds after SIGTERM before sending SIGKILL. As long as your longest job finishes within that window, you get zero dropped work.

### Solution — compose service

```yaml
services:
  worker:
    build:
      context: .
      dockerfile: worker/Dockerfile
    env_file:
      - .env
    stop_grace_period: 30s    # Docker waits up to 30s after SIGTERM before SIGKILL
    depends_on:
      redis:
        condition: service_healthy
    networks:
      - app_net
    restart: unless-stopped
```

### Worker Dockerfile

```dockerfile
FROM python:3.12-slim

RUN groupadd -r worker && useradd -r -g worker worker

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=worker:worker worker/ ./worker/

USER worker

CMD ["python", "-m", "worker.main"]
```

---

## Exercise 3: Test Service in docker-compose

### Solution — `tests/Dockerfile`

```dockerfile
FROM python:3.12-slim

WORKDIR /tests

COPY tests/requirements-test.txt .
RUN pip install --no-cache-dir -r requirements-test.txt

COPY tests/ .

CMD ["pytest", "-v", "--tb=short"]
```

### Solution — `tests/requirements-test.txt`

```
pytest>=8.0.0
httpx>=0.27.0
pytest-asyncio>=0.23.0
anyio[trio]>=4.0.0
```

### Solution — `tests/test_api.py`

```python
# tests/test_api.py
import logging
import os

import httpx
import pytest

logger = logging.getLogger(__name__)

# The base URL is injected via environment variable in the compose service.
API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")


@pytest.fixture(scope="session")
def client() -> httpx.Client:
    """Synchronous HTTP client pointing at the live API service."""
    with httpx.Client(base_url=API_BASE_URL, timeout=10.0) as c:
        yield c


class TestHealth:
    def test_health_returns_200(self, client: httpx.Client) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_body(self, client: httpx.Client) -> None:
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"


class TestMetrics:
    def test_metrics_returns_200(self, client: httpx.Client) -> None:
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_content_type(self, client: httpx.Client) -> None:
        response = client.get("/metrics")
        assert "text/plain" in response.headers["content-type"]

    def test_metrics_contains_request_counter(self, client: httpx.Client) -> None:
        # Hit /health once so the counter is non-zero
        client.get("/health")
        response = client.get("/metrics")
        assert "http_requests_total" in response.text


class TestTasks:
    def test_tasks_returns_200(self, client: httpx.Client) -> None:
        response = client.get("/tasks")
        assert response.status_code == 200

    def test_tasks_response_shape(self, client: httpx.Client) -> None:
        response = client.get("/tasks")
        data = response.json()
        assert "tasks" in data
        assert isinstance(data["tasks"], list)


class TestNotFound:
    def test_unknown_route_returns_404(self, client: httpx.Client) -> None:
        response = client.get("/does-not-exist")
        assert response.status_code == 404
```

### Solution — `docker-compose.yml` test service

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    healthcheck:
      test: ["CMD", "python", "-c",
             "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 5s
      start_period: 10s
      retries: 3
    networks:
      - app_net

  api-tests:
    build:
      context: .
      dockerfile: tests/Dockerfile
    environment:
      API_BASE_URL: "http://api:8000"   # connect to the live api service
    depends_on:
      api:
        condition: service_healthy      # only start when api passes health check
    networks:
      - app_net
    profiles:
      - test                            # excluded from default `docker compose up`

networks:
  app_net:
    driver: bridge
```

### How to run the tests

```bash
# Run tests against the live API (starts api if not already running):
docker compose --profile test run --rm api-tests

# Or start everything including tests in one command:
docker compose --profile test up --abort-on-container-exit --exit-code-from api-tests
```

### Why `profiles: ["test"]`?

Without `profiles`, every `docker compose up` would run the test suite. Profiles let you opt in:

- `docker compose up` — starts `api`, worker, redis, etc. — no tests.
- `docker compose --profile test run --rm api-tests` — starts tests (and their `depends_on` chain).

### Why `condition: service_healthy` on `depends_on`?

A `service_started` condition (the default) only waits for the container to start, not for the app inside to be ready. If the API is still loading when tests run, every request fails with `Connection refused`. `service_healthy` waits for the HEALTHCHECK to return healthy — guaranteeing the app is actually serving traffic before the first test fires.
