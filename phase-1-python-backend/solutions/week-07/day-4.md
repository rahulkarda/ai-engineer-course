# Week 07 · Day 4 — Solutions

JWT authentication, authorization, and async SQLAlchemy.

---

## Exercise 1 — `create_access_token`

Implement a token factory using `python-jose`. `SECRET_KEY` and `ALGORITHM` come
exclusively from the environment — never hard-coded.

```python
# auth/tokens.py
import os
from datetime import datetime, timedelta, timezone

from jose import jwt

# Read from environment at module load; will be None in test isolation
# unless overridden via monkeypatch / pytest fixture.
SECRET_KEY: str | None = os.getenv("JWT_SECRET_KEY")
ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")


def create_access_token(data: dict, expires_delta: timedelta) -> str:
    """
    Encode *data* as a signed JWT with an absolute expiry.

    Args:
        data:          Claims to embed (typically {"sub": user_id, "role": role}).
        expires_delta: How long the token should remain valid.

    Returns:
        A signed JWT string.

    Raises:
        RuntimeError: If JWT_SECRET_KEY is not set in the environment.
    """
    if not SECRET_KEY:
        raise RuntimeError(
            "JWT_SECRET_KEY environment variable is not set. "
            "Set it before starting the application."
        )

    payload = data.copy()
    expire = datetime.now(tz=timezone.utc) + expires_delta
    payload.update({"exp": expire})

    encoded_jwt: str = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
```

**Usage example** (inside a login endpoint):

```python
from datetime import timedelta
from auth.tokens import create_access_token

ACCESS_TOKEN_EXPIRE_MINUTES = 30

@router.post("/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect credentials")

    token = create_access_token(
        data={"sub": str(user.id), "role": user.role},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": token, "token_type": "bearer"}
```

---

## Exercise 2 — `get_current_user` dependency

Extract and validate a JWT from the `Authorization: Bearer <token>` header.

```python
# auth/dependencies.py
import os
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from db.session import get_db
from models.user import User

SECRET_KEY: str | None = os.getenv("JWT_SECRET_KEY")
ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")

# FastAPI's built-in scheme: reads the Bearer token and feeds it to the dependency.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    session: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """
    Validate the JWT and return the corresponding User from the database.

    Raises:
        401 UNAUTHORIZED — token missing, malformed, expired, or subject not found.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not SECRET_KEY:
        raise RuntimeError("JWT_SECRET_KEY is not configured.")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str | None = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    result = await session.execute(select(User).where(User.id == int(user_id)))
    user: User | None = result.scalar_one_or_none()

    if user is None:
        raise credentials_exception

    return user
```

---

## Exercise 3 — `require_role` dependency factory

A *factory* that returns a FastAPI dependency. The inner dependency reuses
`get_current_user` via `Depends`.

```python
# auth/dependencies.py (continued)
from collections.abc import Callable


def require_role(role: str) -> Callable:
    """
    Return a dependency that enforces *role* on the current user.

    Usage:
        @router.delete(
            "/admin/users/{user_id}",
            dependencies=[Depends(require_role("admin"))],
        )

    Args:
        role: The exact role string the authenticated user must have.

    Returns:
        A FastAPI-compatible async dependency function.

    Raises:
        403 FORBIDDEN if the user's role does not match.
    """

    async def _check_role(
        current_user: Annotated[User, Depends(get_current_user)],
    ) -> User:
        if current_user.role != role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation requires role '{role}'.",
            )
        return current_user

    return _check_role


# ---------------------------------------------------------------------------
# Convenience aliases used throughout the application:
# ---------------------------------------------------------------------------
CurrentUser = Annotated[User, Depends(get_current_user)]
AdminUser = Annotated[User, Depends(require_role("admin"))]
```

**Endpoint example:**

```python
@router.get("/admin/stats")
async def admin_stats(current_user: AdminUser) -> dict:
    return {"total_users": await count_users()}
```

---

## Exercise 4 — Synchronous → Asynchronous SQLAlchemy

### Sync version (before)

```python
# Synchronous — blocks the event loop, do NOT use in async FastAPI endpoints.
from sqlalchemy.orm import Session
from models.customer import Customer


def get_customer_sync(session: Session, customer_id: int) -> Customer | None:
    return (
        session.query(Customer)
        .filter(Customer.id == customer_id)
        .one_or_none()
    )
```

### Async version (after)

```python
# db/session.py
import os
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

DATABASE_URL: str | None = os.getenv("DATABASE_URL")
# asyncpg driver is required for PostgreSQL:
#   pip install asyncpg
# SQLite alternative for tests: "sqlite+aiosqlite:///./test.db"

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL environment variable is not set."
    )

engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)

# async_sessionmaker is the async counterpart of sessionmaker.
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # avoids lazy-load errors after commit
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncSession:
    """FastAPI dependency that yields an AsyncSession."""
    async with AsyncSessionLocal() as session:
        yield session
```

```python
# routers/customers.py
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from models.customer import Customer


async def get_customer_async(
    session: AsyncSession,
    customer_id: int,
) -> Customer | None:
    """
    Async equivalent: uses `await session.execute()` with the 2.0-style
    `select()` construct instead of the legacy `session.query()` API.
    """
    result = await session.execute(
        select(Customer).where(Customer.id == customer_id)
    )
    return result.scalar_one_or_none()
```

Key differences:

| Sync                              | Async                                      |
|-----------------------------------|--------------------------------------------|
| `Session`                         | `AsyncSession`                             |
| `sessionmaker`                    | `async_sessionmaker`                       |
| `session.query(Model).filter(…)`  | `select(Model).where(…)` (2.0 style)       |
| `session.execute(stmt)`           | `await session.execute(stmt)`              |
| `create_engine("postgresql://…")` | `create_async_engine("postgresql+asyncpg://…")` |
