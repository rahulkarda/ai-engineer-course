# Phase 1 · Week 07 · Day 4 — Authentication, Authorization & Async APIs

> Objective: Implement JWT auth, role-based authorization, and write async endpoints  ·  Time: ~1 hr  ·  Prereqs: [[week-07/day-3]]

## Concept

### JWT structure

A JSON Web Token is three base64url-encoded segments joined by dots:

```
header.payload.signature
```

- **Header** — algorithm and token type: `{"alg": "HS256", "typ": "JWT"}`
- **Payload** — claims: registered (`sub`, `exp`, `iat`) plus custom (`role`, `user_id`)
- **Signature** — `HMAC-SHA256(base64url(header) + "." + base64url(payload), secret)`

The signature binds header + payload to a secret. Tampering with the payload invalidates the signature — the server detects it on decode.

`HS256` is a symmetric scheme: the same secret signs and verifies. For service-to-service, `RS256` (asymmetric) is preferred because verifiers only need the public key.

### Password hashing

Passwords must never be stored in plain text or with reversible encryption. `bcrypt` applies a slow one-way hash with a random salt. Cost factor (default 12) controls iterations — each increment doubles work.

```
stored_hash = bcrypt.hash(plain_password)   # includes salt
bcrypt.verify(plain_password, stored_hash)  # returns bool
```

### OAuth2PasswordBearer

FastAPI's `OAuth2PasswordBearer` is a dependency that:
1. Reads the `Authorization: Bearer <token>` header
2. Returns the raw token string
3. Raises `401 Unauthorized` automatically when the header is absent

It is a hint to OpenAPI to show a lock icon and token input — it does not validate the token. Your code validates.

### Role-based authorization

Two-layer dependency chain:

```
get_current_user  →  require_role("admin")
     |                      |
   decodes JWT        checks user.role
   raises 401          raises 403
```

Separating decode from role check keeps each dependency testable in isolation.

## How it works internally

### JWT validation flow

When the server receives a token:
1. Split on `.` — three parts required
2. Verify signature with the secret key — reject if mismatch (tampered)
3. Decode payload — check `exp` claim against current UTC time — reject if expired
4. Extract `sub` (subject) — look up user in DB

`python-jose` / `PyJWT` handle steps 1-3. Step 4 is application logic.

### Async SQLAlchemy execution model

SQLAlchemy 2.0 async uses `asyncio`-native I/O via `asyncpg` (PostgreSQL) or `aiosqlite`. The key rule: **every DB call that touches the network must be awaited**.

```
async_engine  →  async_sessionmaker  →  AsyncSession
                                              |
                              await session.execute(stmt)
                              await session.commit()
```

Under the hood, `await session.execute()` suspends the coroutine, yields control to the event loop, and resumes when the DB driver delivers results. No thread is blocked. FastAPI's ASGI server (Uvicorn/Starlette) runs many such coroutines concurrently on a single thread.

### Dependency injection chain

FastAPI resolves dependencies depth-first. For a protected endpoint:

```
endpoint  →  get_current_active_user  →  get_current_user  →  oauth2_scheme
                                               |
                                         DB session dep
```

Each dependency is called once per request, cached within the request scope by default.

## Worked example

```python
# src/finops/auth.py
import os
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession

from .database import get_session
from .models import User

# --- Config (never hardcode values) ---
SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "")  # must be set in env
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

if not SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY environment variable is not set")

# --- Password hashing ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)

# --- Token creation ---
def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    payload = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload.update({"exp": expire})
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

# --- OAuth2 scheme (reads Authorization: Bearer <token>) ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# --- User lookup dependency ---
async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> User:
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str | None = payload.get("sub")
        if user_id is None:
            raise credentials_exc
    except JWTError:
        raise credentials_exc

    from sqlalchemy import select
    result = await session.execute(select(User).where(User.id == int(user_id)))
    user = result.scalar_one_or_none()
    if user is None:
        raise credentials_exc
    return user

# --- Role-based authorization factory ---
def require_role(*roles: str):
    async def _check(user: Annotated[User, Depends(get_current_user)]) -> User:
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user.role}' is not authorized for this action",
            )
        return user
    return _check
```

```python
# src/finops/routers/auth.py
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import create_access_token, get_current_user, hash_password, verify_password
from ..database import get_session
from ..models import User
from ..schemas import TokenResponse, UserCreate, UserResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse, status_code=201)
async def register(
    body: UserCreate,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> User:
    existing = await session.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")
    user = User(
        email=body.email,
        hashed_password=hash_password(body.password),
        role=body.role or "viewer",
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


@router.post("/login", response_model=TokenResponse)
async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> TokenResponse:
    result = await session.execute(select(User).where(User.email == form.username))
    user = result.scalar_one_or_none()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token({"sub": str(user.id), "role": user.role})
    return TokenResponse(access_token=token, token_type="bearer")


@router.get("/me", response_model=UserResponse)
async def me(current_user: Annotated[User, Depends(get_current_user)]) -> User:
    return current_user


# Role-gated endpoint: only "admin" can call this
from ..auth import require_role

@router.get("/admin-only")
async def admin_dashboard(
    _: Annotated[User, Depends(require_role("admin"))],
) -> dict:
    return {"message": "Welcome, admin"}
```

```python
# src/finops/schemas.py  (auth-related portion)
from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    role: str | None = None


class UserResponse(BaseModel):
    id: int
    email: EmailStr
    role: str

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
```

```python
# src/finops/database.py
import os
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

DATABASE_URL: str = os.getenv("DATABASE_URL", "")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
```

```python
# src/finops/models.py  (User model)
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(50), default="viewer")
```

### Async DB query — standalone example

```python
# Fetching a user by id with async SQLAlchemy
from sqlalchemy import select

async def get_user_by_id(user_id: int, session: AsyncSession) -> User | None:
    result = await session.execute(
        select(User).where(User.id == user_id)
    )
    return result.scalar_one_or_none()
    # scalar_one_or_none():  returns the single value or None; raises if >1 row
    # scalar_one():          raises if 0 or >1 rows
    # scalars().all():       returns list of ORM objects
```

## Exercises

1. The `create_access_token` function currently uses a fixed expiry. Add a `refresh_token` endpoint at `POST /auth/refresh` that accepts a valid access token and returns a new one with a fresh expiry — without requiring the user to re-enter their password.

2. Extend the `require_role` dependency to also accept a list of roles from the JWT payload's `"permissions"` claim (a list of strings). A user passes if `user.role` matches OR if any of their permissions match a required permission string you pass to the factory.

3. The current `GET /auth/me` returns `hashed_password` through the ORM object. Confirm why the current schema prevents this leak, then deliberately break it (add `hashed_password` to `UserResponse`) and write a test that asserts the field is absent in the response. Revert after.

4. Replace the synchronous `pwd_context.hash()` call in `register` with `asyncio.to_thread(pwd_context.hash, body.password)`. Benchmark the difference under 50 concurrent registration requests using `httpx.AsyncClient` and explain why offloading to a thread pool matters for bcrypt specifically.

(Solutions: ../solutions/week-07/day-4.md)

## Self-check

- [ ] I can explain JWT header/payload/signature and what the signature guarantees
- [ ] I can implement bcrypt password hashing and verification with passlib
- [ ] I can wire `OAuth2PasswordBearer` and `get_current_user` as a FastAPI dependency
- [ ] I can build a role-check dependency factory and apply it to a route
- [ ] I can write an async SQLAlchemy query using `await session.execute()` and `select()`
