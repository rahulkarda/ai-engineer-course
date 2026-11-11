# Solutions — Week 08 · Day 2

## Exercise 1

**RFC 7807 Problem Details — Pydantic v2 model + FastAPI exception handler**

```python
import logging
from typing import Any
from urllib.parse import urljoin

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RFC 7807 Problem Details model
# ---------------------------------------------------------------------------

class ProblemDetail(BaseModel):
    """
    RFC 7807 Problem Details for HTTP APIs.
    https://datatracker.ietf.org/doc/html/rfc7807
    """

    type: str = Field(
        default="about:blank",
        description="URI reference identifying the problem type.",
    )
    title: str = Field(
        description="Short, human-readable summary of the problem type."
    )
    status: int = Field(
        description="HTTP status code for this occurrence."
    )
    detail: str | None = Field(
        default=None,
        description="Human-readable explanation specific to this occurrence.",
    )
    instance: str | None = Field(
        default=None,
        description="URI reference that identifies the specific occurrence.",
    )
    # Extension members — RFC 7807 allows arbitrary additional fields
    errors: list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional list of field-level validation errors.",
    )

    model_config = {"extra": "allow"}  # permit extension members


# ---------------------------------------------------------------------------
# Domain exceptions
# ---------------------------------------------------------------------------

class AppException(Exception):
    def __init__(
        self,
        status: int,
        title: str,
        detail: str | None = None,
        type_uri: str = "about:blank",
        errors: list[dict[str, Any]] | None = None,
    ) -> None:
        self.status = status
        self.title = title
        self.detail = detail
        self.type_uri = type_uri
        self.errors = errors
        super().__init__(detail or title)


class NotFoundError(AppException):
    def __init__(self, resource: str, resource_id: Any) -> None:
        super().__init__(
            status=404,
            title="Resource Not Found",
            detail=f"{resource} with id={resource_id} does not exist.",
            type_uri="/problems/not-found",
        )


class ValidationError(AppException):
    def __init__(self, errors: list[dict[str, Any]]) -> None:
        super().__init__(
            status=422,
            title="Validation Failed",
            detail="One or more fields failed validation.",
            type_uri="/problems/validation-error",
            errors=errors,
        )


# ---------------------------------------------------------------------------
# FastAPI exception handlers
# ---------------------------------------------------------------------------

app = FastAPI()


@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    problem = ProblemDetail(
        type=exc.type_uri,
        title=exc.title,
        status=exc.status,
        detail=exc.detail,
        instance=str(request.url),
        errors=exc.errors,
    )
    logger.warning(
        "AppException status=%d title=%s path=%s",
        exc.status,
        exc.title,
        request.url.path,
    )
    return JSONResponse(
        status_code=exc.status,
        content=problem.model_dump(exclude_none=True),
        media_type="application/problem+json",  # RFC 7807 content-type
    )


# ---------------------------------------------------------------------------
# Example endpoint
# ---------------------------------------------------------------------------

BOOKS: dict[int, dict[str, Any]] = {1: {"id": 1, "title": "Clean Code"}}


@app.get("/books/{book_id}")
def get_book(book_id: int) -> dict[str, Any]:
    if book_id not in BOOKS:
        raise NotFoundError("Book", book_id)
    return BOOKS[book_id]
```

**Why this works:**

- `Content-Type: application/problem+json` signals RFC 7807 to clients and API gateways — they can branch on the content-type rather than inspecting the body structure.
- `type` is a URI, not a free-form string. Clients can `GET` that URI for machine-readable documentation.
- `instance` is the full request URL — useful for correlating errors in logs without needing a separate request-id header (though adding one is recommended for high-traffic services).
- `errors` is an extension member for field-level detail, common for 422 responses.
- The handler is registered at the app level, so any layer (service, repository) can raise `AppException` and get a consistent error shape without boilerplate in every route.

---

## Exercise 2

**Cursor-based pagination — base64-encoded last-seen ID**

```python
import base64
import logging
from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class BookOut(BaseModel):
    id: int
    title: str
    author: str


class PageOut(BaseModel):
    items: list[BookOut]
    has_next: bool
    next_cursor: str | None  # None when on the last page


# ---------------------------------------------------------------------------
# Cursor helpers
# ---------------------------------------------------------------------------

def encode_cursor(last_id: int) -> str:
    """Encode last-seen ID as an opaque base64 cursor."""
    return base64.urlsafe_b64encode(str(last_id).encode()).decode()


def decode_cursor(cursor: str) -> int:
    """Decode cursor back to integer ID. Raises ValueError on bad input."""
    try:
        return int(base64.urlsafe_b64decode(cursor.encode()).decode())
    except Exception as exc:
        raise ValueError(f"Invalid cursor: {cursor!r}") from exc


# ---------------------------------------------------------------------------
# Simulated data store (replace with real DB query)
# ---------------------------------------------------------------------------

BOOKS_DB: list[dict[str, Any]] = [
    {"id": i, "title": f"Book {i}", "author": f"Author {i}"}
    for i in range(1, 101)
]


def fetch_books_after(after_id: int | None, limit: int) -> list[dict[str, Any]]:
    """
    Keyset query: SELECT * FROM books WHERE id > after_id ORDER BY id LIMIT limit+1
    Fetching limit+1 rows lets us detect whether a next page exists.
    """
    results = [b for b in BOOKS_DB if after_id is None or b["id"] > after_id]
    return results[: limit + 1]


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.get("/books", response_model=PageOut)
def list_books(
    cursor: str | None = Query(default=None, description="Opaque pagination cursor"),
    limit: int = Query(default=20, ge=1, le=100),
) -> PageOut:
    after_id: int | None = None
    if cursor is not None:
        try:
            after_id = decode_cursor(cursor)
        except ValueError:
            logger.warning("Invalid cursor received: %s", cursor)
            after_id = None

    rows = fetch_books_after(after_id, limit)
    has_next = len(rows) > limit
    page_rows = rows[:limit]

    next_cursor: str | None = None
    if has_next and page_rows:
        next_cursor = encode_cursor(page_rows[-1]["id"])

    logger.debug("Returning %d books has_next=%s", len(page_rows), has_next)
    return PageOut(
        items=[BookOut(**r) for r in page_rows],
        has_next=has_next,
        next_cursor=next_cursor,
    )
```

**Why this works:**

- Fetching `limit + 1` rows is the standard "peek-ahead" trick: if we get back more than `limit`, a next page exists. We slice to `limit` before returning.
- The cursor is opaque to clients (base64-encoded). This decouples the client from the internal ID type — you can change from integer IDs to UUIDs or composite keys without breaking the API contract.
- `urlsafe_b64encode` avoids `+` and `/` characters that would require URL-encoding in query strings.
- Keyset navigation is O(1) at any depth because the database uses the primary-key index directly, unlike `OFFSET` which scans and discards rows.

---

## Exercise 3

**Offset vs Keyset pagination — analysis for a 50M-row events table**

**Offset pagination** (`LIMIT n OFFSET k`) is simple to implement and supports random access ("jump to page 50"), but degrades severely at scale. On a 50M-row events table, `OFFSET 5_000_000` forces the database to read and discard five million rows before returning results. Query time grows linearly with the page number, typically reaching seconds for deep pages. Additionally, rows inserted between requests cause "row drift": a row inserted before the current window shifts all subsequent rows, causing duplicates or omissions across pages. Offset pagination is appropriate for small, read-only datasets or admin interfaces where deep paging is rare.

**Keyset pagination** (also called cursor-based) queries `WHERE id > last_seen_id ORDER BY id LIMIT n`. Because `id` is indexed, the database jumps directly to the starting point in O(log n) time regardless of depth. A scan of page 2_500_000 is as fast as page 1. Row insertions do not affect already-delivered cursors — the client's position in the stream is stable. The tradeoff is that keyset pagination does not support arbitrary page jumps; clients must walk pages sequentially. Sorting on non-unique columns requires a composite cursor (e.g., `(created_at, id)`) to remain stable.

**Recommendation for a 50M-row events table:** use keyset pagination. Events are typically consumed in forward-sequential order (feeds, audit logs, real-time streams). Random access ("go to page 47,000") has no user-facing meaning for event data. The performance and consistency benefits of keyset pagination vastly outweigh the loss of arbitrary navigation.

---

## Exercise 4

**HATEOAS links on a `BookReservation` response**

```python
import logging
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HATEOAS link model
# ---------------------------------------------------------------------------

class HttpMethod(StrEnum):
    GET = "GET"
    POST = "POST"
    DELETE = "DELETE"
    PATCH = "PATCH"
    PUT = "PUT"


class Link(BaseModel):
    href: str
    method: HttpMethod
    rel: str  # relation type, e.g. "self", "cancel", "extend"


# ---------------------------------------------------------------------------
# Domain model with embedded links
# ---------------------------------------------------------------------------

class BookReservation(BaseModel):
    reservation_id: int
    book_id: int
    user_id: int
    status: str  # "active" | "cancelled" | "expired"
    due_date: str  # ISO-8601 date string

    # _links is the conventional HATEOAS envelope key
    links: list[Link] = Field(default_factory=list, alias="_links")

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def populate_links(self) -> "BookReservation":
        """Build links from the current state of the resource."""
        base = f"/reservations/{self.reservation_id}"
        computed: list[Link] = [
            Link(href=base, method=HttpMethod.GET, rel="self"),
        ]
        if self.status == "active":
            computed.append(
                Link(href=base, method=HttpMethod.DELETE, rel="cancel")
            )
            computed.append(
                Link(
                    href=f"{base}/extension",
                    method=HttpMethod.POST,
                    rel="extend",
                )
            )
        self.links = computed
        logger.debug(
            "Links computed for reservation_id=%d status=%s count=%d",
            self.reservation_id,
            self.status,
            len(computed),
        )
        return self


# ---------------------------------------------------------------------------
# Example serialisation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    reservation = BookReservation(
        reservation_id=42,
        book_id=7,
        user_id=3,
        status="active",
        due_date="2027-01-15",
    )
    print(json.dumps(reservation.model_dump(by_alias=True), indent=2))
```

**Expected output:**
```json
{
  "reservation_id": 42,
  "book_id": 7,
  "user_id": 3,
  "status": "active",
  "due_date": "2027-01-15",
  "_links": [
    {"href": "/reservations/42", "method": "GET",    "rel": "self"},
    {"href": "/reservations/42", "method": "DELETE", "rel": "cancel"},
    {"href": "/reservations/42/extension", "method": "POST", "rel": "extend"}
  ]
}
```

**Why this works:**

- Links are computed in a `model_validator` on `mode="after"` so they always reflect the current `status`. A cancelled reservation correctly omits `cancel` and `extend`.
- `rel` (relation type) is the key field: clients look up a link by `rel`, not by position. This means the server can reorder or add links without breaking clients.
- `DELETE /reservations/{id}` for cancel follows REST semantics — cancel is logically destroying the reservation resource.
- `POST /reservations/{id}/extension` models the extension as a sub-resource creation event, which allows auditing multiple extension requests over time.
- `by_alias=True` serialises the field as `_links` (the HAL-style convention most clients expect), while internally the field is named `links` to avoid Python's name-mangling rules for double-underscore prefixes.
