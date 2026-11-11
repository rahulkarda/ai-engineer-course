# Week 4 Day 7 — Pydantic v2: Solutions

## Exercise 1: Transaction Model with Field & Model Validators

```python
from decimal import Decimal
from typing import Literal
import logging

from pydantic import BaseModel, field_validator, model_validator, Field
from pydantic import ConfigDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Transaction(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    id: str = Field(..., min_length=1, description="Unique transaction identifier")
    amount: Decimal = Field(..., description="Transaction amount")
    currency: str = Field(..., description="ISO 4217 currency code")
    status: Literal["pending", "completed", "failed"] = "pending"
    merchant: str = Field(default="", description="Merchant name")

    @field_validator("amount")
    @classmethod
    def amount_must_be_positive(cls, v: Decimal) -> Decimal:
        if v <= Decimal("0"):
            raise ValueError(f"amount must be positive, got {v}")
        return v

    @field_validator("currency")
    @classmethod
    def currency_must_be_valid_iso(cls, v: str) -> str:
        v = v.upper().strip()
        if len(v) != 3 or not v.isalpha():
            raise ValueError(f"currency must be a 3-character alphabetic code, got {v!r}")
        return v

    @model_validator(mode="after")
    def merchant_required_when_completed(self) -> "Transaction":
        if self.status == "completed" and not self.merchant:
            raise ValueError("merchant must be non-empty when status is 'completed'")
        return self


# --- Tests ---
from decimal import Decimal
from pydantic import ValidationError

valid = Transaction(id="txn-001", amount=Decimal("99.99"), currency="usd", status="completed", merchant="Acme")
assert valid.currency == "USD"

try:
    Transaction(id="txn-002", amount=Decimal("-5"), currency="USD")
except ValidationError as e:
    logger.error("Caught: %s", e)

try:
    Transaction(id="txn-003", amount=Decimal("10"), currency="USD", status="completed", merchant="")
except ValidationError as e:
    logger.error("Caught: %s", e)
```

---

## Exercise 2: model_dump(), model_dump_json(), model_validate() Round-trip

```python
from decimal import Decimal
from pydantic import BaseModel, field_serializer
import json


class Transaction(BaseModel):
    id: str
    amount: Decimal
    currency: str

    @field_serializer("amount")
    def serialize_amount(self, v: Decimal) -> str:
        # Store as string to preserve precision across JSON
        return str(v)


txn = Transaction(id="txn-001", amount=Decimal("123.45"), currency="EUR")

# --- model_dump() → plain dict ---
d = txn.model_dump()
assert isinstance(d["amount"], str)       # serializer applied
print("dict:", d)

# --- model_dump_json() → JSON string ---
json_str = txn.model_dump_json()
assert isinstance(json_str, str)
parsed = json.loads(json_str)
assert parsed["currency"] == "EUR"
print("json:", json_str)

# --- model_validate() → reconstruct from dict ---
restored_from_dict = Transaction.model_validate(d)
assert restored_from_dict.amount == txn.amount   # Decimal precision preserved
print("restored from dict:", restored_from_dict)

# --- model_validate_json() → reconstruct from JSON string ---
restored_from_json = Transaction.model_validate_json(json_str)
assert restored_from_json.id == txn.id
print("restored from json:", restored_from_json)

# --- Round-trip identity check ---
assert txn.model_dump() == Transaction.model_validate(txn.model_dump()).model_dump()
print("Round-trip OK")
```

---

## Exercise 3: Nested Model — Order with List[Transaction] and ShippingAddress

```python
from decimal import Decimal
from typing import Annotated
from pydantic import BaseModel, Field, computed_field
from datetime import datetime


class Transaction(BaseModel):
    id: str
    amount: Decimal
    currency: str = "USD"
    description: str = ""


class ShippingAddress(BaseModel):
    street: str
    city: str
    state: str = Field(..., min_length=2, max_length=2)   # US state code
    zip_code: str = Field(..., pattern=r"^\d{5}(-\d{4})?$")
    country: str = "US"


class Order(BaseModel):
    order_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    transactions: list[Transaction] = Field(default_factory=list)
    shipping_address: ShippingAddress

    @computed_field   # type: ignore[misc]
    @property
    def total_amount(self) -> Decimal:
        return sum((t.amount for t in self.transactions), Decimal("0"))

    @computed_field   # type: ignore[misc]
    @property
    def transaction_count(self) -> int:
        return len(self.transactions)


# --- Usage ---
order = Order(
    order_id="ORD-2024-001",
    transactions=[
        Transaction(id="t1", amount=Decimal("49.99"), description="Widget A"),
        Transaction(id="t2", amount=Decimal("19.99"), description="Widget B"),
    ],
    shipping_address=ShippingAddress(
        street="123 Main St",
        city="Springfield",
        state="IL",
        zip_code="62701",
    ),
)

assert order.total_amount == Decimal("69.98")
assert order.transaction_count == 2
print(order.model_dump_json(indent=2))
```

---

## Exercise 4: BaseSettings — Reading from Environment Variables

```python
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """
    Application settings sourced from environment variables.
    .env file is loaded as a fallback (requires python-dotenv).
    """
    model_config = SettingsConfigDict(
        env_prefix="APP_",          # all vars prefixed with APP_
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database
    db_host: str = Field(default="localhost", description="PostgreSQL hostname")
    db_port: int = Field(default=5432, ge=1, le=65535)
    db_name: str = Field(default="mydb")
    db_password: SecretStr = Field(default=SecretStr(""), description="DB password")

    # App
    debug: bool = False
    log_level: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    api_key: SecretStr = Field(default=SecretStr(""))


# With no env vars set, all defaults apply:
settings = AppSettings()
print(f"host={settings.db_host}, port={settings.db_port}")

# SecretStr hides the value in repr/logs:
print(settings.db_password)              # **********
print(settings.db_password.get_secret_value())   # actual value

# In tests: override with _env_file=None to avoid reading .env
test_settings = AppSettings(_env_file=None, db_host="test-db", debug=True)
assert test_settings.debug is True
assert test_settings.db_host == "test-db"
```

**Required `.env.example`:**

```
APP_DB_HOST=localhost
APP_DB_PORT=5432
APP_DB_NAME=mydb
APP_DB_PASSWORD=changeme
APP_DEBUG=false
APP_LOG_LEVEL=INFO
APP_API_KEY=your-api-key-here
```

---

## Exercise 5: ValidationError — Catch, Iterate, Log

```python
from decimal import Decimal
import logging
from pydantic import BaseModel, field_validator, ValidationError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Transaction(BaseModel):
    id: str
    amount: Decimal
    currency: str

    @field_validator("amount")
    @classmethod
    def positive(cls, v: Decimal) -> Decimal:
        if v <= 0:
            raise ValueError("must be positive")
        return v

    @field_validator("currency")
    @classmethod
    def valid_currency(cls, v: str) -> str:
        if len(v) != 3:
            raise ValueError("must be 3 characters")
        return v.upper()


def safe_parse_transaction(data: dict) -> Transaction | None:
    try:
        return Transaction.model_validate(data)
    except ValidationError as exc:
        for error in exc.errors():
            field_path = " -> ".join(str(loc) for loc in error["loc"])
            msg = error["msg"]
            err_type = error["type"]
            input_val = error.get("input")
            logger.error(
                "Validation error | field=%r | type=%s | msg=%s | input=%r",
                field_path,
                err_type,
                msg,
                input_val,
            )
        return None


# Trigger multiple errors at once
result = safe_parse_transaction({"id": "x", "amount": -10, "currency": "US"})
assert result is None

# Valid data
result = safe_parse_transaction({"id": "txn-1", "amount": "50.00", "currency": "usd"})
assert result is not None
assert result.currency == "USD"
```

---

## Exercise 6: Discriminated Union — Shape = Circle | Rectangle | Triangle

```python
from typing import Literal, Annotated, Union
from math import pi
from pydantic import BaseModel, Field


class Circle(BaseModel):
    type: Literal["circle"]
    radius: float = Field(..., gt=0)

    def area(self) -> float:
        return pi * self.radius ** 2

    def perimeter(self) -> float:
        return 2 * pi * self.radius


class Rectangle(BaseModel):
    type: Literal["rectangle"]
    width: float = Field(..., gt=0)
    height: float = Field(..., gt=0)

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)


class Triangle(BaseModel):
    type: Literal["triangle"]
    base: float = Field(..., gt=0)
    height: float = Field(..., gt=0)
    side_a: float = Field(..., gt=0)
    side_b: float = Field(..., gt=0)

    def area(self) -> float:
        return 0.5 * self.base * self.height

    def perimeter(self) -> float:
        return self.base + self.side_a + self.side_b


# Discriminated union on the "type" literal field
Shape = Annotated[
    Union[Circle, Rectangle, Triangle],
    Field(discriminator="type"),
]


class Canvas(BaseModel):
    shapes: list[Shape]


# --- Usage ---
canvas = Canvas(
    shapes=[
        {"type": "circle", "radius": 5},
        {"type": "rectangle", "width": 4, "height": 6},
        {"type": "triangle", "base": 3, "height": 4, "side_a": 4, "side_b": 5},
    ]
)

for shape in canvas.shapes:
    print(f"{shape.type}: area={shape.area():.2f}, perimeter={shape.perimeter():.2f}")

# Type narrowing after isinstance check:
circle = canvas.shapes[0]
assert isinstance(circle, Circle)
assert abs(circle.area() - pi * 25) < 1e-9
```

**Key insight:** Discriminated unions allow Pydantic to select the correct model immediately based on the discriminator field value, rather than trying each model in sequence. This is faster, produces clearer error messages, and is required by many JSON Schema tools.
