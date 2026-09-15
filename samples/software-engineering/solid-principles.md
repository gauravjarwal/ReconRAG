# SOLID Principles

SOLID is an acronym for five object-oriented design principles introduced by Robert C. Martin (Uncle Bob). Applying these principles leads to software that is easier to maintain, extend, and test.

---

## S — Single Responsibility Principle (SRP)

A class should have only one reason to change, meaning it should have only one job or responsibility.

### Why it matters
When a class has multiple responsibilities, a change to one responsibility can break or unexpectedly affect the others. Isolating responsibilities makes each class easier to understand, test, and modify.

### Example violation

```python
class UserManager:
    def create_user(self, username, email): ...
    def send_welcome_email(self, user): ...   # email is a separate concern
    def save_to_database(self, user): ...     # persistence is a separate concern
```

### Corrected

```python
class UserService:
    def create_user(self, username, email): ...

class EmailService:
    def send_welcome_email(self, user): ...

class UserRepository:
    def save(self, user): ...
```

---

## O — Open/Closed Principle (OCP)

Software entities (classes, modules, functions) should be open for extension but closed for modification.

### Why it matters
Adding new behaviour by modifying existing code risks introducing bugs in already-working functionality. Designing for extension lets new behaviour be added without touching tested code.

### Example

Instead of a switch statement that must be modified for every new payment type:

```python
# Violates OCP
class PaymentProcessor:
    def process(self, payment_type, amount):
        if payment_type == "card": ...
        elif payment_type == "paypal": ...
        # adding crypto requires modifying this class
```

Use an abstract base class and extend it:

```python
from abc import ABC, abstractmethod

class PaymentMethod(ABC):
    @abstractmethod
    def process(self, amount: float) -> None: ...

class CardPayment(PaymentMethod):
    def process(self, amount): ...

class CryptoPayment(PaymentMethod):          # new type — no existing code changed
    def process(self, amount): ...
```

---

## L — Liskov Substitution Principle (LSP)

Objects of a derived class must be substitutable for objects of the base class without altering the correctness of the program.

### Why it matters
Violating LSP means subclasses change the expected behaviour of the base class, leading to subtle bugs when code works with base-class references.

### Classic violation: the Square–Rectangle problem

```python
class Rectangle:
    def set_width(self, w): self.width = w
    def set_height(self, h): self.height = h
    def area(self): return self.width * self.height

class Square(Rectangle):
    def set_width(self, w):
        self.width = w
        self.height = w   # forces height to match — breaks Rectangle's contract

def resize_and_measure(rect: Rectangle):
    rect.set_width(5)
    rect.set_height(10)
    assert rect.area() == 50  # fails for Square!
```

A Square is not behaviourally substitutable for a Rectangle despite the is-a relationship. The fix is to not inherit Square from Rectangle but instead make both implement a Shape interface.

---

## I — Interface Segregation Principle (ISP)

Clients should not be forced to depend on interfaces they do not use.

### Why it matters
Fat interfaces couple unrelated functionality together. When an interface changes in one area, all classes implementing it must be updated even if they only use part of the interface.

### Violation

```python
class Worker(ABC):
    @abstractmethod
    def work(self): ...
    @abstractmethod
    def eat(self): ...   # robots don't eat!

class Robot(Worker):
    def work(self): ...
    def eat(self): raise NotImplementedError  # forced to implement meaningless method
```

### Corrected — segregate into focused interfaces

```python
class Workable(ABC):
    @abstractmethod
    def work(self): ...

class Eatable(ABC):
    @abstractmethod
    def eat(self): ...

class HumanWorker(Workable, Eatable):
    def work(self): ...
    def eat(self): ...

class Robot(Workable):
    def work(self): ...   # no forced eat()
```

---

## D — Dependency Inversion Principle (DIP)

High-level modules should not depend on low-level modules. Both should depend on abstractions. Abstractions should not depend on details; details should depend on abstractions.

### Why it matters
Directly depending on concrete implementations makes code hard to swap (e.g., switching from MySQL to PostgreSQL) and hard to test (cannot inject a mock).

### Violation

```python
class OrderService:
    def __init__(self):
        self.db = MySQLDatabase()   # hard-coded concrete dependency

    def place_order(self, order):
        self.db.save(order)
```

### Corrected — depend on an abstraction

```python
from abc import ABC, abstractmethod

class Database(ABC):
    @abstractmethod
    def save(self, record): ...

class MySQLDatabase(Database):
    def save(self, record): ...

class OrderService:
    def __init__(self, db: Database):    # injected — can be any Database
        self.db = db

    def place_order(self, order):
        self.db.save(order)
```

In tests, inject a `FakeDatabase` or `MockDatabase`. In production, inject `MySQLDatabase` or `PostgresDatabase`. The `OrderService` never needs to change.

---

## Summary Table

| Principle | One-line summary | Key benefit |
|---|---|---|
| SRP | One class, one job | Easier to understand and change |
| OCP | Extend by adding, not editing | Reduces regression risk |
| LSP | Subtypes are substitutable | Reliable polymorphism |
| ISP | Small, focused interfaces | Avoids forced dependencies |
| DIP | Depend on abstractions | Enables testing and flexibility |

---

## Practical Notes

SOLID principles are guidelines, not absolute rules. Blindly applying them everywhere leads to over-engineering. Apply them when:
- A class is hard to test because it does too much (SRP).
- Adding a feature requires modifying multiple files (OCP).
- A subclass breaks an existing test (LSP).
- An interface change propagates to unrelated classes (ISP).
- You cannot test a class in isolation because it instantiates its own dependencies (DIP).
