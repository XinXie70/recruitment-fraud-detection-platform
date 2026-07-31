

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Query


@dataclass(frozen=True)
class Pagination:
    page: int
    size: int

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.size

    @property
    def limit(self) -> int:
        return self.size


def paginate(
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(20, ge=1, le=100, alias="page_size", description="Items per page"),
) -> Pagination:
    return Pagination(page=page, size=page_size)
