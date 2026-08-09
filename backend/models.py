from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    username: Mapped[str] = mapped_column(String(80), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    analyses: Mapped[list["AnalysisHistory"]] = relationship(
        "AnalysisHistory",
        back_populates="user",
        lazy="selectin",
        passive_deletes=True,
    )


class AnalysisHistory(Base):
    """Persistent record of each analysis request for user history + admin audit."""

    __tablename__ = "analysis_history"
    __table_args__ = (
        CheckConstraint(
            "risk_score >= 0 AND risk_score <= 1",
            name="ck_analysis_history_risk_score_range",
        ),
        CheckConstraint(
            "risk_level IN ('low', 'medium', 'high')",
            name="ck_analysis_history_risk_level",
        ),
        CheckConstraint(
            "status IN ('success', 'degraded')",
            name="ck_analysis_history_status",
        ),
        CheckConstraint(
            "ensemble_available >= 0 AND ensemble_total >= 0 "
            "AND ensemble_available <= ensemble_total",
            name="ck_analysis_history_ensemble_counts",
        ),
        Index("ix_analysis_history_user_created_at", "user_id", "created_at"),
        Index("ix_analysis_history_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    input_preview: Mapped[str] = mapped_column(String(500), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(10), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    ensemble_available: Mapped[int] = mapped_column(Integer, default=0)
    ensemble_total: Mapped[int] = mapped_column(Integer, default=8)
    analysis_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="analyses")
