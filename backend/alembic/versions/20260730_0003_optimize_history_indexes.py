"""Optimize indexes used by analysis history queries."""

from collections.abc import Sequence

from alembic import op

revision: str = "20260730_0003"
down_revision: str | None = "20260726_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_index("ix_users_id", table_name="users")
    op.drop_index("ix_analysis_history_id", table_name="analysis_history")
    op.drop_index("ix_analysis_history_user_id", table_name="analysis_history")
    op.drop_index("ix_analysis_history_input_hash", table_name="analysis_history")
    op.create_index(
        "ix_analysis_history_created_at",
        "analysis_history",
        ["created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_analysis_history_created_at", table_name="analysis_history")
    op.create_index("ix_analysis_history_input_hash", "analysis_history", ["input_hash"])
    op.create_index("ix_analysis_history_user_id", "analysis_history", ["user_id"])
    op.create_index("ix_analysis_history_id", "analysis_history", ["id"])
    op.create_index("ix_users_id", "users", ["id"])
