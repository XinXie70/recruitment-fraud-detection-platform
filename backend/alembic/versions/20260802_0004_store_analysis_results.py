"""Store complete analysis results for cross-device history access."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260802_0004"
down_revision: str | None = "20260730_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "analysis_history",
        sa.Column("analysis_result", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("analysis_history", "analysis_result")
