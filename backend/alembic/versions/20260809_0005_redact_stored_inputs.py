"""Remove unredacted user input from stored analysis results."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260809_0005"
down_revision: str | None = "20260802_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    history = sa.table(
        "analysis_history",
        sa.column("id", sa.Integer()),
        sa.column("input_preview", sa.String(length=500)),
        sa.column("analysis_result", sa.JSON()),
    )
    connection = op.get_bind()
    rows = connection.execute(
        sa.select(history.c.id, history.c.input_preview, history.c.analysis_result).where(
            history.c.analysis_result.is_not(None)
        )
    )

    for row in rows:
        result = dict(row.analysis_result)
        if "inputText" not in result:
            continue
        result["inputText"] = row.input_preview
        connection.execute(
            history.update()
            .where(history.c.id == row.id)
            .values(analysis_result=result)
        )


def downgrade() -> None:
    # Redaction is intentionally irreversible: the removed sensitive text must
    # never be reconstructed or copied back into the database.
    pass
