"""Add analysis history integrity constraints and query index."""

from collections.abc import Sequence

from alembic import op

revision: str = "20260726_0002"
down_revision: str | None = "20260726_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("analysis_history") as batch_op:
        batch_op.create_check_constraint(
            "ck_analysis_history_risk_score_range",
            "risk_score >= 0 AND risk_score <= 1",
        )
        batch_op.create_check_constraint(
            "ck_analysis_history_risk_level",
            "risk_level IN ('low', 'medium', 'high')",
        )
        batch_op.create_check_constraint(
            "ck_analysis_history_status",
            "status IN ('success', 'degraded')",
        )
        batch_op.create_check_constraint(
            "ck_analysis_history_ensemble_counts",
            "ensemble_available >= 0 AND ensemble_total >= 0 "
            "AND ensemble_available <= ensemble_total",
        )
        batch_op.create_index(
            "ix_analysis_history_user_created_at",
            ["user_id", "created_at"],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("analysis_history") as batch_op:
        batch_op.drop_index("ix_analysis_history_user_created_at")
        batch_op.drop_constraint(
            "ck_analysis_history_ensemble_counts", type_="check"
        )
        batch_op.drop_constraint("ck_analysis_history_status", type_="check")
        batch_op.drop_constraint("ck_analysis_history_risk_level", type_="check")
        batch_op.drop_constraint(
            "ck_analysis_history_risk_score_range", type_="check"
        )
