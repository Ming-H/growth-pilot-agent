"""v9: analyses.mode column

Revision ID: 0001_v9_mode
Revises: —
Create Date: 2026-10-28

The orchestration mode (agent | workflow) is persisted per analysis so
HITL resume fetches the same graph topology that produced the checkpoint.
This is the project's first Alembic migration — pre-v9 databases created
via create_all() should stamp this revision after adding the column.
"""

from alembic import op
import sqlalchemy as sa

revision = "0001_v9_mode"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("analyses") as batch:
        batch.add_column(
            sa.Column("mode", sa.String(length=16), nullable=False, server_default="agent")
        )


def downgrade() -> None:
    with op.batch_alter_table("analyses") as batch:
        batch.drop_column("mode")
