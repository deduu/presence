"""drop record_id from image_counts

Revision ID: 54d2e0ca184a
Revises: e22f2e59fd0d
Create Date: 2025-07-03 15:55:17.194127

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '54d2e0ca184a'
down_revision: Union[str, None] = 'e22f2e59fd0d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    with op.batch_alter_table("image_counts") as batch:
        # or batch.alter_column(..., nullable=True)
        batch.drop_column("record_id")


def downgrade():
    with op.batch_alter_table("image_counts") as batch:
        batch.add_column(
            # nullable so future upgrades work
            sa.Column("record_id", sa.Integer(), nullable=True)
        )
