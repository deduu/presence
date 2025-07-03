"""drop image_path from image_counts

Revision ID: e22f2e59fd0d
Revises: dafacbda8cba
Create Date: 2025-07-03 15:51:23.284360

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e22f2e59fd0d'
down_revision: Union[str, None] = 'dafacbda8cba'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("image_counts") as batch:
        batch.drop_column("image_path")      # ← remove the column


def downgrade() -> None:
    with op.batch_alter_table("image_counts") as batch:
        batch.add_column(
            sa.Column("image_path", sa.String(), nullable=False)
        )
