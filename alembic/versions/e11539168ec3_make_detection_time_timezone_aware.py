"""Make detection_time timezone-aware

Revision ID: e11539168ec3
Revises: ef56b62f8d57
Create Date: 2025-06-28 16:04:07.756202

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e11539168ec3'
down_revision: Union[str, None] = 'ef56b62f8d57'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.alter_column(
        'image_records',
        'detection_time',
        type_=sa.TIMESTAMP(timezone=True),
        existing_type=sa.TIMESTAMP(timezone=False),
        existing_nullable=False
    )


def downgrade():
    op.alter_column(
        'image_records',
        'detection_time',
        type_=sa.TIMESTAMP(timezone=False),
        existing_type=sa.TIMESTAMP(timezone=True),
        existing_nullable=False
    )
