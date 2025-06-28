"""Add image_width and image_height to ImageRecord

Revision ID: 087c5bbb87f7
Revises: e11539168ec3
Create Date: 2025-06-28 20:02:36.991567

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '087c5bbb87f7'
down_revision: Union[str, None] = 'e11539168ec3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column('image_records', sa.Column(
        'image_width', sa.Integer(), nullable=True))
    op.add_column('image_records', sa.Column(
        'image_height', sa.Integer(), nullable=True))


def downgrade():
    op.drop_column('image_records', 'image_height')
    op.drop_column('image_records', 'image_width')
