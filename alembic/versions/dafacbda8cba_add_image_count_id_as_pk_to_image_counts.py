"""Add image_count_id as PK to image_counts

Revision ID: dafacbda8cba
Revises: b7397af234d3
Create Date: 2025-07-03 14:44:38.232640

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'dafacbda8cba'
down_revision: Union[str, None] = 'b7397af234d3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column('image_counts', sa.Column('image_count_id',
                  sa.Integer(), primary_key=True, autoincrement=True))


def downgrade():
    op.drop_column('image_counts', 'image_count_id')
