"""Drop image_path from ImageRecord

Revision ID: b7397af234d3
Revises: 34f81b6192ed
Create Date: 2025-07-03 14:32:07.449928

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b7397af234d3'
down_revision: Union[str, None] = '34f81b6192ed'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.drop_column('image_records', 'image_path')


def downgrade():
    op.add_column('image_records', sa.Column(
        'image_path', sa.String(), nullable=True))
