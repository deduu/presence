"""Add batch_Tag to ImageRecord

Revision ID: 7ac7acb09c67
Revises: 087c5bbb87f7
Create Date: 2025-07-01 21:43:38.964936

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7ac7acb09c67'
down_revision: Union[str, None] = '087c5bbb87f7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column('image_records', sa.Column(
        'batch_tag', sa.String(), nullable=True))


def downgrade():
    op.drop_column('image_records', 'batch_tag')
