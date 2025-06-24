"""Fix date_of_birth column type

Revision ID: 7ad01551c4b5
Revises: aa8c35ad676a
Create Date: 2025-06-24 14:07:12.631615

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '7ad01551c4b5'
down_revision: Union[str, None] = 'aa8c35ad676a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.alter_column(
        'people',
        'date_of_birth',
        existing_type=sa.Date(),  # assume original type was `date`, still represented as `sa.Date`
        type_=sa.Date(),
        existing_nullable=True
    )


def downgrade():
    op.alter_column(
        'people',
        'date_of_birth',
        existing_type=sa.Date(),
        type_=sa.Date(),  # No actual type change, but preserve symmetry
        existing_nullable=True
    )
