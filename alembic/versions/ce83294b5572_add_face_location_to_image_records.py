"""Add face_location to image_records

Revision ID: ce83294b5572
Revises: cc98b01d007a
Create Date: 2025-06-20 21:51:55.493218

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'ce83294b5572'
down_revision: Union[str, None] = 'cc98b01d007a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""


    op.add_column('image_records', sa.Column('face_location', sa.String(), nullable=True))




def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('image_records', 'face_location')