"""Add ON DELETE CASCADE to face_id in image_records

Revision ID: a77cdcaea797
Revises: b9f4c4ba4e20
Create Date: 2025-07-02 16:41:37.248557

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'a77cdcaea797'
down_revision: Union[str, None] = 'b9f4c4ba4e20'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Drop the old foreign key constraint
    op.drop_constraint('image_records_face_id_fkey',
                       'image_records', type_='foreignkey')

    # Create a new one with ON DELETE CASCADE
    op.create_foreign_key(
        'image_records_face_id_fkey',
        'image_records',
        'faces',
        ['face_id'],
        ['face_id'],
        ondelete='CASCADE'
    )


def downgrade():
    # Drop the cascade version
    op.drop_constraint('image_records_face_id_fkey',
                       'image_records', type_='foreignkey')

    # Recreate the original one without ON DELETE CASCADE
    op.create_foreign_key(
        'image_records_face_id_fkey',
        'image_records',
        'faces',
        ['face_id'],
        ['face_id']
    )
