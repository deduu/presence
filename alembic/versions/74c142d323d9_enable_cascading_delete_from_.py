"""Enable cascading delete from ImageRecord to Image and ImageCount

Revision ID: 74c142d323d9
Revises: 54d2e0ca184a
Create Date: 2025-07-03 16:40:25.713314

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '74c142d323d9'
down_revision: Union[str, None] = '54d2e0ca184a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Drop old FKs


def upgrade():
    op.drop_constraint('image_records_image_id_fkey',
                       'image_records', type_='foreignkey', if_exists=True)
    op.drop_constraint('image_counts_image_id_fkey',
                       'image_counts', type_='foreignkey', if_exists=True)

    op.create_foreign_key(
        'image_records_image_id_fkey',
        'image_records', 'images',
        ['image_id'], ['image_id'],
        ondelete='CASCADE'
    )
    op.create_foreign_key(
        'image_counts_image_id_fkey',
        'image_counts', 'images',
        ['image_id'], ['image_id'],
        ondelete='CASCADE'
    )


def downgrade():
    # Drop CASCADE constraints
    op.drop_constraint('image_records_image_id_fkey',
                       'image_records', type_='foreignkey')
    op.drop_constraint('image_counts_image_id_fkey',
                       'image_counts', type_='foreignkey')

    # Recreate original constraints WITHOUT CASCADE
    op.create_foreign_key(
        'image_records_image_id_fkey',
        'image_records', 'images',
        ['image_id'], ['image_id']
    )
    op.create_foreign_key(
        'image_counts_image_id_fkey',
        'image_counts', 'images',
        ['image_id'], ['image_id']
    )
