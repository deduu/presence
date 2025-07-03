"""Enable cascading deletes for face/image_records

Revision ID: 3a84f1926f36
Revises: a77cdcaea797
Create Date: 2025-07-02 20:48:41.647687

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3a84f1926f36'
down_revision: Union[str, None] = 'a77cdcaea797'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Drop the existing foreign key constraint
    op.drop_constraint(
        "image_records_face_id_fkey",
        "image_records",
        type_="foreignkey"
    )

    # Recreate the foreign key constraint with ON DELETE CASCADE
    op.create_foreign_key(
        "image_records_face_id_fkey",
        "image_records",
        "faces",
        ["face_id"],
        ["face_id"],
        ondelete="CASCADE"
    )


def downgrade():
    # Drop the ON DELETE CASCADE foreign key
    op.drop_constraint(
        "image_records_face_id_fkey",
        "image_records",
        type_="foreignkey"
    )

    # Recreate the original foreign key without cascade
    op.create_foreign_key(
        "image_records_face_id_fkey",
        "image_records",
        "faces",
        ["face_id"],
        ["face_id"]
    )
