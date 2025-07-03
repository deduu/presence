"""Cascade deletes person→face & face→image_record

Revision ID: 1ad36769f250
Revises: 3a84f1926f36
Create Date: 2025-07-02 21:17:47.929847

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1ad36769f250'
down_revision: Union[str, None] = '3a84f1926f36'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # faces.person_id → people.person_id
    op.drop_constraint("faces_person_id_fkey", "faces", type_="foreignkey")
    op.create_foreign_key(
        "faces_person_id_fkey",
        "faces", "people",
        ["person_id"], ["person_id"],
        ondelete="CASCADE",
    )

    # image_records.face_id → faces.face_id
    op.drop_constraint(
        "image_records_face_id_fkey", "image_records", type_="foreignkey"
    )
    op.create_foreign_key(
        "image_records_face_id_fkey",
        "image_records", "faces",
        ["face_id"], ["face_id"],
        ondelete="CASCADE",
    )


def downgrade():
    # Revert to the original (no-cascade) behaviour
    op.drop_constraint("faces_person_id_fkey", "faces", type_="foreignkey")
    op.create_foreign_key(
        "faces_person_id_fkey",
        "faces", "people",
        ["person_id"], ["person_id"],
    )

    op.drop_constraint(
        "image_records_face_id_fkey", "image_records", type_="foreignkey"
    )
    op.create_foreign_key(
        "image_records_face_id_fkey",
        "image_records", "faces",
        ["face_id"], ["face_id"],
    )
