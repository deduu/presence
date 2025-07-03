"""Link image_counts to image_records by record_id

Revision ID: b949b13e3fdb
Revises: 1ad36769f250
Create Date: 2025-07-03 11:52:15.376153

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b949b13e3fdb'
down_revision: Union[str, None] = '1ad36769f250'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column("image_counts", sa.Column(
        "record_id", sa.Integer(), nullable=True))

    # Populate it if possible (if mapping exists)
    # op.execute("UPDATE image_counts SET record_id = ...") ← Optional logic

    # Add FK
    op.create_unique_constraint(
        "uq_image_counts_record_id", "image_counts", ["record_id"])
    op.create_foreign_key(
        "fk_image_counts_record_id_image_records",
        "image_counts",
        "image_records",
        ["record_id"],
        ["record_id"],
        ondelete="CASCADE"
    )

    # Then set column as non-nullable if all rows are backfilled
    op.alter_column("image_counts", "record_id", nullable=False)


def downgrade():
    # Drop the foreign key constraint first
    op.drop_constraint(
        "fk_image_counts_record_id_image_records",
        "image_counts",
        type_="foreignkey"
    )

    # Drop the unique constraint on record_id
    op.drop_constraint(
        "uq_image_counts_record_id",
        "image_counts",
        type_="unique"
    )

    # Drop the column itself
    op.drop_column("image_counts", "record_id")
