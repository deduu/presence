"""Normalize image table

Revision ID: 34f81b6192ed
Revises: b949b13e3fdb
Create Date: 2025-07-03 13:50:51.725688

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '34f81b6192ed'
down_revision: Union[str, None] = 'b949b13e3fdb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.create_table(
        "images",
        sa.Column("image_id", sa.Integer, primary_key=True, index=True),
        sa.Column("image_path", sa.String, unique=True, nullable=False),
    )

    op.add_column("image_records", sa.Column(
        "image_id", sa.Integer, nullable=True))

    # Add foreign key constraints
    op.create_foreign_key(
        "fk_image_records_image_id_images",
        "image_records", "images",
        ["image_id"], ["image_id"],
        ondelete="CASCADE"
    )
    op.create_foreign_key(
        "fk_image_counts_image_id_images",
        "image_counts", "images",
        ["image_id"], ["image_id"],
        ondelete="CASCADE"
    )

    # ⚠️ Manually migrate data in a separate script after this migration


def downgrade():
    op.drop_constraint("fk_image_counts_image_id_images",
                       "image_counts", type_="foreignkey")
    op.drop_constraint("fk_image_records_image_id_images",
                       "image_records", type_="foreignkey")

    op.drop_column("image_counts", "image_id")
    op.drop_column("image_records", "image_id")

    op.drop_table("images")
