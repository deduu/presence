"""Add unique constraint to Person

Revision ID: b9f4c4ba4e20
Revises: 7ac7acb09c67
Create Date: 2025-07-02 16:10:10.964190

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'b9f4c4ba4e20'
down_revision: Union[str, None] = '7ac7acb09c67'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.create_unique_constraint(
        "uq_person_identity", "people", [
            "name", "date_of_birth", "contact_number"]
    )


def downgrade():
    op.drop_constraint("uq_person_identity", "people", type_="unique")
