"""Add Person table and link to Face

Revision ID: cc98b01d007a
Revises: 
Create Date: 2025-06-15 21:28:22.179888
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision: str = 'cc98b01d007a'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'people',
        sa.Column('person_id', sa.Integer(), primary_key=True),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('date_of_birth', sa.DateTime(), nullable=True),
        sa.Column('address', sa.String(), nullable=True),
        sa.Column('contact_number', sa.String(), nullable=True),
    )
    op.add_column('faces', sa.Column('person_id', sa.Integer(), nullable=True))
    op.create_foreign_key(
        'fk_faces_person_id_people',
        source_table='faces',
        referent_table='people',
        local_cols=['person_id'],
        remote_cols=['person_id'],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint('fk_faces_person_id_people', 'faces', type_='foreignkey')
    op.drop_column('faces', 'person_id')
    op.drop_table('people')
