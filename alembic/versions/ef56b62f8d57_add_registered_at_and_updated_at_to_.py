"""Add registered_at and updated_at to Person

Revision ID: ef56b62f8d57
Revises: 7ad01551c4b5
Create Date: 2025-06-27 21:09:27.048348

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'ef56b62f8d57'
down_revision: Union[str, None] = '7ad01551c4b5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    op.add_column('people',
                  sa.Column('registered_at', sa.DateTime(timezone=True),
                            server_default=sa.text('CURRENT_TIMESTAMP'))
                  )
    op.add_column('people',
                  sa.Column('updated_at', sa.DateTime(
                      timezone=True), nullable=True)
                  )
    # Alter 'first_seen' and 'last_seen' to use server-side default timestamp
    op.alter_column('faces', 'first_seen',
                    existing_type=sa.DateTime(timezone=True),
                    server_default=sa.text('CURRENT_TIMESTAMP'),
                    existing_nullable=False
                    )
    op.alter_column('faces', 'last_seen',
                    existing_type=sa.DateTime(timezone=True),
                    server_default=sa.text('CURRENT_TIMESTAMP'),
                    existing_nullable=False
                    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('people', 'updated_at')
    op.drop_column('people', 'registered_at')
    # Revert back to no server default (note: timezone-aware Python-side default can't be restored here)
    op.alter_column('faces', 'first_seen',
                    existing_type=sa.DateTime(timezone=True),
                    server_default=None,
                    existing_nullable=False
                    )
    op.alter_column('faces', 'last_seen',
                    existing_type=sa.DateTime(timezone=True),
                    server_default=None,
                    existing_nullable=False
                    )
