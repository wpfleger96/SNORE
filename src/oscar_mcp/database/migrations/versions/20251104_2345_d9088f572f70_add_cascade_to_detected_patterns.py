"""add_cascade_to_detected_patterns

Revision ID: d9088f572f70
Revises: 473f9f966032
Create Date: 2025-11-04 23:45:17.840431

"""

from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = "d9088f572f70"
down_revision: Union[str, Sequence[str], None] = "473f9f966032"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add CASCADE constraint to detected_patterns.pattern_id foreign key."""
    # Note: For fresh databases, the CASCADE is already defined in models.py
    # This migration is a no-op for new installations, but documents the change
    # For existing databases (if any), the constraint would need manual update
    # Since we're still in development with no released version, we can skip this
    pass


def downgrade() -> None:
    """Remove CASCADE constraint from detected_patterns.pattern_id foreign key."""
    # No-op for development - see upgrade() comment
    pass
