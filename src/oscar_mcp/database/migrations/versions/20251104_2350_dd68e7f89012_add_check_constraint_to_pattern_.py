"""add_check_constraint_to_pattern_relationships

Revision ID: dd68e7f89012
Revises: d9088f572f70
Create Date: 2025-11-04 23:50:14.396370

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "dd68e7f89012"
down_revision: Union[str, Sequence[str], None] = "d9088f572f70"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add CHECK constraint to prevent self-referential pattern relationships."""
    # SQLite requires batch mode to add constraints
    with op.batch_alter_table("pattern_relationships", schema=None) as batch_op:
        batch_op.create_check_constraint(
            "chk_no_self_reference", "from_pattern_id != to_pattern_id"
        )


def downgrade() -> None:
    """Remove CHECK constraint for self-referential patterns."""
    with op.batch_alter_table("pattern_relationships", schema=None) as batch_op:
        batch_op.drop_constraint("chk_no_self_reference", type_="check")
