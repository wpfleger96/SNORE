"""add_profile_and_day_tables

Revision ID: 0a6f54d621a0
Revises: dd68e7f89012
Create Date: 2025-11-05 00:22:19.072465

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0a6f54d621a0"
down_revision: Union[str, Sequence[str], None] = "dd68e7f89012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create profiles table
    op.create_table(
        "profiles",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("username", sa.String(length=100), nullable=False),
        sa.Column("first_name", sa.String(length=100), nullable=True),
        sa.Column("last_name", sa.String(length=100), nullable=True),
        sa.Column("date_of_birth", sa.Date(), nullable=True),
        sa.Column("height_cm", sa.Integer(), nullable=True),
        sa.Column("settings", sa.Text(), nullable=True),  # ValidatedJSONWithDefault
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("length(username) > 0", name="chk_username"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("username"),
    )

    # Create days table
    op.create_table(
        "days",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("session_count", sa.Integer(), nullable=True),
        sa.Column("total_therapy_hours", sa.Float(), nullable=True),
        sa.Column("obstructive_apneas", sa.Integer(), nullable=True),
        sa.Column("central_apneas", sa.Integer(), nullable=True),
        sa.Column("hypopneas", sa.Integer(), nullable=True),
        sa.Column("reras", sa.Integer(), nullable=True),
        sa.Column("ahi", sa.Float(), nullable=True),
        sa.Column("oai", sa.Float(), nullable=True),
        sa.Column("cai", sa.Float(), nullable=True),
        sa.Column("hi", sa.Float(), nullable=True),
        sa.Column("pressure_min", sa.Float(), nullable=True),
        sa.Column("pressure_max", sa.Float(), nullable=True),
        sa.Column("pressure_median", sa.Float(), nullable=True),
        sa.Column("pressure_mean", sa.Float(), nullable=True),
        sa.Column("pressure_95th", sa.Float(), nullable=True),
        sa.Column("leak_min", sa.Float(), nullable=True),
        sa.Column("leak_max", sa.Float(), nullable=True),
        sa.Column("leak_median", sa.Float(), nullable=True),
        sa.Column("leak_mean", sa.Float(), nullable=True),
        sa.Column("leak_95th", sa.Float(), nullable=True),
        sa.Column("spo2_min", sa.Float(), nullable=True),
        sa.Column("spo2_max", sa.Float(), nullable=True),
        sa.Column("spo2_mean", sa.Float(), nullable=True),
        sa.Column("spo2_avg", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["profile_id"], ["profiles.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("profile_id", "date", name="uq_profile_date"),
    )

    # Add profile_id to devices table (using batch mode for SQLite)
    with op.batch_alter_table("devices", schema=None) as batch_op:
        batch_op.add_column(sa.Column("profile_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_devices_profile_id", "profiles", ["profile_id"], ["id"], ondelete="CASCADE"
        )

    # Add day_id to sessions table (using batch mode for SQLite)
    with op.batch_alter_table("sessions", schema=None) as batch_op:
        batch_op.add_column(sa.Column("day_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_sessions_day_id", "days", ["day_id"], ["id"], ondelete="CASCADE"
        )


def downgrade() -> None:
    """Downgrade schema."""
    # Remove day_id from sessions (using batch mode for SQLite)
    with op.batch_alter_table("sessions", schema=None) as batch_op:
        batch_op.drop_constraint("fk_sessions_day_id", type_="foreignkey")
        batch_op.drop_column("day_id")

    # Remove profile_id from devices (using batch mode for SQLite)
    with op.batch_alter_table("devices", schema=None) as batch_op:
        batch_op.drop_constraint("fk_devices_profile_id", type_="foreignkey")
        batch_op.drop_column("profile_id")

    # Drop tables in reverse order
    op.drop_table("days")
    op.drop_table("profiles")
