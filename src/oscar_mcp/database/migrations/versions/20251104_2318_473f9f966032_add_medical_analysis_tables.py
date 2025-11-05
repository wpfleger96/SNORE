"""add_medical_analysis_tables

Revision ID: 473f9f966032
Revises: be586d3cb24e
Create Date: 2025-11-04 23:18:01.768444

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "473f9f966032"
down_revision: Union[str, Sequence[str], None] = "be586d3cb24e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add medical analysis infrastructure tables."""
    # Create standalone tables (no foreign key dependencies)
    op.create_table(
        "algorithm_configs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("algorithm_name", sa.String(length=100), nullable=False),
        sa.Column("version", sa.String(length=50), nullable=False),
        sa.Column("parameters_json", sa.Text(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.Column("performance_metrics_json", sa.Text(), nullable=True),
        sa.Column("last_updated", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("algorithm_name"),
    )

    op.create_table(
        "analysis_patterns",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("pattern_id", sa.String(length=100), nullable=False),
        sa.Column("pattern_name", sa.String(length=255), nullable=False),
        sa.Column("category", sa.String(length=50), nullable=False),
        sa.Column("visual_description", sa.Text(), nullable=True),
        sa.Column("clinical_significance", sa.Text(), nullable=True),
        sa.Column("algorithm_rules_json", sa.Text(), nullable=True),
        sa.Column("threshold_values_json", sa.Text(), nullable=True),
        sa.Column("example_metrics_json", sa.Text(), nullable=True),
        sa.Column("severity_level", sa.String(length=20), nullable=True),
        sa.Column("confidence_threshold", sa.Float(), nullable=True),
        sa.Column("reference_image_path", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("pattern_id"),
    )

    op.create_table(
        "knowledge_base",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("knowledge_id", sa.String(length=100), nullable=False),
        sa.Column("category", sa.String(length=50), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("source", sa.String(length=255), nullable=True),
        sa.Column("version", sa.String(length=50), nullable=True),
        sa.Column("content_type", sa.String(length=50), nullable=True),
        sa.Column("priority", sa.Integer(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("knowledge_id"),
    )

    # Create tables dependent on knowledge_base
    op.create_table(
        "knowledge_content",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("knowledge_id", sa.String(length=100), nullable=False),
        sa.Column("content_key", sa.String(length=100), nullable=False),
        sa.Column("content_value", sa.Text(), nullable=False),
        sa.Column("data_type", sa.String(length=20), nullable=False),
        sa.Column("display_order", sa.Integer(), nullable=True),
        sa.Column("is_required", sa.Boolean(), nullable=True),
        sa.ForeignKeyConstraint(
            ["knowledge_id"], ["knowledge_base.knowledge_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "knowledge_media",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("knowledge_id", sa.String(length=100), nullable=False),
        sa.Column("media_type", sa.String(length=50), nullable=False),
        sa.Column("file_path", sa.String(length=500), nullable=False),
        sa.Column("caption", sa.Text(), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.Column("file_hash", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["knowledge_id"], ["knowledge_base.knowledge_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create tables dependent on analysis_patterns
    op.create_table(
        "pattern_relationships",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("from_pattern_id", sa.String(length=100), nullable=False),
        sa.Column("to_pattern_id", sa.String(length=100), nullable=False),
        sa.Column("relationship_type", sa.String(length=50), nullable=False),
        sa.Column("strength", sa.Float(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["from_pattern_id"], ["analysis_patterns.pattern_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["to_pattern_id"], ["analysis_patterns.pattern_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create tables dependent on existing sessions table
    op.create_table(
        "analysis_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("timestamp_start", sa.DateTime(), nullable=False),
        sa.Column("timestamp_end", sa.DateTime(), nullable=False),
        sa.Column("programmatic_result_json", sa.Text(), nullable=True),
        sa.Column("llm_result_json", sa.Text(), nullable=True),
        sa.Column("combined_result_json", sa.Text(), nullable=True),
        sa.Column("agreement_score", sa.Float(), nullable=True),
        sa.Column("processing_time_ms", sa.Integer(), nullable=True),
        sa.Column("engine_versions_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create tables dependent on analysis_results
    op.create_table(
        "analysis_feedback",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("analysis_result_id", sa.Integer(), nullable=False),
        sa.Column("feedback_type", sa.String(length=50), nullable=False),
        sa.Column("discrepancy_description", sa.Text(), nullable=True),
        sa.Column("suggested_improvement", sa.Text(), nullable=True),
        sa.Column("implemented", sa.Boolean(), nullable=True),
        sa.Column("reviewed_by", sa.String(length=100), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["analysis_result_id"], ["analysis_results.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "detected_patterns",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("analysis_result_id", sa.Integer(), nullable=False),
        sa.Column("pattern_id", sa.String(length=100), nullable=False),
        sa.Column("start_time", sa.DateTime(), nullable=False),
        sa.Column("duration", sa.Float(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("detected_by", sa.String(length=20), nullable=False),
        sa.Column("metrics_json", sa.Text(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["analysis_result_id"], ["analysis_results.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["pattern_id"],
            ["analysis_patterns.pattern_id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Add indexes for performance
    op.create_index(
        "ix_analysis_results_session", "analysis_results", ["session_id", "timestamp_start"]
    )
    op.create_index("ix_detected_patterns_pattern", "detected_patterns", ["pattern_id"])
    op.create_index("ix_detected_patterns_start_time", "detected_patterns", ["start_time"])
    op.create_index("ix_knowledge_base_category", "knowledge_base", ["category", "is_active"])
    op.create_index("ix_knowledge_content_knowledge", "knowledge_content", ["knowledge_id"])
    op.create_index("ix_knowledge_media_knowledge", "knowledge_media", ["knowledge_id"])
    op.create_index("ix_analysis_patterns_category", "analysis_patterns", ["category"])
    op.create_index(
        "ix_algorithm_configs_active", "algorithm_configs", ["algorithm_name", "is_active"]
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    """Remove medical analysis infrastructure tables."""
    # Drop indexes first
    op.drop_index("ix_algorithm_configs_active", table_name="algorithm_configs")
    op.drop_index("ix_analysis_patterns_category", table_name="analysis_patterns")
    op.drop_index("ix_knowledge_media_knowledge", table_name="knowledge_media")
    op.drop_index("ix_knowledge_content_knowledge", table_name="knowledge_content")
    op.drop_index("ix_knowledge_base_category", table_name="knowledge_base")
    op.drop_index("ix_detected_patterns_start_time", table_name="detected_patterns")
    op.drop_index("ix_detected_patterns_pattern", table_name="detected_patterns")
    op.drop_index("ix_analysis_results_session", table_name="analysis_results")

    # Drop tables in reverse dependency order
    op.drop_table("detected_patterns")
    op.drop_table("analysis_feedback")
    op.drop_table("analysis_results")
    op.drop_table("pattern_relationships")
    op.drop_table("knowledge_media")
    op.drop_table("knowledge_content")
    op.drop_table("knowledge_base")
    op.drop_table("analysis_patterns")
    op.drop_table("algorithm_configs")
    # ### end Alembic commands ###
