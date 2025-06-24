"""Initial migration for content guidelines tables."""

from typing import Any
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic
revision = "001_create_content_guidelines"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create content guidelines tables."""
    # Create enum type for severity levels
    op.execute(
        """
        CREATE TYPE severity_level AS ENUM ('low', 'medium', 'high');
    """
    )

    # Create guideline categories table
    op.create_table(
        "guideline_categories",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(50), nullable=False),
        sa.Column("description", sa.String(255)),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )

    # Create content guidelines table
    op.create_table(
        "content_guidelines",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("category_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.String(255), nullable=False),
        sa.Column(
            "severity",
            postgresql.ENUM("low", "medium", "high", name="severity_level"),
            nullable=False,
        ),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(["category_id"], ["guideline_categories.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes
    op.create_index("ix_content_guidelines_category_id", "content_guidelines", ["category_id"])
    op.create_index("ix_content_guidelines_name", "content_guidelines", ["name"])


def downgrade() -> None:
    """Remove content guidelines tables."""
    # Drop tables
    op.drop_table("content_guidelines")
    op.drop_table("guideline_categories")

    # Drop enum type
    op.execute(
        """
        DROP TYPE severity_level;
    """
    )
