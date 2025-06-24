"""Initial meme models.

Revision ID: initial_meme_models
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "initial_meme_models"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create meme_templates table
    op.create_table(
        "meme_templates",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("example_url", sa.String(length=255), nullable=True),
        sa.Column("format_type", sa.String(length=50), nullable=False),
        sa.Column("structure", postgresql.JSONB(), nullable=True),
        sa.Column("popularity_score", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )

    # Create generated_memes table
    op.create_table(
        "generated_memes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("template_id", sa.Integer(), nullable=True),
        sa.Column("topic", sa.String(length=255), nullable=True),
        sa.Column("caption", sa.Text(), nullable=True),
        sa.Column("image_prompt", sa.Text(), nullable=True),
        sa.Column("image_url", sa.String(length=255), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("user_request", sa.Text(), nullable=True),
        sa.Column("feedback", postgresql.JSONB(), nullable=True),
        sa.ForeignKeyConstraint(["template_id"], ["meme_templates.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create trending_topics table
    op.create_table(
        "trending_topics",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("topic", sa.String(length=255), nullable=False),
        sa.Column("source", sa.String(length=50), nullable=True),
        sa.Column("relevance_score", sa.Float(), nullable=True),
        sa.Column("timestamp", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create meme_trend_associations table
    op.create_table(
        "meme_trend_associations",
        sa.Column("meme_id", sa.Integer(), nullable=False),
        sa.Column("trend_id", sa.Integer(), nullable=False),
        sa.Column("association_strength", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["meme_id"], ["generated_memes.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["trend_id"], ["trending_topics.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("meme_id", "trend_id"),
    )

    # Create user_feedback table
    op.create_table(
        "user_feedback",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("meme_id", sa.Integer(), nullable=False),
        sa.Column("rating", sa.Integer(), nullable=True),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("feedback_type", sa.String(length=50), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.CheckConstraint("rating >= 1 AND rating <= 5", name="rating_range"),
        sa.ForeignKeyConstraint(["meme_id"], ["generated_memes.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes
    op.create_index(
        op.f("ix_meme_templates_format_type"), "meme_templates", ["format_type"], unique=False
    )
    op.create_index(
        op.f("ix_generated_memes_template_id"), "generated_memes", ["template_id"], unique=False
    )
    op.create_index(op.f("ix_trending_topics_topic"), "trending_topics", ["topic"], unique=False)
    op.create_index(
        op.f("ix_trending_topics_timestamp"), "trending_topics", ["timestamp"], unique=False
    )
    op.create_index(op.f("ix_user_feedback_meme_id"), "user_feedback", ["meme_id"], unique=False)


def downgrade() -> None:
    # Drop indexes
    op.drop_index(op.f("ix_user_feedback_meme_id"), table_name="user_feedback")
    op.drop_index(op.f("ix_trending_topics_timestamp"), table_name="trending_topics")
    op.drop_index(op.f("ix_trending_topics_topic"), table_name="trending_topics")
    op.drop_index(op.f("ix_generated_memes_template_id"), table_name="generated_memes")
    op.drop_index(op.f("ix_meme_templates_format_type"), table_name="meme_templates")

    # Drop tables
    op.drop_table("user_feedback")
    op.drop_table("meme_trend_associations")
    op.drop_table("trending_topics")
    op.drop_table("generated_memes")
    op.drop_table("meme_templates")
