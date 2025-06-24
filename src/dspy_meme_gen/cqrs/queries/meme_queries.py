"""Query handlers for meme read models."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy import text, select
import redis.asyncio as redis


@dataclass
class MemeDto:
    """Data transfer object for meme data."""

    meme_id: str
    topic: str
    format: str
    text: str
    image_url: str
    status: str
    score: Optional[float]
    view_count: int
    share_count: int
    created_at: datetime
    engagement_score: float


@dataclass
class MemeListQuery:
    """Query for listing memes."""

    status: Optional[str] = None
    topic: Optional[str] = None
    format: Optional[str] = None
    min_score: Optional[float] = None
    sort_by: str = "created_at"  # created_at, score, engagement_score
    sort_order: str = "desc"
    limit: int = 20
    offset: int = 0


@dataclass
class TrendingQuery:
    """Query for trending memes."""

    time_window: str = "day"  # hour, day, week
    topic: Optional[str] = None
    limit: int = 10


@dataclass
class AnalyticsQuery:
    """Query for analytics data."""

    metric: str  # generated, approved, rejected, views, shares
    granularity: str = "day"  # hour, day
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


class MemeQueryService:
    """Service for querying meme read models."""

    def __init__(self, connection_string: str, redis_url: Optional[str] = None):
        self.engine = create_async_engine(connection_string)
        self.redis_url = redis_url
        self._redis: Optional[redis.Redis] = None

    async def initialize(self):
        """Initialize connections."""
        if self.redis_url:
            self._redis = await redis.from_url(self.redis_url)

    async def get_meme(self, meme_id: UUID) -> Optional[MemeDto]:
        """Get a single meme by ID."""
        async with AsyncSession(self.engine) as session:
            result = await session.execute(
                text(
                    """
                    SELECT meme_id, topic, format, text, image_url,
                           status, score, view_count, share_count,
                           created_at, engagement_score
                    FROM meme_read_model
                    WHERE meme_id = :meme_id
                    AND is_deleted = false
                """
                ),
                {"meme_id": str(meme_id)},
            )

            row = result.first()
            if row:
                return MemeDto(
                    meme_id=row.meme_id,
                    topic=row.topic,
                    format=row.format,
                    text=row.text,
                    image_url=row.image_url,
                    status=row.status,
                    score=row.score,
                    view_count=row.view_count,
                    share_count=row.share_count,
                    created_at=row.created_at,
                    engagement_score=row.engagement_score,
                )
            return None

    async def list_memes(self, query: MemeListQuery) -> List[MemeDto]:
        """List memes based on query criteria."""
        async with AsyncSession(self.engine) as session:
            # Build query
            sql = """
                SELECT meme_id, topic, format, text, image_url,
                       status, score, view_count, share_count,
                       created_at, engagement_score
                FROM meme_read_model
                WHERE is_deleted = false
            """

            params = {}

            # Add filters
            if query.status:
                sql += " AND status = :status"
                params["status"] = query.status

            if query.topic:
                sql += " AND topic ILIKE :topic"
                params["topic"] = f"%{query.topic}%"

            if query.format:
                sql += " AND format = :format"
                params["format"] = query.format

            if query.min_score is not None:
                sql += " AND score >= :min_score"
                params["min_score"] = query.min_score

            # Add sorting
            sort_column = {
                "created_at": "created_at",
                "score": "score",
                "engagement_score": "engagement_score",
            }.get(query.sort_by, "created_at")

            sort_order = "DESC" if query.sort_order.lower() == "desc" else "ASC"
            sql += f" ORDER BY {sort_column} {sort_order}"

            # Add pagination
            sql += " LIMIT :limit OFFSET :offset"
            params["limit"] = query.limit
            params["offset"] = query.offset

            result = await session.execute(text(sql), params)

            memes = []
            for row in result:
                memes.append(
                    MemeDto(
                        meme_id=row.meme_id,
                        topic=row.topic,
                        format=row.format,
                        text=row.text,
                        image_url=row.image_url,
                        status=row.status,
                        score=row.score,
                        view_count=row.view_count,
                        share_count=row.share_count,
                        created_at=row.created_at,
                        engagement_score=row.engagement_score,
                    )
                )

            return memes

    async def get_trending_memes(self, query: TrendingQuery) -> List[Dict[str, Any]]:
        """Get trending memes."""
        async with AsyncSession(self.engine) as session:
            sql = """
                SELECT t.meme_id, t.trend_score, t.view_count, t.share_count,
                       m.topic, m.format, m.text, m.image_url, m.created_at
                FROM trending_memes t
                JOIN meme_read_model m ON t.meme_id = m.meme_id
                WHERE t.time_window = :time_window
                AND m.is_deleted = false
            """

            params = {"time_window": query.time_window}

            if query.topic:
                sql += " AND m.topic ILIKE :topic"
                params["topic"] = f"%{query.topic}%"

            sql += " ORDER BY t.trend_score DESC LIMIT :limit"
            params["limit"] = query.limit

            result = await session.execute(text(sql), params)

            trending = []
            for row in result:
                trending.append(
                    {
                        "meme_id": row.meme_id,
                        "trend_score": row.trend_score,
                        "view_count": row.view_count,
                        "share_count": row.share_count,
                        "topic": row.topic,
                        "format": row.format,
                        "text": row.text,
                        "image_url": row.image_url,
                        "created_at": row.created_at.isoformat(),
                    }
                )

            return trending

    async def get_analytics(self, query: AnalyticsQuery) -> Dict[str, Any]:
        """Get analytics data."""
        if not self._redis:
            return {"error": "Analytics not available"}

        # Set date range
        end_date = query.end_date or datetime.utcnow()
        start_date = query.start_date or (end_date - timedelta(days=7))

        # Collect data points
        data_points = []
        current = start_date

        while current <= end_date:
            if query.granularity == "hour":
                key = f"stats:{current.strftime('%Y-%m-%d:%H')}"
                current += timedelta(hours=1)
            else:
                key = f"stats:{current.strftime('%Y-%m-%d')}"
                current += timedelta(days=1)

            value = await self._redis.hget(key, query.metric)
            data_points.append(
                {"timestamp": current.isoformat(), "value": int(value) if value else 0}
            )

        # Calculate summary statistics
        values = [p["value"] for p in data_points]
        total = sum(values)
        avg = total / len(values) if values else 0

        return {
            "metric": query.metric,
            "granularity": query.granularity,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "data_points": data_points,
            "summary": {
                "total": total,
                "average": avg,
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
            },
        }

    async def search_memes(self, search_term: str, limit: int = 20) -> List[MemeDto]:
        """Full-text search for memes."""
        async with AsyncSession(self.engine) as session:
            result = await session.execute(
                text(
                    """
                    SELECT meme_id, topic, format, text, image_url,
                           status, score, view_count, share_count,
                           created_at, engagement_score
                    FROM meme_read_model
                    WHERE is_deleted = false
                    AND status = 'approved'
                    AND (
                        topic ILIKE :search_term
                        OR text ILIKE :search_term
                    )
                    ORDER BY score DESC, engagement_score DESC
                    LIMIT :limit
                """
                ),
                {"search_term": f"%{search_term}%", "limit": limit},
            )

            memes = []
            for row in result:
                memes.append(
                    MemeDto(
                        meme_id=row.meme_id,
                        topic=row.topic,
                        format=row.format,
                        text=row.text,
                        image_url=row.image_url,
                        status=row.status,
                        score=row.score,
                        view_count=row.view_count,
                        share_count=row.share_count,
                        created_at=row.created_at,
                        engagement_score=row.engagement_score,
                    )
                )

            return memes

    async def get_popular_formats(self) -> List[Dict[str, Any]]:
        """Get popular meme formats."""
        async with AsyncSession(self.engine) as session:
            result = await session.execute(
                text(
                    """
                    SELECT format, 
                           COUNT(*) as meme_count,
                           AVG(score) as avg_score,
                           SUM(view_count) as total_views,
                           SUM(share_count) as total_shares
                    FROM meme_read_model
                    WHERE is_deleted = false
                    AND status = 'approved'
                    AND created_at > NOW() - INTERVAL '30 days'
                    GROUP BY format
                    ORDER BY meme_count DESC
                    LIMIT 10
                """
                )
            )

            formats = []
            for row in result:
                formats.append(
                    {
                        "format": row.format,
                        "meme_count": row.meme_count,
                        "avg_score": float(row.avg_score) if row.avg_score else 0,
                        "total_views": row.total_views,
                        "total_shares": row.total_shares,
                    }
                )

            return formats
