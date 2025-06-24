"""Trend Scanning Agent for identifying and analyzing trending topics."""

from typing import Dict, Any, TypedDict, Optional, List
import os
from datetime import datetime, timedelta
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import dspy
from loguru import logger


class TrendSource(TypedDict):
    """Type definition for trend sources."""

    name: str
    url: str
    api_key: Optional[str]
    category: str
    weight: float


class TrendInfo(TypedDict):
    """Type definition for trend information."""

    topic: str
    source: str
    relevance_score: float
    timestamp: str
    metadata: Dict[str, Any]


class TrendResult(TypedDict):
    """Type definition for trend scanning results."""

    trends: List[TrendInfo]
    top_trend: str
    relevance_scores: Dict[str, float]
    meme_potential: Dict[str, float]


class TrendScanningAgent(dspy.Module):
    """
    Agent responsible for identifying and analyzing trending topics.

    This agent:
    - Connects to various trend sources (Twitter, Reddit, Google Trends)
    - Extracts trending topics and themes
    - Analyzes trend relevance for meme creation
    - Provides structured trend data for other agents
    """

    def __init__(self) -> None:
        """Initialize the Trend Scanning Agent with necessary predictors and APIs."""
        super().__init__()

        # Trend analysis predictor
        self.trend_analyzer = dspy.ChainOfThought(
            """Given a set of trending topics and an optional meme topic, analyze:
            1. The meme potential of each trend
            2. Relevance to the given topic (if provided)
            3. Current cultural context and impact
            4. Potential humor angles and formats
            
            Provide a detailed analysis with scores and reasoning.
            """
        )

        # Initialize trend sources
        self.sources = self._initialize_sources()

        # Cache for trend data
        self.trend_cache: Dict[str, Any] = {}
        self.cache_duration = timedelta(minutes=15)

    def forward(
        self, topic: Optional[str] = None, sources: Optional[List[str]] = None, limit: int = 5
    ) -> TrendResult:
        """
        Scan for trending topics and analyze their meme potential.

        Args:
            topic: Optional topic to analyze trends against
            sources: Optional list of specific sources to scan
            limit: Maximum number of trends to return

        Returns:
            TrendResult containing analyzed trends and scores
        """
        try:
            logger.info(f"Scanning trends{f' for topic: {topic}' if topic else ''}")

            # Get trends from all or specified sources
            raw_trends = self._fetch_trends(sources)

            # Analyze trends with DSPy
            analysis = self.trend_analyzer(trends=raw_trends, topic=topic if topic else "general")

            # Structure the results
            trends_list = []
            for trend in analysis.trends[:limit]:
                trend_info: TrendInfo = {
                    "topic": trend.topic,
                    "source": trend.source,
                    "relevance_score": float(trend.relevance_score),
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {
                        "category": trend.category if hasattr(trend, "category") else "general",
                        "sentiment": trend.sentiment if hasattr(trend, "sentiment") else "neutral",
                        "volume": trend.volume if hasattr(trend, "volume") else 1.0,
                    },
                }
                trends_list.append(trend_info)

            result: TrendResult = {
                "trends": trends_list,
                "top_trend": analysis.top_trend,
                "relevance_scores": {
                    trend.topic: float(trend.relevance_score) for trend in analysis.trends[:limit]
                },
                "meme_potential": {
                    trend.topic: float(trend.meme_potential) for trend in analysis.trends[:limit]
                },
            }

            logger.debug(f"Trend analysis complete: {result}")
            return result

        except Exception as e:
            logger.error(f"Error in trend scanning: {str(e)}")
            raise RuntimeError(f"Failed to scan trends: {str(e)}")

    def _initialize_sources(self) -> Dict[str, TrendSource]:
        """
        Initialize the available trend sources.

        Returns:
            Dictionary of trend sources and their configurations
        """
        return {
            "twitter": {
                "name": "Twitter",
                "url": "https://api.twitter.com/2/trends/place/1",
                "api_key": os.getenv("TWITTER_API_KEY"),
                "category": "social",
                "weight": 0.4,
            },
            "reddit": {
                "name": "Reddit",
                "url": "https://www.reddit.com/r/all/top.json",
                "api_key": os.getenv("REDDIT_API_KEY"),
                "category": "social",
                "weight": 0.3,
            },
            "google_trends": {
                "name": "Google Trends",
                "url": "https://trends.google.com/trends/api/dailytrends",
                "api_key": None,  # No API key needed
                "category": "search",
                "weight": 0.3,
            },
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _fetch_trends(self, sources: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Fetch trends from specified or all sources.

        Args:
            sources: Optional list of specific sources to fetch from

        Returns:
            List of trend data from all sources
        """
        all_trends = []
        source_list = sources if sources else self.sources.keys()

        for source_name in source_list:
            if source_name not in self.sources:
                logger.warning(f"Unknown source: {source_name}")
                continue

            source = self.sources[source_name]

            # Check cache first
            cache_key = f"{source_name}_trends"
            if cache_key in self.trend_cache:
                cache_time, cache_data = self.trend_cache[cache_key]
                if datetime.now() - cache_time < self.cache_duration:
                    all_trends.extend(cache_data)
                    continue

            try:
                # Fetch trends from source
                trends = self._fetch_from_source(source)

                # Update cache
                self.trend_cache[cache_key] = (datetime.now(), trends)

                all_trends.extend(trends)

            except Exception as e:
                logger.error(f"Error fetching from {source_name}: {str(e)}")
                # Try to use cached data if available
                if cache_key in self.trend_cache:
                    _, cache_data = self.trend_cache[cache_key]
                    all_trends.extend(cache_data)

        return all_trends

    def _fetch_from_source(self, source: TrendSource) -> List[Dict[str, Any]]:
        """
        Fetch trends from a specific source.

        Args:
            source: Source configuration to fetch from

        Returns:
            List of trends from the source
        """
        headers = {}
        if source["api_key"]:
            headers["Authorization"] = f"Bearer {source['api_key']}"

        response = requests.get(source["url"], headers=headers)
        response.raise_for_status()

        # Parse response based on source
        if source["name"] == "Twitter":
            return self._parse_twitter_trends(response.json())
        elif source["name"] == "Reddit":
            return self._parse_reddit_trends(response.json())
        elif source["name"] == "Google Trends":
            return self._parse_google_trends(response.json())
        else:
            return []

    def _parse_twitter_trends(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse Twitter API response."""
        trends = []
        for trend in data[0].get("trends", []):
            trends.append(
                {
                    "topic": trend["name"],
                    "source": "twitter",
                    "volume": trend.get("tweet_volume", 0),
                    "timestamp": datetime.now().isoformat(),
                }
            )
        return trends

    def _parse_reddit_trends(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse Reddit API response."""
        trends = []
        for post in data.get("data", {}).get("children", []):
            post_data = post["data"]
            trends.append(
                {
                    "topic": post_data["title"],
                    "source": "reddit",
                    "volume": post_data.get("score", 0),
                    "timestamp": datetime.now().isoformat(),
                }
            )
        return trends

    def _parse_google_trends(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse Google Trends API response."""
        trends = []
        for trend in data.get("default", {}).get("trendingSearchesDays", []):
            for search in trend.get("trendingSearches", []):
                trends.append(
                    {
                        "topic": search["title"]["query"],
                        "source": "google_trends",
                        "volume": search.get("formattedTraffic", "0+").rstrip("+"),
                        "timestamp": datetime.now().isoformat(),
                    }
                )
        return trends
