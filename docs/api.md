# API Documentation

## Overview

The DSPy Meme Generation API provides endpoints for generating, managing, and retrieving memes. The API follows RESTful principles and uses JSON for request and response payloads.

## Base URL

```
https://your-domain.com/api/v1
```

## Authentication

API requests require an API key to be included in the `Authorization` header:

```
Authorization: Bearer your-api-key
```

## Rate Limiting

- Free tier: 100 requests per hour
- Pro tier: 1000 requests per hour
- Enterprise tier: Custom limits

## Endpoints

### Generate Meme

Generate a new meme based on provided parameters.

```http
POST /generate
```

#### Request Body

```json
{
    "topic": "python programming",
    "style": "minimalist",
    "constraints": {
        "aspect_ratio": "1:1",
        "max_text_length": 50,
        "style_requirements": ["clean", "modern"]
    },
    "verification_needs": {
        "factuality": true,
        "instructions": true,
        "appropriateness": true
    },
    "use_trends": true,
    "quality_threshold": 0.8
}
```

#### Response

```json
{
    "status": "success",
    "meme": {
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "template": {
            "id": 1,
            "name": "drake_format",
            "description": "Drake approval format"
        },
        "image_url": "https://cloudinary.com/image.jpg",
        "caption": "When you use Python instead of Java",
        "final_score": 0.85
    },
    "verification_results": {
        "is_factual": true,
        "constraints_met": true,
        "is_appropriate": true
    },
    "trend_analysis": {
        "relevance_score": 0.75,
        "trending_topics": ["Python 3.12", "AI Development"]
    },
    "alternatives": [
        {
            "image_url": "https://cloudinary.com/alt1.jpg",
            "caption": "Alternative caption 1",
            "final_score": 0.82
        }
    ]
}
```

### Get Trending Memes

Retrieve currently trending memes.

```http
GET /trending
```

#### Query Parameters

- `limit` (optional): Number of memes to return (default: 10)
- `offset` (optional): Pagination offset (default: 0)
- `timeframe` (optional): Time window for trends ("hour", "day", "week", default: "day")

#### Response

```json
{
    "memes": [
        {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "image_url": "https://cloudinary.com/image.jpg",
            "caption": "Trending meme caption",
            "popularity_score": 0.95,
            "created_at": "2024-03-15T14:30:00Z"
        }
    ],
    "total": 100,
    "next_offset": 10
}
```

### Search Memes

Search for memes based on criteria.

```http
GET /search
```

#### Query Parameters

- `q` (required): Search query
- `fields` (optional): Fields to search ("caption", "topic", "tags", default: all)
- `limit` (optional): Number of results (default: 10)
- `offset` (optional): Pagination offset (default: 0)

#### Response

```json
{
    "memes": [
        {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "image_url": "https://cloudinary.com/image.jpg",
            "caption": "Matching meme caption",
            "relevance_score": 0.85
        }
    ],
    "total": 50,
    "next_offset": 10
}
```

### List Templates

Get available meme templates.

```http
GET /templates
```

#### Query Parameters

- `limit` (optional): Number of templates (default: 10)
- `offset` (optional): Pagination offset (default: 0)
- `format_type` (optional): Filter by format type
- `sort_by` (optional): Sort field ("popularity", "name", default: "popularity")

#### Response

```json
{
    "templates": [
        {
            "id": 1,
            "name": "drake_format",
            "description": "Drake approval format",
            "format_type": "image",
            "structure": {
                "text": ["top", "bottom"]
            },
            "popularity_score": 0.95,
            "example_url": "https://cloudinary.com/example.jpg"
        }
    ],
    "total": 20,
    "next_offset": 10
}
```

## Error Responses

The API uses standard HTTP status codes and returns errors in the following format:

```json
{
    "code": "ERROR_CODE",
    "message": "Detailed error message",
    "type": "ErrorClassName",
    "details": {
        "additional": "error information"
    }
}
```

### Common Error Codes

- `1000-1999`: Database errors
- `2000-2999`: Cache errors
- `3000-3999`: Agent errors
- `4000-4999`: Content errors
- `5000-5999`: External service errors
- `9000-9999`: Configuration errors

### HTTP Status Codes

- `400`: Bad Request - Invalid parameters
- `401`: Unauthorized - Missing or invalid API key
- `403`: Forbidden - Insufficient permissions
- `404`: Not Found - Resource doesn't exist
- `429`: Too Many Requests - Rate limit exceeded
- `500`: Internal Server Error - Server-side error

## Pagination

List endpoints support pagination using `limit` and `offset` parameters:

- `limit`: Number of items per page (max: 100)
- `offset`: Starting position
- Response includes `total` count and `next_offset` for subsequent requests

## Caching

The API implements caching for improved performance:

- Generated memes are cached for 5 minutes
- Templates are cached for 1 hour
- Trend data is cached for 15 minutes

## WebSocket Support

Real-time updates are available through WebSocket connections:

```
ws://your-domain.com/api/v1/ws
```

### Events

- `meme.generated`: New meme generated
- `meme.trending`: Meme becomes trending
- `template.added`: New template available

## SDK Support

Official SDKs are available for:

- Python: `pip install dspy-meme-gen-client`
- JavaScript: `npm install dspy-meme-gen-js`
- Go: `go get github.com/yourusername/dspy-meme-gen-go`

## Rate Limiting Headers

Response headers include rate limit information:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1615910400
``` 