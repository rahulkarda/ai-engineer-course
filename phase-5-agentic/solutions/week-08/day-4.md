# Solutions — Week 08 · Day 4

## Exercise 1 — delete_article tool

```python
# Add to knowledge_server.py

@mcp.tool()
async def delete_article(id: str, ctx: Context = None) -> str:
    """Delete an article from the knowledge base by its ID.

    Returns the deleted article's title on success, or an error message if not found.
    """
    if id not in ARTICLES:
        return f"Error: article '{id}' not found."

    title = ARTICLES[id]["title"]
    del ARTICLES[id]
    logger.info("Article deleted: %s", id)
    if ctx:
        await ctx.warning(f"Article deleted: {id} ({title})")
    return f"Deleted article '{id}': {title}"
```

## Exercise 2 — search resource + test

```python
# Add to knowledge_server.py

@mcp.resource("kb://search/{query}")
def search_resource(query: str) -> str:
    """Return articles matching query as formatted text."""
    query_lower = query.lower()
    results = [
        article for article in ARTICLES.values()
        if (
            query_lower in article["title"].lower()
            or query_lower in article["content"].lower()
            or any(query_lower in tag for tag in article["tags"])
        )
    ]
    if not results:
        return f"No articles found for: {query}"
    lines = [f"Search results for '{query}':"]
    for art in results:
        lines.append(f"  [{art['id']}] {art['title']} ({art['category']})")
    return "\n".join(lines)
```

```python
# tests/test_knowledge_server_resources.py
import pytest

# Note: call resource handlers directly — they are plain Python functions
# The @mcp.resource decorator still lets the function be called normally.
from knowledge_server import search_resource, ARTICLES


@pytest.fixture(autouse=True)
def reset_articles():
    """Ensure known test articles exist before each test."""
    original = dict(ARTICLES)
    ARTICLES.clear()
    ARTICLES.update({
        "mcp-intro": {
            "id": "mcp-intro",
            "title": "Introduction to MCP",
            "category": "protocols",
            "content": "MCP standardizes tool and data integration.",
            "tags": ["mcp", "ai"],
            "created_at": "2024-11-15",
        },
    })
    yield
    ARTICLES.clear()
    ARTICLES.update(original)


def test_search_resource_finds_match():
    result = search_resource("MCP")
    assert "mcp-intro" in result
    assert "Introduction to MCP" in result


def test_search_resource_no_match():
    result = search_resource("kubernetes")
    assert "No articles found" in result
    assert "kubernetes" in result
```

## Exercise 3 — summarize_category prompt

```python
# Add to knowledge_server.py

@mcp.prompt()
def summarize_category(category: str) -> str:
    """Generate a prompt to summarize the main themes in an article category.

    Args:
        category: The article category to summarize.
    """
    articles_in_category = [
        art for art in ARTICLES.values() if art["category"] == category
    ]
    if not articles_in_category:
        return f"No articles found in category '{category}'. Nothing to summarize."

    article_summaries = "\n\n".join(
        f"Title: {art['title']}\nContent: {art['content'][:200]}..."
        for art in articles_in_category
    )

    return (
        f"You are a technical writer. Summarize the main themes across the following "
        f"articles in the '{category}' category.\n\n"
        f"{article_summaries}\n\n"
        f"Provide a cohesive 2-3 paragraph summary covering the key ideas."
    )
```

## Exercise 4 — pytest tests for knowledge_server

```python
# tests/test_knowledge_server.py
import pytest
import asyncio
from knowledge_server import (
    search_articles,
    get_article_stats,
    add_article,
    ARTICLES,
)


@pytest.fixture(autouse=True)
def clean_articles():
    """Run each test with a known, clean state."""
    ARTICLES.clear()
    ARTICLES["test-article"] = {
        "id": "test-article",
        "title": "Test Article",
        "category": "testing",
        "content": "This is a test article about unit testing practices.",
        "tags": ["testing", "pytest"],
        "created_at": "2024-01-01",
    }
    yield
    ARTICLES.clear()


@pytest.mark.asyncio
async def test_search_articles_finds_by_keyword():
    results = await search_articles(query="testing")
    assert "test-article" in results
    assert "Test Article" in results


@pytest.mark.asyncio
async def test_search_articles_no_match():
    results = await search_articles(query="kubernetes")
    assert "No articles found" in results


@pytest.mark.asyncio
async def test_search_articles_respects_max_results():
    # Add a second article
    ARTICLES["article-2"] = {
        "id": "article-2",
        "title": "Another Test",
        "category": "testing",
        "content": "testing content",
        "tags": ["testing"],
        "created_at": "2024-01-02",
    }
    results = await search_articles(query="testing", max_results=1)
    # Should return only 1 result
    assert results.count("[") == 1  # one bracketed ID


@pytest.mark.asyncio
async def test_get_article_stats_counts_categories():
    stats = await get_article_stats()
    assert "testing: 1" in stats
    assert "Total articles: 1" in stats


@pytest.mark.asyncio
async def test_add_article_creates_new():
    result = await add_article(
        id="new-article",
        title="New Article",
        category="new-category",
        content="New content here.",
        tags=["new"],
    )
    assert "new-article" in result
    assert "new-article" in ARTICLES


@pytest.mark.asyncio
async def test_add_article_duplicate_id_returns_error():
    result = await add_article(
        id="test-article",  # already exists
        title="Duplicate",
        category="testing",
        content="dup content",
    )
    assert "already exists" in result.lower()


@pytest.mark.asyncio
async def test_add_article_empty_title_returns_error():
    result = await add_article(
        id="empty-title",
        title="   ",  # whitespace only
        category="testing",
        content="content",
    )
    assert "error" in result.lower() or "empty" in result.lower()
```
