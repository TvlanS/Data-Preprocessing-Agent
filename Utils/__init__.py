import re


def sanitise_topic(topic: str) -> str:
    """Convert a topic string to a safe directory slug."""
    slug = topic.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s-]+", "_", slug)
    return slug.strip("_")
