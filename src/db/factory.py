from functools import lru_cache

from src.config import get_settings
from src.db.interfaces.base import BaseDatabase
from src.db.interfaces.postgresql import PostgreSQLDatabase
from src.schemas.database.config import PostgreSQLSettings


@lru_cache(maxsize=1)
def make_database() -> BaseDatabase:
    """Factory function to create a database instance.

    Cached: every caller shares one connection pool. Without this, main.py's
    startup call and make_agentic_rag()'s call would each open their own pool
    (and each run startup()).

    :returns: An instance of the database
    :rtype: BaseDatabase
    """
    # Get settings from centralized config
    settings = get_settings()

    # Create PostgreSQL config from settings
    config = PostgreSQLSettings(
        database_url=settings.postgres_database_url,
        echo_sql=settings.postgres_echo_sql,
        pool_size=settings.postgres_pool_size,
        max_overflow=settings.postgres_max_overflow,
    )

    database = PostgreSQLDatabase(config=config)
    database.startup()
    return database
