from sqlalchemy import event
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Database:
    """Holds the engine and session factory so tests can swap the database URL."""

    def __init__(self, url: str) -> None:
        self.engine = create_async_engine(url)
        if url.startswith("sqlite"):
            # SQLite ignores foreign keys (and ON DELETE CASCADE) unless asked.
            @event.listens_for(self.engine.sync_engine, "connect")
            def _enable_fk(dbapi_conn, _record) -> None:
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def create_tables(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def dispose(self) -> None:
        await self.engine.dispose()
