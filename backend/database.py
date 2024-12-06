from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.pool import StaticPool
import os
import sys

# Add the parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Determine the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define database URL (SQLite in-memory or file-based)
DATABASE_URL = os.environ.get(
    'DATABASE_URL', 
    f'sqlite:///{os.path.join(BASE_DIR, "daicn.db")}'
)

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL, 
    connect_args={'check_same_thread': False}  # Only for SQLite
)

# Create a configured "Session" class
SessionLocal = sessionmaker(
    bind=engine, 
    autocommit=False, 
    autoflush=False
)

# Create a Base class for declarative models
Base = declarative_base()

def get_db() -> Session:
    """
    Dependency that creates a new database session for each request
    
    Yields:
        SQLAlchemy session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """
    Initialize the database by creating all tables defined in models
    """
    # Import models here to avoid circular imports
    from backend import models
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully.")
