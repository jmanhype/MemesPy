import logging
import sys
import os

# Ensure the script can find the source directory
# This adjusts the path based on where the script is run from
# You might need to adjust this depending on your execution context
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from dspy_meme_gen.models.connection import db_manager
    from dspy_meme_gen.models.db_models.memes import Base
    from dspy_meme_gen.config.config import settings # To log which DB is used
except ImportError as e:
    logger.error(f"Failed to import necessary modules. Make sure PYTHONPATH is set or run from project root. Error: {e}")
    sys.exit(1)

def initialize_database():
    """Creates all database tables defined in SQLAlchemy models."""
    logger.info(f"Initializing database: {settings.database_url}")
    try:
        logger.info("Creating tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully (if they didn't exist).")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    initialize_database() 