"""Entry point for SNORE server."""

import argparse
import logging

from importlib.metadata import version

from snore.constants import DEFAULT_DATABASE_PATH
from snore.database.session import init_database
from snore.server import server

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("snore")


def main() -> int:
    """Main entry point for SNORE server."""
    parser = argparse.ArgumentParser(
        description="SNORE: MCP server for OSCAR CPAP/APAP therapy data"
    )
    parser.add_argument(
        "--database",
        default=DEFAULT_DATABASE_PATH,
        help=f"Path to database file (default: {DEFAULT_DATABASE_PATH})",
    )
    args = parser.parse_args()

    logger.info(f"Starting SNORE v{version('snore')}...")
    logger.info(f"Using database: {args.database}")

    try:
        # Initialize database
        init_database(args.database)
        logger.info("Database initialized successfully")

        # Run server
        logger.info("Starting MCP server...")
        server.run()
        return 0

    except Exception as e:
        logger.error(f"Server failed to start: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
