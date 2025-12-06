"""
Register all available device parsers.

This module provides explicit parser registration, which is safer than
auto-registration at module import time. Call register_all_parsers()
at application startup to enable all available parsers.
"""

import logging

logger = logging.getLogger(__name__)


def register_all_parsers() -> None:
    """
    Register all available device parsers with the global registry.

    This function attempts to import and register each parser individually,
    with proper error handling. If a parser fails to import or register,
    a warning is logged but other parsers continue to load.

    This should be called once at application startup.
    """
    from snore.parsers.registry import parser_registry

    # ResMed EDF+ Parser
    try:
        from snore.parsers.resmed_edf import ResmedEDFParser

        parser_registry.register(ResmedEDFParser())
        logger.info("Registered ResMed EDF+ parser")
    except ImportError as e:
        logger.warning(f"ResMed EDF+ parser not available: {e}")
    except Exception as e:
        logger.error(f"Failed to register ResMed EDF+ parser: {e}", exc_info=True)

    # Future parsers go here...
    #
    # Philips Binary Parser
    # try:
    #     from snore.parsers.philips_binary import PhilipsBinaryParser
    #     parser_registry.register(PhilipsBinaryParser())
    #     logger.info("Registered Philips Binary parser")
    # except ImportError as e:
    #     logger.warning(f"Philips Binary parser not available: {e}")
    # except Exception as e:
    #     logger.error(f"Failed to register Philips Binary parser: {e}", exc_info=True)
    #
    # Fisher & Paykel Parser
    # try:
    #     from snore.parsers.fisher_paykel import FisherPaykelParser
    #     parser_registry.register(FisherPaykelParser())
    #     logger.info("Registered Fisher & Paykel parser")
    # except ImportError as e:
    #     logger.warning(f"Fisher & Paykel parser not available: {e}")
    # except Exception as e:
    #     logger.error(f"Failed to register Fisher & Paykel parser: {e}", exc_info=True)

    registered_count = len(parser_registry.list_parsers())
    logger.info(f"Parser registration complete: {registered_count} parser(s) available")
