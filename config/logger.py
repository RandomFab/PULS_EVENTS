"""
Configuration du logger pour l'application PULS_EVENTS.

Fournit un logger configuré avec :
- Niveau INFO par défaut
- Format : timestamp - niveau - message
- Sortie console (stdout)

Usage:
    from config.logger import logger
    logger.info("Message d'information")
    logger.warning("Message d'avertissement")
    logger.error("Message d'erreur")
"""
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
