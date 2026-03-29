import json
import logging
import os

logger = logging.getLogger("uvicorn.error")

def load_config(default_cfg: dict, config_path: str) -> dict:
    """Load config from config.json, falling back to defaults."""
    cfg = dict(default_cfg)
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                user_cfg = json.load(f)
            cfg.update(user_cfg)
            logger.info(f"⚙️ Configuration chargée depuis {config_path}")
        except Exception as e:
            logger.warning(f"⚠️ Erreur lecture config, valeurs par défaut utilisées : {e}")
    else:
        logger.info(f"⚙️ Aucun fichier de config json trouvé à {config_path}, valeurs par défaut utilisées")
    return cfg
