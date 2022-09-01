"""Docstring"""


def generate_default_config(classes: list) -> dict:
    """Docstring"""
    return {
        c.__name__: {
            key: val for _, (key, val) in c._config_map.items()
        } for c in classes if hasattr(c, '_config_map')
    }
