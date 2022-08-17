"""Docstring"""

def configurable(original_class):
    """Docstring"""
    orig_init = original_class.__init__

    def __init__(self, config: dict, *args, **kwargs):
        for var, (key, _) in original_class._config.items():
            setattr(self, var, config[key])
        orig_init(self, *args, **kwargs)

    @classmethod
    def generate_config(cls):
        return {key: val for _, (key, val) in original_class._config.items()}

    setattr(original_class, '__init__', __init__)
    setattr(original_class, 'generate_config', generate_config)
    return original_class


def generate_default_config(classes: list) -> dict:
    """Docstring"""
    return {
        c.__name__: c.generate_config() for c in classes if hasattr(c, 'generate_config')}
