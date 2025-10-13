"""Flask application factory for the ML IDS web interface."""


def create_app(*args, **kwargs):
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)

__all__ = ["create_app"]
