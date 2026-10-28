"""Keep public source usable without the optional private protection package."""

from importlib.util import find_spec

from jasna._frozen import is_frozen


class ProtectionUnavailableError(RuntimeError):
    pass


def protection_available() -> bool:
    if is_frozen():
        return True
    spec = find_spec("jasna.protection")
    if spec is None or spec.origin is None:
        return False
    return find_spec("jasna.protection.protected_tensors") is not None


def require_protection() -> None:
    if not protection_available():
        raise ProtectionUnavailableError("Supporter models and activation require the official Jasna release.")
