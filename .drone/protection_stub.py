class ProtectionError(RuntimeError):
    pass


class _LicenseStore:
    def load_license(self):
        return None

    def is_licensed(self):
        return False

    def set_license(self, email, key):
        raise ProtectionError("The protection module is not available in this public build.")


class _ProtectedModel:
    def _missing(self, *args, **kwargs):
        raise ProtectionError("The protection module is not available in this public build.")

    decrypt_model_to_buffer = _missing
    decrypt_model_bytes = _missing
    decrypt_engine_bytes = _missing
    encrypt_engine_bytes = _missing


license_store = _LicenseStore()
protected_model = _ProtectedModel()
