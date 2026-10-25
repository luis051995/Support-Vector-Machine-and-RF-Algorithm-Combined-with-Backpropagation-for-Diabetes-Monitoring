from cryptography.fernet import Fernet
from django.conf import settings
from django.core.checks import Error,register,Tags
@register(Tags.security)
def check_encryption_keys(app_configs,**kwargs):
    try:
        for k in settings.DATA_ENCRYPTION_KEYS:Fernet(k.encode())
    except (ValueError,TypeError):return [Error('DATA_ENCRYPTION_KEYS contiene una clave Fernet inválida.',id='polisalud.E001')]
    return []
