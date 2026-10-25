import json
from cryptography.fernet import Fernet,MultiFernet,InvalidToken
from django.conf import settings
from django.db import models
from django.core.exceptions import ImproperlyConfigured
def cipher():
    try:return MultiFernet([Fernet(k.encode()) for k in settings.DATA_ENCRYPTION_KEYS])
    except (ValueError,TypeError):raise ImproperlyConfigured('DATA_ENCRYPTION_KEYS inválida.')
class EncryptedJSONField(models.TextField):
    description='JSON cifrado Fernet: AES-128-CBC + HMAC-SHA256'
    def from_db_value(self,value,expression,connection):
        if value is None:return None
        try:return json.loads(cipher().decrypt(value.encode()).decode())
        except (InvalidToken,ValueError,UnicodeError):raise ImproperlyConfigured('No se puede descifrar el registro; verifica las claves sin sobrescribir la base.')
    def get_prep_value(self,value):
        if value is None:return None
        return cipher().encrypt(json.dumps(value,ensure_ascii=False,allow_nan=False).encode()).decode()
