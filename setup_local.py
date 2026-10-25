"""Crea .env solo una vez; jamás sobrescribe las claves existentes."""
import secrets
from pathlib import Path
from cryptography.fernet import Fernet
p=Path(__file__).resolve().parent/'.env'
if p.exists():print('.env ya existe. Se conservan las claves.')
else:
    # Modo exclusivo evita reemplazar claves por error o por ejecución concurrente.
    with p.open('x') as f:f.write('APP_ENV=development\nDJANGO_SECRET_KEY='+secrets.token_urlsafe(64)+'\nDATA_ENCRYPTION_KEYS='+Fernet.generate_key().decode()+'\nDJANGO_ALLOWED_HOSTS=localhost,127.0.0.1,testserver\n')
    p.chmod(0o600);print('Configuración local creada. Respalda .env de forma segura; no lo subas a Git.')
