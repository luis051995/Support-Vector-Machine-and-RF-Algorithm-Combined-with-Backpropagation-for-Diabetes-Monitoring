from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.db import IntegrityError
from clinical.domain.entities import InvalidData
from .models import User
class DjangoPatientAccounts:
    def validate(self,username,password):
        u=User(username=username)
        try:
            User._meta.get_field('username').run_validators(username)
            validate_password(password,u)
        except ValidationError as e:raise InvalidData(' '.join(e.messages))
    def create_patient_account(self,username,password,patient_id):
        try:
            # El cliente nunca determina el rol, permisos ni expediente vinculado.
            u=User.objects.create_user(username=username,password=password,role='patient',patient_id=patient_id,is_staff=False,is_superuser=False)
        except IntegrityError:raise InvalidData('No se pudo crear la cuenta con esos datos. Prueba otro nombre de usuario.')
        return u.id
