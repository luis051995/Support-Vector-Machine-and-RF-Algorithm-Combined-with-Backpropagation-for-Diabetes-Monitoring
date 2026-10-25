from django.contrib.auth.models import AbstractUser
from django.db import models
class User(AbstractUser):
    role=models.CharField(max_length=12,choices=[('admin','Administrador'),('clinician','Profesional'),('patient','Paciente')],default='patient')
    patient=models.ForeignKey('records.Patient',null=True,blank=True,on_delete=models.PROTECT)
    class Meta:
        constraints=[models.CheckConstraint(condition=(models.Q(is_superuser=True)|~models.Q(role='patient')|models.Q(patient__isnull=False)),name='patient_account_requires_chart'),models.CheckConstraint(condition=models.Q(role__in=['admin','clinician','patient']),name='valid_user_role')]
class MobileSession(models.Model):
    token_hash=models.CharField(max_length=64,primary_key=True)
    user=models.ForeignKey(User,on_delete=models.CASCADE)
    expires_at=models.DateTimeField()
    created_at=models.DateTimeField(auto_now_add=True)
class RegistrationAttempt(models.Model):
    key=models.CharField(max_length=64,primary_key=True)
    count=models.PositiveIntegerField(default=0)
    started_at=models.DateTimeField()
