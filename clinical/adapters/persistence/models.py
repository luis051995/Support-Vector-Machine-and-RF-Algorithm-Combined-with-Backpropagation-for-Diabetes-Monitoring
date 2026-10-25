import secrets
from django.conf import settings
from django.db import models
from .fields import EncryptedJSONField
def code():return 'IPN-'+secrets.token_hex(5).upper()
class Patient(models.Model):
    code=models.CharField(max_length=30,unique=True,default=code,editable=False)
    payload=EncryptedJSONField()
    created_at=models.DateTimeField(auto_now_add=True)
    consent_at=models.DateTimeField()
    class Meta:ordering=['id']
class ChartEntry(models.Model):
    patient=models.ForeignKey(Patient,on_delete=models.PROTECT,related_name='entries')
    author=models.ForeignKey(settings.AUTH_USER_MODEL,on_delete=models.PROTECT)
    kind=models.CharField(max_length=10,choices=[('visit','Consulta'),('reading','Medición')])
    payload=EncryptedJSONField()
    created_at=models.DateTimeField(auto_now_add=True)
    class Meta:
        ordering=['-id']
        indexes=[models.Index(fields=['patient','kind','-id'])]
        constraints=[models.CheckConstraint(condition=models.Q(kind__in=['visit','reading']),name='entry_valid_kind')]
class AuditEvent(models.Model):
    actor=models.ForeignKey(settings.AUTH_USER_MODEL,on_delete=models.PROTECT)
    action=models.CharField(max_length=30)
    entity=models.CharField(max_length=30)
    entity_id=models.BigIntegerField(null=True)
    created_at=models.DateTimeField(auto_now_add=True)
    class Meta:ordering=['-id']

class SensorCalibration(models.Model):
    device_id=models.CharField(max_length=64)
    reference=models.CharField(max_length=200)
    payload=EncryptedJSONField()
    author=models.ForeignKey(settings.AUTH_USER_MODEL,on_delete=models.PROTECT)
    created_at=models.DateTimeField(auto_now_add=True)
class SensorSession(models.Model):
    import uuid
    id=models.UUIDField(primary_key=True,default=uuid.uuid4,editable=False)
    patient=models.ForeignKey(Patient,on_delete=models.PROTECT)
    owner=models.ForeignKey(settings.AUTH_USER_MODEL,on_delete=models.PROTECT)
    payload=EncryptedJSONField()
    token_hash=models.CharField(max_length=64)
    expires_at=models.DateTimeField()
    closed_at=models.DateTimeField(null=True)
    created_at=models.DateTimeField(auto_now_add=True)
    last_sequence=models.BigIntegerField(default=-1)
    sample_count=models.PositiveIntegerField(default=0)
class SensorSample(models.Model):
    session=models.ForeignKey(SensorSession,on_delete=models.PROTECT,related_name='samples')
    sequence=models.PositiveBigIntegerField()
    payload=EncryptedJSONField()
    created_at=models.DateTimeField(auto_now_add=True)
    class Meta:
        ordering=['sequence']
        constraints=[models.UniqueConstraint(fields=['session','sequence'],name='unique_sensor_sequence')]
