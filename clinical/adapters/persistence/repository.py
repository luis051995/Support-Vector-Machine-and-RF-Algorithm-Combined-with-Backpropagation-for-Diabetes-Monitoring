from django.db import transaction
from django.utils import timezone
from clinical.domain.entities import Patient as Entity
from .models import Patient,ChartEntry,AuditEvent
class DjangoClinicalRepository:
    def atomic(self):return transaction.atomic()
    def _entity(self,p):return Entity(p.id,p.code,p.payload,p.created_at.isoformat(),p.consent_at.isoformat())
    def list_patients(self,patient_id=None):
        q=Patient.objects.all()
        if patient_id is not None:q=q.filter(id=patient_id)
        return [self._entity(p) for p in q]
    def get_patient(self,pid):
        p=Patient.objects.filter(pk=pid).first();return self._entity(p) if p else None
    def create_patient(self,data):return self._entity(Patient.objects.create(payload=data,consent_at=timezone.now()))
    def update_patient(self,pid,data):
        p=Patient.objects.select_for_update().get(pk=pid);p.payload=data;p.save(update_fields=['payload']);return self._entity(p)
    def _entry(self,r):return {'id':r.id,'patient_id':r.patient_id,'author_id':r.author_id,'created_at':r.created_at.isoformat(),**r.payload}
    def add_entry(self,patient_id,author_id,kind,data):return self._entry(ChartEntry.objects.create(patient_id=patient_id,author_id=author_id,kind=kind,payload=data))
    def entries(self,patient_id,kind):return [self._entry(r) for r in ChartEntry.objects.filter(patient_id=patient_id,kind=kind)]
    def audit(self,actor_id,action,entity,entity_id=None):AuditEvent.objects.create(actor_id=actor_id,action=action,entity=entity,entity_id=entity_id)
