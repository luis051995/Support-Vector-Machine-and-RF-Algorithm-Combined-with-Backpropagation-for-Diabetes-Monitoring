from django.db import transaction
from .models import SensorCalibration,SensorSession,SensorSample,Patient
from .repository import DjangoClinicalRepository
class DjangoLiveRepository:
    def atomic(self):return transaction.atomic()
    def patient(self,pid):return DjangoClinicalRepository().get_patient(pid)
    def lock_patient(self,pid):Patient.objects.select_for_update().get(pk=pid)
    def audit(self,*args):return DjangoClinicalRepository().audit(*args)
    def active(self,pid,now):return SensorSession.objects.filter(patient_id=pid,closed_at__isnull=True,expires_at__gt=now).exists()
    def dto(self,s,internal=True):
        d={'id':str(s.pk),'patient_id':s.patient_id,'meta':s.payload,'created_at':s.created_at.isoformat(),'expires_at':s.expires_at.isoformat(),'closed_at':s.closed_at.isoformat() if s.closed_at else None,'last_sequence':s.last_sequence,'sample_count':s.sample_count}
        if internal:d['token_hash']=s.token_hash
        return d
    def create(self,pid,uid,meta,hash,expires):return self.dto(SensorSession.objects.create(patient_id=pid,owner_id=uid,payload=meta,token_hash=hash,expires_at=expires))
    def get(self,sid,lock=False):
        q=SensorSession.objects
        if lock:q=q.select_for_update()
        s=q.filter(pk=sid).first();return self.dto(s) if s else None
    def calibration(self,id):
        c=SensorCalibration.objects.filter(pk=id).first()
        return {'id':c.id,'device_id':c.device_id,'reference':c.reference,'data':c.payload} if c else None
    def create_calibration(self,uid,device,reference,data):
        c=SensorCalibration.objects.create(author_id=uid,device_id=device,reference=reference,payload=data);return {'id':c.id}
    def samples(self,sid,after,limit):return [p.payload for p in SensorSample.objects.filter(session_id=sid,sequence__gt=after)[:limit]]
    def existing_sample(self,sid,seq):
        p=SensorSample.objects.filter(session_id=sid,sequence=seq).first();return p.payload if p else None
    def latest(self,sid):
        p=SensorSample.objects.filter(session_id=sid).order_by('-sequence').first();return p.payload if p else None
    def add_sample(self,s,data):
        SensorSample.objects.create(session_id=s['id'],sequence=data['seq'],payload=data)
        SensorSession.objects.filter(pk=s['id']).update(last_sequence=data['seq'],sample_count=s['sample_count']+1)
    def close(self,s,now):SensorSession.objects.filter(pk=s['id']).update(payload=s['meta'],closed_at=now,token_hash='')
    def add_reading(self,pid,uid,data):return DjangoClinicalRepository().add_entry(pid,uid,'reading',data)['id']
