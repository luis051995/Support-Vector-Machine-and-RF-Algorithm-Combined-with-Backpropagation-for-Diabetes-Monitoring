"""Casos de uso independientes de Django, HTTP y ORM."""
from datetime import date
from clinical.domain.entities import InvalidData,NotFound,numeric,text,validate_patient,age_at
from clinical.domain.ports import ClinicalRepository,RiskModel
class ClinicalService:
    def __init__(self,repository:ClinicalRepository,risk_model:RiskModel,today=date.today):self.repo=repository;self.model=risk_model;self.today=today
    def _patient(self,actor,pid):
        actor.require_patient(pid);p=self.repo.get_patient(pid)
        if p is None:raise NotFound('Expediente no disponible.')
        return p
    def list(self,actor):
        with self.repo.atomic():
            # Cuenta paciente sin vínculo nunca puede listar todos los registros.
            if not actor.staff and actor.patient_id is None:return []
            patients=self.repo.list_patients(None if actor.staff else actor.patient_id);self.repo.audit(actor.id,'list','patient')
            return [p.as_dict() for p in patients]
    def create(self,actor,data):
        actor.require_staff()
        if data.get('consent') is not True:raise InvalidData('Consentimiento obligatorio.')
        clean=validate_patient(data,self.today())
        with self.repo.atomic():
            p=self.repo.create_patient(clean);self.repo.audit(actor.id,'create','patient',p.id);return {'id':p.id}
    def update(self,actor,pid,data):
        actor.require_staff();clean=validate_patient(data,self.today())
        with self.repo.atomic():
            self._patient(actor,pid);p=self.repo.update_patient(pid,clean);self.repo.audit(actor.id,'update','patient',pid);return p.as_dict()
    def detail(self,actor,pid,export=False):
        with self.repo.atomic():
            p=self._patient(actor,pid);result={'patient':p.as_dict(),'visits':self.repo.entries(pid,'visit'),'readings':self.repo.entries(pid,'reading')};self.repo.audit(actor.id,'export' if export else 'read','patient',pid);return result
    def visit(self,actor,pid,data):
        actor.require_staff();clean={k:text(data,k,10000,k=='reason') for k in ('reason','notes','plan')}
        with self.repo.atomic():
            self._patient(actor,pid);entry=self.repo.add_entry(pid,actor.id,'visit',clean);self.repo.audit(actor.id,'create','visit',entry['id']);return entry
    def reading(self,actor,pid,data):
        actor.require_staff()
        clean={'acetone':numeric(data,'acetone',0,100),'weight':numeric(data,'weight',30,300),'height':numeric(data,'height',100,230),'source':data.get('source','manual'),'device_id':text(data,'device_id',120),'calibration':text(data,'calibration',300)}
        if clean['source'] not in ('manual','sensor'):raise InvalidData('Origen inválido.')
        if clean['source']=='sensor' and not(clean['device_id'] and clean['calibration']):raise InvalidData('Sensor requiere dispositivo y calibración.')
        with self.repo.atomic():
            p=self._patient(actor,pid);age=age_at(date.fromisoformat(p.data['birth_date']),self.today())
            if not 18<=age<=100:raise InvalidData('Paciente fuera del dominio del modelo.')
            features={'Edad':age,'Sexo':p.data['sex'],'Peso':clean['weight'],'Altura':clean['height'],'Antecedentes':p.data['family_history'],'Acetona':clean['acetone']}
            clean.update(self.model.infer(features));clean['features_json']=features
            entry=self.repo.add_entry(pid,actor.id,'reading',clean);self.repo.audit(actor.id,'create','reading',entry['id']);return entry
    def metrics(self,actor):actor.require_staff();return self.model.metrics()
