import secrets,hashlib
from datetime import timedelta,date
from statistics import median
from clinical.domain.entities import InvalidData,Forbidden,NotFound,numeric,text,age_at
from clinical.domain.sensor import sample,calibration
from clinical.domain.ports import LiveRepository,RiskModel
class LiveService:
    """Pruebas acotadas por paciente. Puertos inyectados; sin Django ni hardware."""
    def __init__(self,repo:LiveRepository,model:RiskModel,now,today):self.repo=repo;self.model=model;self.now=now;self.today=today
    def _get(self,actor,sid,lock=False):
        s=self.repo.get(sid,lock)
        if not s:raise NotFound('Prueba no disponible.')
        actor.require_patient(s['patient_id']);return s
    def start(self,actor,pid,data):
        actor.require_patient(pid)
        if not self.repo.patient(pid):raise NotFound('Expediente no disponible.')
        if data.get('conditioned_sensor') is not True:raise InvalidData('Confirma que el sensor fue acondicionado según el fabricante.')
        device=text(data,'device_id',64,True)
        transport=data.get('transport')
        if transport not in ['wifi','ble']:raise InvalidData('Selecciona Wi-Fi o BLE.')
        profile=None
        if data.get('calibration_id'):
            if type(data['calibration_id']) is not int:raise InvalidData('Identificador de calibración inválido.')
            profile=self.repo.calibration(data['calibration_id'])
            if not profile or profile['device_id']!=device:raise InvalidData('La calibración no corresponde al dispositivo.')
        meta={'conditioned_sensor':True,'device_id':device,'transport':transport,'weight':numeric(data,'weight',30,300),'height':numeric(data,'height',100,230),'calibration':profile,'summary':None}
        secret=secrets.token_urlsafe(32)
        with self.repo.atomic():
            # Bloqueo del paciente evita dos sesiones simultáneas incluso con solicitudes concurrentes.
            self.repo.lock_patient(pid)
            if self.repo.active(pid,self.now()):raise InvalidData('Ya existe una prueba activa. Finalízala o espera su vencimiento (10 minutos).')
            s=self.repo.create(pid,actor.id,meta,hashlib.sha256(secret.encode()).hexdigest(),self.now()+timedelta(minutes=10))
            self.repo.audit(actor.id,'sensor_start','patient',pid)
        return {'session':s,'device_token':secret}
    def status(self,actor,sid,after=-1):
        s=self._get(actor,sid)
        return {'session':s,'samples':self.repo.samples(sid,after,300),'expired':s['expires_at']<=self.now().isoformat()}
    def ingest(self,sid,data,token=None,actor=None):
        with self.repo.atomic():
            s=self.repo.get(sid,True)
            if not s:raise NotFound('Prueba no disponible.')
            if actor:
                actor.require_patient(s['patient_id'])
                if s['meta']['transport']!='ble':raise Forbidden('Esta sesión recibe datos por Wi-Fi.')
            elif not token or not secrets.compare_digest(s['token_hash'],hashlib.sha256(token.encode()).hexdigest()):raise Forbidden('Credencial del dispositivo inválida.')
            elif s['meta']['transport']!='wifi':raise Forbidden('Esta sesión recibe datos por BLE.')
            if s['closed_at'] or s['expires_at']<=self.now().isoformat():raise Forbidden('La prueba terminó o venció.')
            clean=sample(data,s['meta']['calibration']['data'] if s['meta']['calibration'] else None)
            if clean['device_id']!=s['meta']['device_id']:raise Forbidden('Dispositivo incorrecto.')
            if clean['seq']<=s['last_sequence']:
                prior=self.repo.existing_sample(sid,clean['seq'])
                if prior and all(prior[k]==clean[k] for k in ['raw_adc','adc_mv','uptime_ms','device_id']):return {'ok':True,'duplicate':True}
                raise InvalidData('Secuencia repetida, fuera de orden o reinicio del dispositivo. Abre otra prueba si reiniciaste el ESP32.')
            if s['sample_count']>=300:raise InvalidData('Límite de 300 muestras alcanzado. Finaliza la prueba.')
            last=self.repo.latest(sid)
            if last and clean['uptime_ms']-last['uptime_ms']<800:raise InvalidData('Muestreo demasiado rápido o reloj reiniciado; máximo aproximado 1 Hz.')
            self.repo.add_sample(s,clean)
        return {'ok':True,'quality':clean['quality']}
    def finish(self,actor,sid,cancel=False):
        with self.repo.atomic():
            s=self._get(actor,sid,True)
            if s['closed_at']:return s['meta']['summary']
            points=self.repo.samples(sid,-1,300);valid=[p['estimated_ppm'] for p in points if p['quality']=='calibrated_experimental']
            summary={'status':'cancelled' if cancel else 'raw_only','sample_count':len(points),'valid_count':len(valid),'reading_id':None,'score':None,'message':'Se conserva señal cruda. Sin inferencia: faltan calibración o al menos cinco muestras válidas.'}
            if not cancel and len(valid)>=5:
                p=self.repo.patient(s['patient_id']);d=p.data
                features={'Edad':age_at(date.fromisoformat(d['birth_date']),self.today()),'Sexo':d['sex'],'Peso':s['meta']['weight'],'Altura':s['meta']['height'],'Antecedentes':d['family_history'],'Acetona':median(valid[-20:])}
                if not 18<=features['Edad']<=100:raise InvalidData('Paciente fuera del dominio del modelo.')
                result=self.model.infer(features)
                reading={'acetone':features['Acetona'],'weight':features['Peso'],'height':features['Altura'],'source':'sensor','device_id':s['meta']['device_id'],'calibration':s['meta']['calibration']['reference'],'features_json':features,'sensor_session':str(sid),'aggregation':'median_last_20_valid',**result}
                rid=self.repo.add_reading(s['patient_id'],actor.id,reading)
                summary.update(status='calibrated_experimental',reading_id=rid,score=result['score'],acetone=features['Acetona'],message='Resultado experimental; no confirma ni descarta diabetes.')
            s['meta']['summary']=summary;self.repo.close(s,self.now());self.repo.audit(actor.id,'sensor_cancel' if cancel else 'sensor_finish','patient',s['patient_id'])
            return summary
    def add_calibration(self,actor,data):
        actor.require_staff();clean=calibration(data)
        with self.repo.atomic():
            r=self.repo.create_calibration(actor.id,text(data,'device_id',64,True),text(data,'reference',200,True),clean)
            self.repo.audit(actor.id,'sensor_calibration','calibration',r['id'])
            return r
