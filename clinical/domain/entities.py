from dataclasses import dataclass
from datetime import date
from math import isfinite
class InvalidData(ValueError):pass
class NotFound(Exception):pass
class Forbidden(Exception):pass
@dataclass(frozen=True)
class Actor:
    id:int
    role:str
    patient_id:int|None=None
    @property
    def staff(self):return self.role=='admin'
    def require_staff(self):
        if not self.staff:raise Forbidden('Acceso exclusivo del administrador.')
    def require_patient(self,pid):
        if not self.staff and self.patient_id!=pid:raise NotFound('Expediente no disponible.')
@dataclass(frozen=True)
class Patient:
    id:int
    code:str
    data:dict
    created_at:str
    consent_at:str
    def as_dict(self):return {'id':self.id,'code':self.code,**self.data,'created_at':self.created_at,'consent_at':self.consent_at}
def text(data,key,max_length=10000,required=False):
    value=data.get(key,'')
    if not isinstance(value,str) or len(value)>max_length or (required and not value.strip()):raise InvalidData(f'Campo inválido: {key}')
    return value.strip()
def numeric(data,key,minimum,maximum):
    try:value=float(data[key])
    except (KeyError,TypeError,ValueError):raise InvalidData(f'Campo numérico requerido: {key}')
    if not isfinite(value) or not minimum<=value<=maximum:raise InvalidData(f'{key}: rango {minimum} a {maximum}.')
    return value
def age_at(born,today):return today.year-born.year-((today.month,today.day)<(born.month,born.day))
def validate_patient(data,today):
    try:born=date.fromisoformat(data['birth_date'])
    except (ValueError,KeyError,TypeError):raise InvalidData('Fecha de nacimiento inválida.')
    if not 18<=age_at(born,today)<=100:raise InvalidData('Solo adultos de 18 a 100 años para este modelo.')
    for k in ('sex','family_history'):
        if type(data.get(k)) is not int or data[k] not in (0,1):raise InvalidData(f'{k}: 0 o 1.')
    return {'name':text(data,'name',150,True),'birth_date':born.isoformat(),'sex':data['sex'],'family_history':data['family_history'],'phone':text(data,'phone',40)}
