from clinical.domain.entities import InvalidData,text,validate_patient
from datetime import date
from clinical.domain.ports import ClinicalRepository,PatientAccounts
class PatientRegistration:
    """Alta atómica: una cuenta paciente nueva y su propio expediente nuevo."""
    def __init__(self,repository:ClinicalRepository,accounts:PatientAccounts,today=date.today):self.repo=repository;self.accounts=accounts;self.today=today
    def register(self,data):
        if data.get('consent') is not True:raise InvalidData('Debes aceptar el registro de tus datos para crear tu expediente.')
        patient=validate_patient(data,self.today())
        username=text(data,'username',100,True);password=text(data,'password',256,True)
        if password!=data.get('password_confirm'):raise InvalidData('Las contraseñas no coinciden.')
        self.accounts.validate(username,password)
        with self.repo.atomic():
            chart=self.repo.create_patient(patient)
            uid=self.accounts.create_patient_account(username,password,chart.id)
            self.repo.audit(uid,'self_register','patient',chart.id)
        return {'ok':True,'message':'Cuenta creada. Inicia sesión para consultar tu historial.'}
