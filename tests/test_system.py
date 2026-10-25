import json
from datetime import timedelta
from unittest.mock import patch
from django.test import TestCase,Client,override_settings
from django.db import connection
from django.utils import timezone
from cryptography.fernet import Fernet
from identity.models import User,MobileSession
from clinical.adapters.persistence.models import Patient,ChartEntry,AuditEvent
from clinical.adapters.persistence.fields import cipher
from clinical.domain.entities import Actor,Forbidden
from clinical.application.services import ClinicalService

PASSWORD='Correcta-de-prueba-123!'
class IntegrationTests(TestCase):
    @classmethod
    def setUpTestData(cls):cls.admin=User.objects.create_superuser(username='admin',password=PASSWORD,role='admin')
    def setUp(self):self.client=Client(enforce_csrf_checks=True);self.login()
    def login(self):
        csrf=self.client.get('/api/csrf').json()['csrf'];r=self.client.post('/api/login',data=json.dumps({'username':'admin','password':PASSWORD}),content_type='application/json',HTTP_X_CSRFTOKEN=csrf)
        self.assertEqual(r.status_code,200,r.content);self.csrf=r.json()['csrf']
    def post(self,url,data,client=None,csrf=None):return (client or self.client).post(url,json.dumps(data),content_type='application/json',HTTP_X_CSRFTOKEN=csrf or self.csrf)
    def add(self,name='Paciente ficticio'):
        r=self.post('/api/patients',{'name':name,'birth_date':'1987-03-12','sex':0,'family_history':1,'phone':'555-ejemplo','consent':True});self.assertEqual(r.status_code,201,r.content);return r.json()['id']
    def test_encrypted_storage_and_integrity(self):
        pid=self.add('Nombre secreto único');self.post(f'/api/patients/{pid}/visits',{'reason':'Control','notes':'Nota clínica confidencial','plan':'Seguimiento'})
        with connection.cursor() as c:
            c.execute('SELECT payload FROM records_patient WHERE id=%s',[pid]);raw=c.fetchone()[0]
            c.execute('SELECT payload FROM records_chartentry WHERE patient_id=%s',[pid]);note=c.fetchone()[0]
        self.assertNotIn('Nombre secreto',raw);self.assertNotIn('confidencial',note);self.assertEqual(json.loads(cipher().decrypt(raw.encode()))['name'],'Nombre secreto único')
        with override_settings(DATA_ENCRYPTION_KEYS=[Fernet.generate_key().decode()]):
            from django.core.exceptions import ImproperlyConfigured
            with self.assertRaises(ImproperlyConfigured):Patient.objects.get(pk=pid).payload
        self.assertEqual(self.client.get(f'/api/patients/{pid}').json()['visits'][0]['notes'],'Nota clínica confidencial')
    def test_inference_contract_persistence_and_export(self):
        pid=self.add();r=self.post(f'/api/patients/{pid}/readings',{'acetone':2.1,'height':160,'weight':65})
        self.assertEqual(r.status_code,201,r.content);self.assertTrue(0<=r.json()['score']<=1)
        self.assertEqual(r.json()['features_json']['Acetona'],2.1)
        self.assertEqual(list(r.json()['features_json']),['Edad','Sexo','Peso','Altura','Antecedentes','Acetona'])
        d=self.client.get(f'/api/patients/{pid}/export');self.assertEqual(d.status_code,200);self.assertEqual(len(d.json()['readings']),1)
        self.assertTrue(AuditEvent.objects.filter(action='export').exists())
    def test_patient_isolation_and_roles(self):
        a=self.add();b=self.add();r=self.post('/api/users',{'username':'consulta','password':PASSWORD,'role':'patient','patient_id':a});self.assertEqual(r.status_code,201,r.content)
        c=Client(enforce_csrf_checks=True);csrf=c.get('/api/csrf').json()['csrf'];r=self.post('/api/login',{'username':'consulta','password':PASSWORD},c,csrf);csrf=r.json()['csrf']
        self.assertEqual(len(c.get('/api/patients').json()),1)
        self.assertEqual(c.get(f'/api/patients/{b}').status_code,404);self.assertEqual(c.get(f'/api/patients/{b}/export').status_code,404)
        self.assertEqual(self.post(f'/api/patients/{a}/visits',{'reason':'x'},c,csrf).status_code,403)
        self.assertEqual(c.get('/api/audit').status_code,403);self.assertEqual(c.get('/api/model').status_code,403)
    def test_csrf_login_and_mutations(self):
        c=Client(enforce_csrf_checks=True)
        self.assertEqual(c.post('/api/login',json.dumps({'username':'admin','password':PASSWORD}),content_type='application/json').status_code,403)
        self.assertEqual(self.client.post('/api/patients','{}',content_type='application/json').status_code,403)
        self.assertEqual(self.client.post('/api/patients','{}',content_type='application/json',HTTP_X_CSRFTOKEN=self.csrf,HTTP_ORIGIN='https://evil.invalid').status_code,403)
    def test_non_object_json_rejected(self):
        r=self.client.post('/api/patients','[]',content_type='application/json',HTTP_X_CSRFTOKEN=self.csrf)
        self.assertEqual(r.status_code,400)
    def test_no_consent_invalid_sensor_and_nan(self):
        self.assertEqual(self.post('/api/patients',{'name':'x'}).status_code,400);pid=self.add()
        for value in ['NaN','Infinity',-1,101]:self.assertEqual(self.post(f'/api/patients/{pid}/readings',{'acetone':value,'weight':70,'height':170}).status_code,400)
        self.assertEqual(self.post(f'/api/patients/{pid}/readings',{'acetone':1,'weight':70,'height':170,'source':'sensor'}).status_code,400)
    def test_native_token_expiry_password_revocation(self):
        c=Client();r=c.post('/api/login',json.dumps({'username':'admin','password':PASSWORD,'native':True}),content_type='application/json',HTTP_ORIGIN='https://localhost')
        self.assertEqual(r.status_code,200,r.content);token=r.json()['token'];c=Client(enforce_csrf_checks=True)
        self.assertEqual(c.get('/api/patients',HTTP_AUTHORIZATION='Bearer '+token).status_code,200)
        MobileSession.objects.update(expires_at=timezone.now()-timedelta(seconds=1));self.assertEqual(c.get('/api/patients',HTTP_AUTHORIZATION='Bearer '+token).status_code,401)
        self.assertEqual(c.post('/api/login',json.dumps({'username':'admin','password':PASSWORD,'native':True}),content_type='application/json',HTTP_ORIGIN='https://evil.invalid').status_code,403)
    def test_password_invalidates_web_sessions(self):
        c=Client();c.force_login(self.admin)
        self.assertEqual(self.post('/api/password',{'current':PASSWORD,'new':'Otra-clave-segura-456!'}).status_code,200)
        self.assertEqual(c.get('/api/me').status_code,401)
    def test_headers_and_sql_injection(self):
        pid=self.add("Paciente '); DROP TABLE identity_user; --")
        r=self.client.get(f'/api/patients/{pid}');self.assertEqual(r.status_code,200);self.assertEqual(r['Cache-Control'],'no-store');self.assertIn("frame-ancestors 'none'",r['Content-Security-Policy']);self.assertTrue(User.objects.filter(pk=self.admin.pk).exists())
    def test_no_sensitive_public_routes(self):
        c=Client();self.assertEqual(c.get('/api/patients').status_code,401);self.assertEqual(c.get('/models/hybrid.joblib').status_code,404);self.assertEqual(c.get('/.env').status_code,404);self.assertEqual(c.get('/').status_code,200)
    def test_login_throttled(self):
        c=Client(enforce_csrf_checks=True);csrf=c.get('/api/csrf').json()['csrf']
        for _ in range(9):r=c.post('/api/login',json.dumps({'username':'no-existe','password':'bad'}),content_type='application/json',HTTP_X_CSRFTOKEN=csrf)
        self.assertEqual(r.status_code,429)

class DomainWithoutDjangoTests(TestCase):
    def test_application_rejects_patient_before_adapters(self):
        class NoAdapter:
            def __getattr__(self,key):raise AssertionError('El adaptador no debe ejecutarse')
        s=ClinicalService(NoAdapter(),NoAdapter())
        with self.assertRaises(Forbidden):s.create(Actor(1,'patient',1),{})

class SelfRegistrationTests(TestCase):
    def setUp(self):self.client=Client(enforce_csrf_checks=True);self.csrf=self.client.get('/api/csrf').json()['csrf']
    def payload(self,username='paciente_nuevo'):
        return {'username':username,'password':PASSWORD,'password_confirm':PASSWORD,'name':'Paciente nuevo ficticio','birth_date':'1990-01-02','sex':0,'family_history':1,'phone':'','consent':True}
    def register(self,data):return self.client.post('/api/register',json.dumps(data),content_type='application/json',HTTP_X_CSRFTOKEN=self.csrf)
    def test_public_signup_fixed_role_and_own_chart(self):
        d=self.payload();d.update(role='admin',is_superuser=True,patient_id=999)
        r=self.register(d);self.assertEqual(r.status_code,201,r.content)
        u=User.objects.get(username=d['username']);self.assertEqual(u.role,'patient');self.assertFalse(u.is_superuser);self.assertFalse(u.is_staff);self.assertIsNotNone(u.patient_id)
        self.assertEqual(u.patient.payload['name'],d['name'])
        r=self.client.post('/api/login',json.dumps({'username':d['username'],'password':PASSWORD}),content_type='application/json',HTTP_X_CSRFTOKEN=self.csrf)
        self.assertEqual(r.status_code,200,r.content)
        self.assertEqual(len(self.client.get('/api/patients').json()),1)
        self.assertEqual(self.client.get(f'/api/patients/{u.patient_id}').status_code,200)
        self.assertEqual(self.client.get('/api/patients/999').status_code,404)
        self.assertEqual(self.client.get('/api/audit').status_code,403)
    def test_signup_rollback_duplicate_and_consent(self):
        self.assertEqual(self.register(self.payload()).status_code,201)
        self.assertEqual(self.register(self.payload()).status_code,400)
        self.assertEqual(Patient.objects.count(),1)
        d=self.payload('otro');d['consent']=False;self.assertEqual(self.register(d).status_code,400)
        self.assertEqual(Patient.objects.count(),1)
    def test_signup_csrf_and_rate_limit(self):
        self.assertEqual(self.client.post('/api/register',json.dumps(self.payload()),content_type='application/json').status_code,403)
        for _ in range(5):self.register({})
        self.assertEqual(self.register(self.payload()).status_code,429)
    def test_legacy_clinician_cannot_access_other_patients(self):
        from django.utils import timezone
        p=Patient.objects.create(payload={'name':'Ficticio'},consent_at=timezone.now())
        u=User.objects.create_user(username='legacy',password=PASSWORD,role='clinician')
        self.client.force_login(u)
        self.assertEqual(self.client.get('/api/patients').json(),[])
        self.assertEqual(self.client.get(f'/api/patients/{p.id}').status_code,404)
        self.assertEqual(self.client.get('/api/model').status_code,403)
