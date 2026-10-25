import json
from datetime import timedelta
from django.test import TestCase,Client
from django.utils import timezone
from django.db import connection
from identity.models import User
from clinical.adapters.persistence.models import Patient,SensorSession,SensorSample,ChartEntry
from clinical.domain.sensor import sample
PASSWORD='Prueba-segura-ESP32-123!'
class SensorTests(TestCase):
    def setUp(self):
        self.p=Patient.objects.create(payload={'name':'Paciente ficticio','birth_date':'1980-01-01','sex':0,'family_history':0},consent_at=timezone.now())
        self.p2=Patient.objects.create(payload={'name':'Otro','birth_date':'1990-01-01','sex':1,'family_history':1},consent_at=timezone.now())
        self.u=User.objects.create_user(username='paciente',password=PASSWORD,role='patient',patient=self.p)
        self.admin=User.objects.create_superuser(username='admin',password=PASSWORD,role='admin')
        self.c=Client();self.c.force_login(self.u)
    def post(self,path,data,c=None):return (c or self.c).post(path,json.dumps(data),content_type='application/json')
    def start(self,transport='wifi',cal=None):
        r=self.post(f'/api/patients/{self.p.id}/sensor-sessions',{'device_id':'PoliMQ-test','transport':transport,'weight':65,'height':160,'calibration_id':cal,'conditioned_sensor':True});self.assertEqual(r.status_code,201,r.content);return r.json()
    def point(self,seq):return {'device_id':'PoliMQ-test','seq':seq,'uptime_ms':180000+seq*1000,'raw_adc':2048,'adc_mv':1000}
    def send(self,start,seq,token=None):
        sid=start['session']['id'];return Client().post(f'/api/device/{sid}/samples',json.dumps(self.point(seq)),content_type='application/json',HTTP_AUTHORIZATION='Device '+(token or start['device_token']))
    def test_wifi_auth_scope_raw_and_finish(self):
        s=self.start();sid=s['session']['id'];self.assertEqual(self.send(s,0,'incorrecto').status_code,403)
        self.assertEqual(self.send(s,0).status_code,200);self.assertTrue(self.send(s,0).json()['duplicate']);self.assertEqual(SensorSample.objects.count(),1)
        self.assertEqual(self.c.get(f'/api/sensor-sessions/{sid}').json()['samples'][0]['quality'],'raw_only')
        self.assertNotIn('token_hash',str(self.c.get(f'/api/sensor-sessions/{sid}').json()))
        r=self.post(f'/api/sensor-sessions/{sid}/finish',{});self.assertEqual(r.json()['status'],'raw_only');self.assertEqual(ChartEntry.objects.count(),0)
        self.assertEqual(self.send(s,1).status_code,403)
        exported=self.c.get(f'/api/patients/{self.p.id}/export').json()
        self.assertEqual(len(exported['sensor_sessions'][0]['samples']),1)
        self.assertNotIn('token_hash',str(exported))
    def test_ble_owner_isolation_and_transport(self):
        s=self.start('ble');sid=s['session']['id'];self.assertEqual(self.send(s,0).status_code,403)
        self.assertEqual(self.post(f'/api/sensor-sessions/{sid}/samples',self.point(0)).status_code,200)
        other=Client();u=User.objects.create_user(username='other',password=PASSWORD,role='patient',patient=self.p2);other.force_login(u)
        self.assertEqual(other.get(f'/api/sensor-sessions/{sid}').status_code,404)
        self.assertEqual(self.post(f'/api/sensor-sessions/{sid}/samples',self.point(1),other).status_code,404)
        self.assertEqual(self.post(f'/api/patients/{self.p2.id}/sensor-sessions',{}).status_code,404)
    def test_calibration_admin_only_and_inference_once(self):
        profile={'device_id':'PoliMQ-test','reference':'SINTETICA SOLO TEST','r0_kohm':10,'rl_kohm':10,'vc_mv':5000,'divider':3,'m':-1,'b':0,'min_ppm':.1,'max_ppm':10,'warmup_s':60,'acetone_reference_verified':True}
        self.assertEqual(self.post('/api/calibrations',profile).status_code,403)
        admin=Client();admin.force_login(self.admin);r=self.post('/api/calibrations',profile,admin);self.assertEqual(r.status_code,201,r.content)
        s=self.start(cal=r.json()['id']);sid=s['session']['id']
        for i in range(5):self.assertEqual(self.send(s,i).status_code,200)
        result=self.post(f'/api/sensor-sessions/{sid}/finish',{}).json();self.assertEqual(result['status'],'calibrated_experimental');self.assertAlmostEqual(result['acetone'],1.5)
        self.assertEqual(ChartEntry.objects.count(),1);self.assertTrue(0<=result['score']<=1)
        self.post(f'/api/sensor-sessions/{sid}/finish',{});self.assertEqual(ChartEntry.objects.count(),1)
    def test_expiry_wrong_device_order_and_rate(self):
        s=self.start();sid=s['session']['id'];self.send(s,1)
        self.assertEqual(self.send(s,0).status_code,400)
        p=self.point(2);p['uptime_ms']=181100
        r=Client().post(f'/api/device/{sid}/samples',json.dumps(p),content_type='application/json',HTTP_AUTHORIZATION='Device '+s['device_token']);self.assertEqual(r.status_code,400)
        SensorSession.objects.filter(pk=sid).update(expires_at=timezone.now()-timedelta(seconds=1));self.assertEqual(self.send(s,2).status_code,403)
    def test_signal_validation_and_ciphertext(self):
        self.assertEqual(sample(self.point(0))['estimated_ppm'],None)
        data=self.point(0);data['raw_adc']=4095;self.assertEqual(sample(data)['quality'],'saturated_or_disconnected')
        s=self.start();self.send(s,0)
        with connection.cursor() as c:
            c.execute('SELECT payload FROM records_sensorsample');self.assertNotIn('adc_mv',c.fetchone()[0])
    def test_single_active_and_cancel(self):
        s=self.start();r=self.post(f'/api/patients/{self.p.id}/sensor-sessions',{'device_id':'PoliMQ-test','transport':'wifi','weight':65,'height':160,'conditioned_sensor':True});self.assertEqual(r.status_code,400)
        self.assertEqual(self.post('/api/sensor-sessions/'+s['session']['id']+'/finish',{'cancel':True}).json()['status'],'cancelled')
        self.assertEqual(self.send(s,0).status_code,403)
    def test_ble_csrf_and_anonymous_rejected(self):
        s=self.start('ble');url='/api/sensor-sessions/'+s['session']['id']+'/samples'
        self.assertEqual(self.post(url,self.point(0),Client()).status_code,401)
        guarded=Client(enforce_csrf_checks=True);guarded.force_login(self.u)
        self.assertEqual(self.post(url,self.point(0),guarded).status_code,403)
    def test_limits_and_wrong_device(self):
        s=self.start();sid=s['session']['id'];p=self.point(0);p['device_id']='wrong'
        r=Client().post(f'/api/device/{sid}/samples',json.dumps(p),content_type='application/json',HTTP_AUTHORIZATION='Device '+s['device_token']);self.assertEqual(r.status_code,403)
        SensorSession.objects.filter(pk=sid).update(sample_count=300);self.assertEqual(self.send(s,1).status_code,400)
