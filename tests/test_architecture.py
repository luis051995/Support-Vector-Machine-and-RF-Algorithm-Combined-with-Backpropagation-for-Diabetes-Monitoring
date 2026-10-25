import subprocess,sys,unittest
from contextlib import nullcontext
from datetime import date
from clinical.domain.entities import Actor,Patient
from clinical.application.services import ClinicalService
class MemoryRepository:
    def __init__(self):self.rows={};self.history=[];self.logs=[]
    def atomic(self):return nullcontext()
    def create_patient(self,data):
        p=Patient(len(self.rows)+1,'TEST',data,'2026-10-24','2026-10-24');self.rows[p.id]=p;return p
    def get_patient(self,pid):return self.rows.get(pid)
    def audit(self,*args):self.logs.append(args)
    def add_entry(self,pid,author,kind,data):self.history.append(data);return {'id':len(self.history),**data}
class FakeRisk:
    def infer(self,features):return {'score':.25,'classification':'Prueba','model_version':'fake','received_order':list(features)}
class HexagonalTests(unittest.TestCase):
    def test_domain_imports_without_site_packages(self):
        r=subprocess.run([sys.executable,'-S','-c','from clinical.application.services import ClinicalService; from clinical.domain.ports import ClinicalRepository'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
    def test_case_runs_with_in_memory_ports(self):
        repo=MemoryRepository();service=ClinicalService(repo,FakeRisk(),today=lambda:date(2026,10,24));a=Actor(1,'admin')
        pid=service.create(a,{'name':'Paciente de prueba','birth_date':'1980-01-01','sex':1,'family_history':0,'consent':True})['id']
        r=service.reading(a,pid,{'acetone':1,'weight':70,'height':170})
        self.assertEqual(r['score'],.25);self.assertEqual(r['features_json']['Edad'],46);self.assertEqual(len(repo.logs),2)
