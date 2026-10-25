import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.utils import timezone
from rest_framework.decorators import api_view
from rest_framework.response import Response
from clinical.application.live import LiveService
from clinical.adapters.persistence.live import DjangoLiveRepository
from clinical.adapters.persistence.models import SensorCalibration,SensorSession
from clinical.adapters.ml.risk import SklearnRiskAdapter
from clinical.domain.entities import InvalidData,NotFound,Forbidden
from .views import actor

def service():return LiveService(DjangoLiveRepository(),SklearnRiskAdapter(),timezone.now,timezone.localdate)
def public(data):
    if isinstance(data,dict):return {k:public(v) for k,v in data.items() if k!='token_hash'}
    if isinstance(data,list):return [public(v) for v in data]
    return data
@api_view(['GET','POST'])
def sessions(request,pid):
    a=actor(request);a.require_patient(pid)
    if request.method=='POST':return Response(public(service().start(a,pid,request.data)),status=201)
    return Response([DjangoLiveRepository().dto(s,False) for s in SensorSession.objects.filter(patient_id=pid).order_by('-created_at')])
@api_view(['GET'])
def status(request,sid):
    try:after=int(request.GET.get('after','-1'))
    except ValueError:raise InvalidData('Secuencia inválida.')
    return Response(public(service().status(actor(request),sid,after)))
@api_view(['POST'])
def ble_sample(request,sid):return Response(service().ingest(sid,request.data,actor=actor(request)))
@api_view(['POST'])
def finish(request,sid):return Response(service().finish(actor(request),sid,request.data.get('cancel') is True))
@api_view(['GET','POST'])
def calibrations(request):
    if request.method=='POST':return Response(service().add_calibration(actor(request),request.data),status=201)
    return Response([{'id':c.id,'device_id':c.device_id,'reference':c.reference} for c in SensorCalibration.objects.all().order_by('-id')[:100]])
@csrf_exempt
@require_POST
def device_ingest(request,sid):
    # Credencial efímera de una sola prueba, sin cookies, PII ni permisos de lectura.
    raw=request.headers.get('Authorization','')
    if not raw.startswith('Device '):return JsonResponse({'error':'Credencial requerida.'},status=401)
    if request.content_type!='application/json':return JsonResponse({'error':'Solo JSON.'},status=400)
    try:
        data=json.loads(request.body)
        if not isinstance(data,dict):raise InvalidData('Objeto JSON requerido.')
        return JsonResponse(service().ingest(sid,data,token=raw[7:]))
    except (InvalidData,ValueError) as e:return JsonResponse({'error':str(e)},status=400)
    except Forbidden as e:return JsonResponse({'error':str(e)},status=403)
    except NotFound as e:return JsonResponse({'error':str(e)},status=404)
