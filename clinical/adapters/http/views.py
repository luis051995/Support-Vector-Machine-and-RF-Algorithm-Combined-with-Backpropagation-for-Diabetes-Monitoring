import hashlib,json,secrets
from datetime import timedelta
from functools import wraps
from django.conf import settings
from django.contrib.auth import authenticate,login as session_login,logout as session_logout
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.db import transaction,IntegrityError
from django.http import JsonResponse
from django.middleware.csrf import get_token
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.csrf import csrf_protect,csrf_exempt,ensure_csrf_cookie
from django.views.decorators.http import require_POST,require_GET
from django.views.decorators.debug import sensitive_post_parameters,sensitive_variables
from rest_framework.decorators import api_view
from rest_framework.response import Response
from identity.models import User,MobileSession
from clinical.bootstrap import service
from clinical.domain.entities import Actor,InvalidData,Forbidden,NotFound,text
from clinical.adapters.persistence.models import Patient,AuditEvent

def actor(request):return Actor(request.user.id,'admin' if request.user.is_superuser else request.user.role,request.user.patient_id)
def profile(user):return {'id':user.id,'username':user.username,'role':'admin' if user.is_superuser else user.role,'patient_id':user.patient_id}
def home(request):return render(request,'index.html')
@require_GET
@ensure_csrf_cookie
def csrf(request):return JsonResponse({'csrf':get_token(request)})
@require_GET
def health(request):return JsonResponse({'status':'ok','framework':'Django','model_available':(settings.BASE_DIR/'models/hybrid.joblib').exists(),'clinical_validation':False})

@sensitive_variables('data','password','raw','token','user')
def _login(request,native=False):
    if request.content_type!='application/json':return JsonResponse({'error':'Solo JSON.'},status=400)
    try:data=json.loads(request.body)
    except (ValueError,TypeError):return JsonResponse({'error':'JSON inválido.'},status=400)
    if not isinstance(data,dict):return JsonResponse({'error':'Objeto JSON requerido.'},status=400)
    username=data.get('username','');password=data.get('password','')
    if not isinstance(username,str) or not isinstance(password,str) or len(username)>150 or len(password)>256:return JsonResponse({'error':'Credenciales inválidas.'},status=400)
    user=authenticate(request,username=username,password=password)
    if user is None:return JsonResponse({'error':'Usuario o contraseña incorrectos.'},status=401)
    token=None
    with transaction.atomic():
        if native:
            token=secrets.token_urlsafe(48)
            MobileSession.objects.create(token_hash=hashlib.sha256(token.encode()).hexdigest(),user=user,expires_at=timezone.now()+timedelta(hours=8))
        else:session_login(request,user)
        AuditEvent.objects.create(actor=user,action='login',entity='user',entity_id=user.id)
    return JsonResponse({'user':profile(user),'csrf':get_token(request),'token':token})
@csrf_exempt
@require_POST
def login(request):
    # Solo el login nativo con origen exacto omite CSRF; el login web se protege.
    try:data=json.loads(request.body)
    except (ValueError,TypeError):return JsonResponse({'error':'JSON inválido.'},status=400)
    native=isinstance(data,dict) and data.get('native') is True
    if native:
        if request.headers.get('Origin') not in settings.CORS_ALLOWED_ORIGINS:return JsonResponse({'error':'Origen móvil no permitido.'},status=403)
        return _login(request,True)
    return csrf_protect(_login)(request)
@api_view(['GET'])
def me(request):return Response({'user':profile(request.user),'csrf':get_token(request)})
@api_view(['POST'])
def logout(request):
    AuditEvent.objects.create(actor=request.user,action='logout',entity='user',entity_id=request.user.id)
    if isinstance(request.auth,MobileSession):request.auth.delete()
    else:session_logout(request)
    return Response({'ok':True})
@api_view(['GET','POST'])
def patients(request):
    s=service();a=actor(request)
    return Response(s.list(a) if request.method=='GET' else s.create(a,request.data),status=200 if request.method=='GET' else 201)
@api_view(['GET','PUT'])
def patient(request,pid):
    s=service();a=actor(request)
    return Response(s.detail(a,pid) if request.method=='GET' else s.update(a,pid,request.data))
@api_view(['POST'])
def visits(request,pid):return Response(service().visit(actor(request),pid,request.data),status=201)
@api_view(['POST'])
def readings(request,pid):return Response(service().reading(actor(request),pid,request.data),status=201)
@api_view(['GET'])
def export(request,pid):
    data=service().detail(actor(request),pid,True)
    from clinical.adapters.persistence.models import SensorSession
    from clinical.adapters.persistence.live import DjangoLiveRepository
    repo=DjangoLiveRepository();data['sensor_sessions']=[{**repo.dto(s,False),'samples':repo.samples(s.pk,-1,300)} for s in SensorSession.objects.filter(patient_id=pid).order_by('created_at')]
    r=Response(data);r['Content-Disposition']=f'attachment; filename=expediente-{pid}.json';return r
@api_view(['GET'])
def model(request):return Response(service().metrics(actor(request)))
@api_view(['GET'])
def audit(request):
    if actor(request).role!='admin':raise Forbidden('Solo administradores.')
    return Response([{'id':r.id,'username':r.actor.username,'action':r.action,'entity':r.entity,'entity_id':r.entity_id,'created_at':r.created_at.isoformat()} for r in AuditEvent.objects.select_related('actor')[:200]])
@api_view(['POST'])
@sensitive_variables('data','password','user')
def users(request):
    if actor(request).role!='admin':raise Forbidden('Solo administradores.')
    data=request.data
    username=text(data,'username',100,True);password=text(data,'password',256,True);role=data.get('role');pid=data.get('patient_id')
    if role not in ('admin','patient'):raise InvalidData('Rol inválido.')
    if role=='patient' and (type(pid) is not int or not Patient.objects.filter(pk=pid).exists()):raise InvalidData('Expediente requerido para el paciente.')
    if role!='patient':pid=None
    u=User(username=username,role=role,patient_id=pid)
    try:validate_password(password,u)
    except ValidationError as e:raise InvalidData(' '.join(e.messages))
    try:
        with transaction.atomic():
            u.set_password(password);u.full_clean();u.save();AuditEvent.objects.create(actor=request.user,action='create',entity='user',entity_id=u.id)
    except (IntegrityError,ValidationError):raise InvalidData('Usuario duplicado o datos de cuenta inválidos.')
    return Response({'id':u.id},status=201)
@api_view(['POST'])
@sensitive_variables('data','old','new')
def password(request):
    data=request.data;old=data.get('current','');new=data.get('new','')
    if not isinstance(old,str) or not isinstance(new,str) or len(new)>256 or not request.user.check_password(old):raise InvalidData('Contraseña actual incorrecta o nueva inválida.')
    try:validate_password(new,request.user)
    except ValidationError as e:raise InvalidData(' '.join(e.messages))
    with transaction.atomic():
        request.user.set_password(new);request.user.save(update_fields=['password']);MobileSession.objects.filter(user=request.user).delete();AuditEvent.objects.create(actor=request.user,action='password_changed',entity='user',entity_id=request.user.id)
    # Django invalida el hash de autenticación de las demás sesiones automáticamente.
    session_logout(request);return Response({'ok':True})

@sensitive_variables('data')
def _register(request):
    from clinical.application.registration import PatientRegistration
    from clinical.adapters.persistence.repository import DjangoClinicalRepository
    from identity.registration import DjangoPatientAccounts
    from identity.models import RegistrationAttempt
    from django.utils.crypto import salted_hmac
    if request.content_type!='application/json':return JsonResponse({'error':'Solo JSON.'},status=400)
    key=salted_hmac('registration-rate',request.META.get('REMOTE_ADDR','')).hexdigest()
    with transaction.atomic():
        RegistrationAttempt.objects.get_or_create(key=key,defaults={'started_at':timezone.now()})
        attempt=RegistrationAttempt.objects.select_for_update().get(pk=key)
        if timezone.now()-attempt.started_at>=timedelta(hours=1):attempt.count=0;attempt.started_at=timezone.now()
        if attempt.count>=5:return JsonResponse({'error':'Límite de registros alcanzado. Intenta de nuevo en una hora.'},status=429)
        attempt.count+=1;attempt.save()
    try:
        data=json.loads(request.body)
        if not isinstance(data,dict):raise InvalidData('Se requiere un objeto JSON.')
        result=PatientRegistration(DjangoClinicalRepository(),DjangoPatientAccounts(),today=timezone.localdate).register(data)
        return JsonResponse(result,status=201)
    except (InvalidData,ValueError) as e:return JsonResponse({'error':str(e)},status=400)
@csrf_exempt
@require_POST
def register(request):
    try:data=json.loads(request.body)
    except (ValueError,TypeError):return JsonResponse({'error':'JSON inválido.'},status=400)
    if isinstance(data,dict) and data.get('native') is True:
        if request.headers.get('Origin') not in settings.CORS_ALLOWED_ORIGINS:return JsonResponse({'error':'Origen móvil no permitido.'},status=403)
        return _register(request)
    return csrf_protect(_register)(request)
