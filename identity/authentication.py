import hashlib
from django.utils import timezone
from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed
from .models import MobileSession
class ExpiringBearerAuthentication(BaseAuthentication):
    def authenticate(self,request):
        raw=request.headers.get('Authorization','')
        if not raw.startswith('Bearer '):return None
        token=raw[7:]
        if len(token)>200:raise AuthenticationFailed('Sesión inválida.')
        s=MobileSession.objects.select_related('user').filter(token_hash=hashlib.sha256(token.encode()).hexdigest(),expires_at__gt=timezone.now()).first()
        if s is None or not s.user.is_active:raise AuthenticationFailed('Sesión inválida o expirada.')
        return s.user,s
    def authenticate_header(self,request):return 'Bearer'
