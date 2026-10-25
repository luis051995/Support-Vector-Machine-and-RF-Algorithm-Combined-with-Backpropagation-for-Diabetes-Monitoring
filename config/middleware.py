from django.views.debug import SafeExceptionReporterFilter
class RedactingExceptionFilter(SafeExceptionReporterFilter):
    def get_post_parameters(self,request):return {'body':'[redacted]'}
    def get_traceback_frame_variables(self,request,tb_frame):return []
class SecurityHeadersMiddleware:
    def __init__(self,get_response):self.get_response=get_response
    def __call__(self,request):
        r=self.get_response(request)
        if request.path.startswith('/api') or request.path=='/':r.headers['Cache-Control']='no-store'
        if not request.path.startswith('/admin'):
            r.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self' https:; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
        r.headers['Permissions-Policy']='camera=(), microphone=(), geolocation=()'
        return r
