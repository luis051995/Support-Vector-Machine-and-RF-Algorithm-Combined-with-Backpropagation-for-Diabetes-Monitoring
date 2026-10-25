def client_ip(request):return request.META.get('REMOTE_ADDR')
def csrf_failure(request,reason=''):
    if request.path.startswith('/api/'):
        from django.http import JsonResponse
        return JsonResponse({'error':'La sesión o el token CSRF no son válidos. Recarga e inicia sesión de nuevo.'},status=403)
    from django.views.csrf import csrf_failure as default
    return default(request,reason=reason)
