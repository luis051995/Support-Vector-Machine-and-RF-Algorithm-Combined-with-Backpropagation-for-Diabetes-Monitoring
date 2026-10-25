from rest_framework.response import Response
from rest_framework.views import exception_handler as drf_handler
from clinical.domain.entities import InvalidData,NotFound,Forbidden

def exception_handler(exc,context):
    for typ,status in [(InvalidData,400),(NotFound,404),(Forbidden,403)]:
        if isinstance(exc,typ):return Response({'error':str(exc)},status=status)
    r=drf_handler(exc,context)
    if r is not None:
        # Errores de validación no devuelven el cuerpo clínico original.
        r.data={'error':str(r.data.get('detail','Solicitud inválida.')) if isinstance(r.data,dict) else 'Solicitud inválida.'}
    return r
