"""Importa líneas JSON con ppm YA CALIBRADAS desde puerto serial a un expediente.
No convierte voltaje a ppm sin calibración trazable del dispositivo.
Ejemplo de línea serial: {"acetone_ppm": 1.42}
"""
import argparse,getpass,json,time,urllib.request,urllib.error,http.cookiejar
import serial

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',required=True);p.add_argument('--baud',type=int,default=9600);p.add_argument('--api',default='http://127.0.0.1:8000');p.add_argument('--patient',type=int,required=True);p.add_argument('--weight',type=float,required=True);p.add_argument('--height',type=float,required=True);p.add_argument('--device',required=True);p.add_argument('--calibration',required=True);p.add_argument('--interval',type=float,default=60);a=p.parse_args()
    if a.interval<10:raise SystemExit('Intervalo mínimo: 10 segundos.')
    if not a.api.startswith('https://') and not a.api.startswith('http://127.0.0.1:'):raise SystemExit('Usa HTTPS fuera del equipo local.')
    op=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    def post(path,data,csrf=''):
        req=urllib.request.Request(a.api.rstrip('/')+'/api'+path,json.dumps(data).encode(),headers={'Content-Type':'application/json','X-CSRFToken':csrf,'Referer':a.api.rstrip('/')+'/'})
        with op.open(req,timeout=15) as r:return json.load(r)
    with op.open(a.api.rstrip('/')+'/api/csrf',timeout=15) as response:csrf=json.load(response)['csrf']
    auth=post('/login',{'username':input('Usuario profesional: '),'password':getpass.getpass('Contraseña: ')},csrf)
    last=0
    with serial.Serial(a.port,a.baud,timeout=2) as ser:
        print('Esperando JSON de acetona calibrada; Ctrl+C para detener.')
        while True:
            raw=ser.readline(1024)
            if not raw:continue
            try:
                value=float(json.loads(raw)['acetone_ppm'])
                if not 0<=value<=100:raise ValueError('Fuera de rango.')
                if time.monotonic()-last<a.interval:continue
                post(f'/patients/{a.patient}/readings',{'acetone':value,'weight':a.weight,'height':a.height,'source':'sensor','device_id':a.device,'calibration':a.calibration},auth['csrf']);last=time.monotonic();print('Medición guardada.')
            except (ValueError,KeyError,TypeError):print('Lectura inválida descartada.')
            except urllib.error.HTTPError as e:
                if e.code in (401,403):raise SystemExit('Sesión expirada o permiso insuficiente. Reinicia el puente.')
                print('Error HTTP:',e.code)
            except urllib.error.URLError:print('Sin conexión; lectura no guardada.')
if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('Puente detenido.')
