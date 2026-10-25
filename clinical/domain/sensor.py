from math import log10,isfinite
from .entities import InvalidData,numeric,text

def calibration(data):
    if data.get('acetone_reference_verified') is not True:raise InvalidData('Se requiere referencia trazable específica de acetona, no una curva genérica VOC.')
    out={k:numeric(data,k,*bounds) for k,bounds in {'r0_kohm':(.001,100000),'rl_kohm':(.001,100000),'vc_mv':(1000,5500),'divider':(1,10),'m':(-10,-.001),'b':(-20,20),'min_ppm':(.001,100),'max_ppm':(.001,100),'warmup_s':(60,172800)}.items()}
    if out['max_ppm']<=out['min_ppm']:raise InvalidData('Intervalo ppm inválido.')
    return out

def sample(data,profile=None):
    result={k:numeric(data,k,*bounds) for k,bounds in {'seq':(0,4294967295),'uptime_ms':(0,4294967295),'raw_adc':(0,4095),'adc_mv':(0,3100)}.items()}
    for k in result:
        if result[k]!=int(result[k]):raise InvalidData(f'{k}: entero requerido.')
        result[k]=int(result[k])
    result.update(device_id=text(data,'device_id',64,True),estimated_ppm=None,quality='raw_only')
    if result['raw_adc']<=5 or result['raw_adc']>=4090 or not 20<=result['adc_mv']<=3000:result['quality']='saturated_or_disconnected';return result
    if not profile:return result
    if result['uptime_ms']<profile['warmup_s']*1000:result['quality']='warming';return result
    vs=result['adc_mv']*profile['divider']
    if not 0<vs<profile['vc_mv']:result['quality']='invalid_voltage';return result
    rs=(profile['vc_mv']/vs-1)*profile['rl_kohm']
    exponent=(log10(rs/profile['r0_kohm'])-profile['b'])/profile['m']
    if not -10<=exponent<=5:result['quality']='out_of_calibration';return result
    ppm=10**exponent
    if not isfinite(ppm) or not profile['min_ppm']<=ppm<=profile['max_ppm']:result['quality']='out_of_calibration';return result
    result['estimated_ppm']=ppm;result['quality']='calibrated_experimental';return result
