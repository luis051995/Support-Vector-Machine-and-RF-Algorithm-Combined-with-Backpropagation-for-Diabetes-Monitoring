'use strict';
const LIVE_SERVICE='7de71000-7aa6-4ca0-a670-54ba95b459cf',LIVE_SAMPLE='7de71001-7aa6-4ca0-a670-54ba95b459cf';
let liveState=null,liveTimer=null,bleDisconnect=null,liveBusy=false;
async function openSensorTest(){
 try{
  const profiles=await api('/calibrations'),pid=current.patient.id;
  const previous=await api('/patients/'+pid+'/sensor-sessions');
  const active=previous.find(x=>!x.closed_at&&new Date(x.expires_at)>new Date());
  if(active){showLive(active,null);return;}
  openForm('Prueba en tiempo real · ESP32',`<p class="muted">Paciente: <b>${esc(current.patient.name)}</b>. La prueba queda vinculada a este expediente durante 10 minutos; máximo 300 muestras.</p><div class="form-grid">${field('device_id','Identificador del ESP32 (ver monitor serial)')}${select('transport','Conexión',[['wifi','Wi-Fi: ESP32 → Django'],['ble','Bluetooth LE: ESP32 → este dispositivo']],'wifi')}${field('weight','Peso actual (kg)','number',current.readings[0]?.weight||'')}${field('height','Altura (cm)','number',current.readings[0]?.height||'')}${select('calibration_id','Calibración específica del dispositivo',[['','Solo señal cruda (sin IA)'],...profiles.map(p=>[p.id,p.device_id+' · '+p.reference])],'')}</div><label class="check"><input name="conditioned_sensor" type="checkbox" required>Confirmo acondicionamiento y estabilización del MQ-138 según fabricante y protocolo de laboratorio. Esta prueba es experimental.</label><p class="muted">Sin calibración de acetona solo se guarda la señal eléctrica; no se fabrica una concentración ppm.</p>`,async d=>{
   const r=await api('/patients/'+pid+'/sensor-sessions','POST',{...d,weight:Number(d.weight),height:Number(d.height),calibration_id:d.calibration_id?Number(d.calibration_id):null,conditioned_sensor:!!d.conditioned_sensor});showLive(r.session,r.device_token);
  });document.querySelectorAll('#modal input[type=number]').forEach(e=>e.step='any');
 }catch(e){toast(e.message)}
}
function showLive(session,deviceToken){
 clearTimeout(liveTimer);liveState={session,deviceToken,points:[],after:-1};
 let dialog=$('#sensor-modal');if(!dialog){dialog=document.createElement('dialog');dialog.id='sensor-modal';document.body.append(dialog);}
 dialog.innerHTML=`<div class="modal-head"><div><div class="eyebrow-wine">PRUEBA EXPERIMENTAL · MQ-138</div><h2>Señal en tiempo real</h2></div><button class="quiet" id="live-close" aria-label="Cerrar vista">✕</button></div><p>${esc(current.patient.name)} · <b>${esc(session.meta.device_id)}</b> · ${session.meta.transport==='wifi'?'Wi-Fi':'Bluetooth LE'}</p><div class="stats"><div class="stat"><span>Voltaje ADC</span><strong id="live-mv">—</strong><small>mV · entrada del ESP32</small></div><div class="stat"><span>Concentración estimada</span><strong id="live-ppm">—</strong><small>ppm · solo con calibración</small></div><div class="stat"><span>Muestras guardadas</span><strong id="live-count">${session.sample_count}</strong><small>máximo 300 por prueba</small></div></div><div id="live-plot" class="live-plot"></div><p id="live-status" class="muted" role="status">Esperando lecturas del sensor…</p><div id="wifi-config"></div><div class="actions">${session.meta.transport==='ble'?'<button class="primary" id="connect-ble">Conectar ESP32 por BLE</button>':''}<button class="primary" id="live-finish">Finalizar y guardar resultado</button><button class="secondary" id="live-cancel">Cancelar prueba</button></div><div class="notice">Mantén esta vista abierta para retransmitir BLE. Wi-Fi llega directamente al servidor. Las muestras sin calibración se conservan como señal cruda; no son diagnóstico.</div>`;
 if(session.meta.transport==='wifi'){
  const endpoint=(cfg.apiBase||location.origin)+'/api/device/'+session.id+'/samples';
  if(deviceToken){const command=JSON.stringify({url:endpoint,token:deviceToken,mode:'wifi'});$('#wifi-config').innerHTML='<label>Configuración temporal: pega esta línea en el monitor serial del ESP32<textarea id="device-command" readonly></textarea></label><p class="muted">No compartas esta credencial. Solo permite enviar señal a esta prueba y vence en 10 minutos.</p>';$('#device-command').value=command;}
  else $('#wifi-config').innerHTML='<p class="muted">Prueba reabierta. La credencial se mostró solo al crearla. Si el ESP32 no la conserva, cancela y abre una nueva.</p>';
 }
 $('#live-close').onclick=closeLive;dialog.oncancel=e=>{e.preventDefault();closeLive()};$('#live-finish').onclick=()=>finishLive(false);$('#live-cancel').onclick=()=>finishLive(true);$('#connect-ble')?.addEventListener('click',connectLiveBLE);dialog.showModal();pollLive();
}
function paintLive(){
 if(!liveState)return;const pts=liveState.points.slice(-60),last=pts.at(-1);if(!last)return;
 $('#live-mv').textContent=last.adc_mv;$('#live-ppm').textContent=last.estimated_ppm==null?'—':last.estimated_ppm.toFixed(3);
 const coords=pts.map((p,i)=>`${10+i*580/Math.max(1,pts.length-1)},${140-p.adc_mv/3100*130}`).join(' ');
 $('#live-plot').innerHTML=`<svg viewBox="0 0 610 160" role="img" aria-label="Últimas muestras de voltaje ADC, escala de 0 a 3100 milivoltios"><path d="M10 10V140H600" stroke="#ddd" fill="none"/><polyline points="${coords}" fill="none" stroke="#681637" stroke-width="2"/></svg><small class="muted">Últimas 60 muestras · 0–3100 mV. No representa glucosa.</small>`;
 $('#live-status').textContent=({raw_only:'Señal cruda: sin calibración de acetona.',warming:'Sensor estabilizándose; todavía no se estima ppm.',saturated_or_disconnected:'Entrada saturada o posible desconexión. Revisa el circuito.',invalid_voltage:'Voltaje incompatible con la calibración.',out_of_calibration:'Lectura fuera del intervalo calibrado.',calibrated_experimental:'Estimación experimental dentro del intervalo de calibración.'}[last.quality]||last.quality);
}
async function pollLive(){
 if(!liveState)return;const id=liveState.session.id;
 try{const r=await api('/sensor-sessions/'+id+'?after='+liveState.after);if(!liveState||id!==liveState.session.id)return;
  liveState.session=r.session;for(const p of r.samples){liveState.points.push(p);liveState.after=p.seq;}liveState.points=liveState.points.slice(-300);$('#live-count').textContent=r.session.sample_count;paintLive();
  if(r.expired||r.session.closed_at){$('#live-status').textContent=r.session.closed_at?'Prueba finalizada.':'La credencial venció. Finaliza para conservar el resumen.';await disconnectBLE();return;}
 }catch(e){if($('#live-status'))$('#live-status').textContent=e.message;}
 if(liveState&&id===liveState.session.id)liveTimer=setTimeout(pollLive,1000);
}
function decodeLivePacket(v){if(v.byteLength!==16||v.getUint32(12,true)!==1)throw Error('Paquete BLE incompatible.');return {seq:v.getUint32(0,true),uptime_ms:v.getUint32(4,true),raw_adc:v.getUint16(8,true),adc_mv:v.getUint16(10,true),device_id:liveState.session.meta.device_id};}
async function sendBle(v){
 if(!liveState||liveBusy)return;const sid=liveState.session.id;liveBusy=true;
 try{const point=decodeLivePacket(v);if($('#live-mv'))$('#live-mv').textContent=point.adc_mv;await api('/sensor-sessions/'+sid+'/samples','POST',point);}catch(e){if($('#live-status'))$('#live-status').textContent='Lectura no guardada: '+e.message;}finally{liveBusy=false;}
}
async function connectLiveBLE(){
 try{
  if(window.PoliBle){bleDisconnect=await window.PoliBle.connect(LIVE_SERVICE,LIVE_SAMPLE,sendBle,liveState.session.meta.device_id);}
  else{
   if(!navigator.bluetooth||!window.isSecureContext)throw Error('Este navegador no ofrece BLE aquí. Usa la app nativa Android/iPhone o Chrome/Edge con HTTPS (localhost en laptop también sirve). Para la web HTTP del teléfono usa Wi-Fi.');
   const device=await navigator.bluetooth.requestDevice({filters:[{services:[LIVE_SERVICE],name:liveState.session.meta.device_id}]});const server=await device.gatt.connect();const service=await server.getPrimaryService(LIVE_SERVICE);const c=await service.getCharacteristic(LIVE_SAMPLE);await c.startNotifications();c.addEventListener('characteristicvaluechanged',e=>sendBle(e.target.value));bleDisconnect=async()=>{device.gatt.disconnect()};
  }
  $('#connect-ble').disabled=true;$('#live-status').textContent='BLE conectado. Esperando datos…';
 }catch(e){$('#live-status').textContent=e.message;}
}
async function disconnectBLE(){if(bleDisconnect){try{await bleDisconnect()}catch(e){}bleDisconnect=null;}}
async function closeLive(){clearTimeout(liveTimer);liveState=null;await disconnectBLE();$('#sensor-modal').close();}
async function finishLive(cancel){
 if(!liveState)return;const sid=liveState.session.id,pid=liveState.session.patient_id;$('#live-finish').disabled=true;$('#live-cancel').disabled=true;
 try{const r=await api('/sensor-sessions/'+sid+'/finish','POST',{cancel});await closeLive();await loadPatient(pid);toast(cancel?'Prueba cancelada; señal conservada.':r.message);}
 catch(e){toast(e.message);$('#live-finish').disabled=false;$('#live-cancel').disabled=false;}
}
async function showSensorHistory(){
 try{const sessions=await api('/patients/'+current.patient.id+'/sensor-sessions');$('#history').innerHTML=sessions.length?sessions.map(s=>`<section class="card"><div class="card-body"><h3>${esc(s.meta.device_id)} · ${s.meta.transport==='wifi'?'Wi-Fi':'BLE'}</h3><p class="muted">${esc(date(s.created_at))} · ${s.sample_count} muestras</p><p>${esc(s.meta.summary?.message||'Prueba activa o pendiente de finalizar.')}</p><button class="secondary" data-live-session="${s.id}">Ver señal y estado</button></div></section><br>`).join(''):'<div class="empty">Sin pruebas ESP32.</div>';
  document.querySelectorAll('[data-live-session]').forEach(b=>b.onclick=()=>showLive(sessions.find(s=>s.id===b.dataset.liveSession),null));
 }catch(e){toast(e.message)}
}
async function calibrationForm(){
 openForm('Registrar calibración de acetona',`<p class="muted">Solo para operador de laboratorio: introduce parámetros obtenidos con una referencia conocida. No uses coeficientes VOC genéricos.</p><div class="form-grid">${field('device_id','Identificador del ESP32')}${field('reference','Referencia y protocolo de calibración')}${[['r0_kohm','R0 (kΩ)'],['rl_kohm','RL real (kΩ)'],['vc_mv','Voltaje del circuito (mV)'],['divider','Factor del divisor AO/ADC'],['m','Pendiente m (negativa)'],['b','Intercepto b'],['min_ppm','Mínimo calibrado (ppm)'],['max_ppm','Máximo calibrado (ppm)'],['warmup_s','Estabilización tras encendido (segundos)']].map(([k,l])=>field(k,l,'number')).join('')}</div><label class="check"><input type="checkbox" name="acetone_reference_verified" required>Confirmo calibración documentada específica para acetona y rango declarado.</label>`,async d=>{await api('/calibrations','POST',{...d,acetone_reference_verified:!!d.acetone_reference_verified});toast('Calibración registrada.');});document.querySelectorAll('#modal input[type=number]').forEach(e=>e.step='any');
}
