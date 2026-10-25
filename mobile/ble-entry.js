import {BleClient} from '@capacitor-community/bluetooth-le';
window.PoliBle={async connect(service,characteristic,onValue,name){
 await BleClient.initialize();
 const d=await BleClient.requestDevice({services:[service],name});
 await BleClient.connect(d.deviceId,()=>{const e=document.getElementById('live-status');if(e)e.textContent='BLE desconectado. Reconecta para continuar; los datos guardados permanecen.'});
 try{
  await BleClient.read(d.deviceId,service,characteristic); // Requiere enlace cifrado y PIN.
  await BleClient.startNotifications(d.deviceId,service,characteristic,onValue);
 }catch(e){await BleClient.disconnect(d.deviceId);throw e;}
 return async()=>{try{await BleClient.stopNotifications(d.deviceId,service,characteristic)}finally{await BleClient.disconnect(d.deviceId)}};
}};
