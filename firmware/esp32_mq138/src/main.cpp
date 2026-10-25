#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <BLESecurity.h>
#include <esp_gap_ble_api.h>
#include <time.h>
#if __has_include("polisalud_config.h")
#include "polisalud_config.h"
#else
#include "polisalud_config.example.h"
#endif
static const char* SERVICE="7de71000-7aa6-4ca0-a670-54ba95b459cf";
static const char* SAMPLE="7de71001-7aa6-4ca0-a670-54ba95b459cf";
String deviceId,endpoint,deviceToken,serialLine;
uint32_t sequence=0,lastSample=0,sessionConfiguredAt=0,pairUntil=0;
bool bleConnected=false,bleAuthenticated=false,wasPressed=false;
BLECharacteristic* measure=nullptr;
struct __attribute__((packed)) Packet {uint32_t seq;uint32_t uptime;uint16_t adc;uint16_t mv;uint32_t protocol;};
static_assert(sizeof(Packet)==16,"BLE packet protocol size");
class Security:public BLESecurityCallbacks {
 uint32_t onPassKeyRequest(){return BLE_PAIR_PIN;}
 void onPassKeyNotify(uint32_t pass_key){Serial.printf("PIN de emparejamiento BLE: %06u\n",pass_key);}
 bool onConfirmPIN(uint32_t pin){return (millis()<pairUntil)&&pin==BLE_PAIR_PIN;}
 bool onSecurityRequest(){return millis()<pairUntil;}
 void onAuthenticationComplete(esp_ble_auth_cmpl_t result){bleAuthenticated=result.success;Serial.println(result.success?"BLE autenticado":"BLE rechazado");}
};
class Connections:public BLEServerCallbacks {
 void onConnect(BLEServer* server,esp_ble_gatts_cb_param_t* param){
  bleConnected=true;bleAuthenticated=false;
  if(millis()>=pairUntil){server->disconnect(param->connect.conn_id);return;}
  esp_ble_set_encryption(param->connect.remote_bda,ESP_BLE_SEC_ENCRYPT_MITM);
 }
 void onDisconnect(BLEServer*){bleConnected=false;bleAuthenticated=false;}
};
bool lanUrl(String url){
 if(!url.startsWith("http://"))return false;
 String host=url.substring(7);int slash=host.indexOf('/');if(slash>=0)host=host.substring(0,slash);int colon=host.indexOf(':');if(colon>=0)host=host.substring(0,colon);
 IPAddress ip;if(!ip.fromString(host))return false;
 return ip[0]==10||(ip[0]==192&&ip[1]==168)||(ip[0]==172&&ip[1]>=16&&ip[1]<=31);
}
void sendHttp(Packet p){
 if(endpoint.isEmpty()||WiFi.status()!=WL_CONNECTED||deviceToken.isEmpty())return;
 if(millis()-sessionConfiguredAt>600000){deviceToken="";endpoint="";Serial.println("Credencial Wi-Fi vencida. Abre otra prueba.");return;}
 JsonDocument doc;doc["device_id"]=deviceId;doc["seq"]=p.seq;doc["uptime_ms"]=p.uptime;doc["raw_adc"]=p.adc;doc["adc_mv"]=p.mv;
 String payload;serializeJson(doc,payload);HTTPClient http;WiFiClient plain;WiFiClientSecure tls;
 if(endpoint.startsWith("https://")){
  if(strlen(ROOT_CA_PEM)<40){Serial.println("Configura ROOT_CA_PEM para HTTPS.");return;}
  if(time(nullptr)<1700000000){Serial.println("Esperando hora NTP para TLS.");return;}
  tls.setCACert(ROOT_CA_PEM);http.begin(tls,endpoint);
 }else if(ALLOW_LAN_HTTP&&lanUrl(endpoint)){http.begin(plain,endpoint);}
 else {Serial.println("Se requiere HTTPS; HTTP solo para IP privada y laboratorio habilitado.");return;}
 http.setConnectTimeout(3000);http.setTimeout(3000);http.addHeader("Content-Type","application/json");http.addHeader("Authorization","Device "+deviceToken);
 int result=http.POST(payload);Serial.printf("Muestra %u enviada: HTTP %d\n",p.seq,result);
 if(result==401||result==403){deviceToken="";endpoint="";Serial.println("Prueba cerrada o token rechazado. No se reenvía.");}
 http.end();
}
void command(String line){
 JsonDocument doc;if(deserializeJson(doc,line)){Serial.println("Comando JSON inválido.");return;}
 if(String(doc["mode"]|"")!="wifi"){Serial.println("Comando esperado: mode wifi.");return;}
 String url=doc["url"]|"",token=doc["token"]|"";
 if(token.length()<30||url.length()>300||(!url.startsWith("https://")&&!(ALLOW_LAN_HTTP&&lanUrl(url)))){Serial.println("URL o token inválido.");return;}
 endpoint=url;deviceToken=token;sessionConfiguredAt=millis();Serial.println("Sesión temporal configurada; token oculto y solo en RAM.");
}
void setup(){
 Serial.begin(115200);pinMode(PAIR_BUTTON,INPUT_PULLUP);analogReadResolution(12);analogSetPinAttenuation(SENSOR_PIN,ADC_11db);
 uint64_t mac=ESP.getEfuseMac();char suffix[13];snprintf(suffix,sizeof(suffix),"%012llX",(unsigned long long)mac);deviceId="PoliMQ-"+String(suffix);
 Serial.println("\nDispositivo: "+deviceId);Serial.println("MQ-138: señal ADC cruda. NO es un glucómetro. Usa divisor resistivo y referencia de calibración.");
 WiFi.mode(WIFI_STA);WiFi.begin(WIFI_SSID,WIFI_PASSWORD);configTime(0,0,"pool.ntp.org","time.google.com");
 if(BLE_PAIR_PIN>=100000&&BLE_PAIR_PIN<=999999){
  BLEDevice::init(deviceId.c_str());BLEDevice::setEncryptionLevel(ESP_BLE_SEC_ENCRYPT_MITM);BLEDevice::setSecurityCallbacks(new Security());
  BLESecurity* security=new BLESecurity();security->setAuthenticationMode(ESP_LE_AUTH_REQ_SC_MITM_BOND);security->setCapability(ESP_IO_CAP_OUT);security->setInitEncryptionKey(ESP_BLE_ENC_KEY_MASK|ESP_BLE_ID_KEY_MASK);security->setStaticPIN(BLE_PAIR_PIN);
  BLEServer* server=BLEDevice::createServer();server->setCallbacks(new Connections());BLEService* service=server->createService(SERVICE);
  measure=service->createCharacteristic(SAMPLE,BLECharacteristic::PROPERTY_READ|BLECharacteristic::PROPERTY_NOTIFY);
  measure->setAccessPermissions(ESP_GATT_PERM_READ_ENC_MITM);BLE2902* cccd=new BLE2902();cccd->setAccessPermissions(ESP_GATT_PERM_READ_ENC_MITM|ESP_GATT_PERM_WRITE_ENC_MITM);measure->addDescriptor(cccd);service->start();
  BLEDevice::getAdvertising()->addServiceUUID(SERVICE);BLEDevice::getAdvertising()->setScanResponse(true);
  Serial.println("Pulsa BOOT para anunciar BLE durante 60 s. PIN configurado en polisalud_config.h.");
 }else Serial.println("BLE deshabilitado: configura un PIN único en polisalud_config.h.");
}
void loop(){
 bool pressed=digitalRead(PAIR_BUTTON)==LOW;
 if(pressed&&!wasPressed&&measure){pairUntil=millis()+60000;BLEDevice::startAdvertising();Serial.printf("BLE anunciado 60 s; PIN local %06u\n",BLE_PAIR_PIN);}wasPressed=pressed;
 if(pairUntil&&millis()>pairUntil){BLEDevice::stopAdvertising();pairUntil=0;}
 while(Serial.available()){
  char c=Serial.read();if(c=='\n'){command(serialLine);serialLine="";}else if(c!='\r'){
   if(serialLine.length()<700)serialLine+=c;else serialLine="";
  }
 }
 if(millis()-lastSample<1000){delay(5);return;}lastSample=millis();
 uint32_t adc=0,mv=0;for(int i=0;i<16;i++){adc+=analogRead(SENSOR_PIN);mv+=analogReadMilliVolts(SENSOR_PIN);delay(2);}
 Packet p={sequence++,millis(),uint16_t(adc/16),uint16_t(mv/16),1};
 Serial.printf("seq=%u ADC=%u mV=%u uptime=%u\n",p.seq,p.adc,p.mv,p.uptime);
 if(bleConnected&&bleAuthenticated&&measure){measure->setValue((uint8_t*)&p,sizeof(p));measure->notify();}
 sendHttp(p);
}
