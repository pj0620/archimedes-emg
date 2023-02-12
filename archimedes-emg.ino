#include <WebSocketsServer.h>
#include <WiFi.h>
#include <ESPAsyncWebServer.h>
#include <SPIFFS.h>

/*  Sampling */
#define ADC_PIN 34
#define SAMPLE_RATE 6000
#define SAMPLE_SIZE 100
const float samplePeriod = 1000.0/SAMPLE_RATE;
#define DISABLE_SAMPLING_PIN 14
int adcValue;

/* Web Monitor */
#define USE_WEB_MONITOR false
WebSocketsServer webSocket = WebSocketsServer(81);
const char *SSID = "NETGEAR02";
const char *PASSWORD = "3322207797WiffleBall";
String myString[3] = {"0", "-1", "25"}; //1st index used for ADC data, 2nd as dummy value per sample, 3rd for timing
String JSONtxt;
AsyncWebServer server(80);

int readADC(){ return analogRead(ADC_PIN); }

void updateWebMonitor(int adcValue, int idx) {
  if (!USE_WEB_MONITOR) { return; }
  
  String ADCVal = String(adcValue).c_str();

  myString[0] = ADCVal;
  JSONtxt= "{\"ADC1\":\"" + myString[0] + "\",";
  JSONtxt += "\"ADC2\":\"" + myString[1] + "\",";
  JSONtxt += "\"ADC3\":\"" + myString[2] + "\",";
  JSONtxt += "\"idx\":\"" + String(idx) + "\"}";

  webSocket.broadcastTXT(JSONtxt);
}

void buildSample() {
  int adcValue;
  for (int i=0; i < SAMPLE_SIZE; i++) {
    delay(samplePeriod);
    adcValue = readADC();
    updateWebMonitor(adcValue, i);
    Serial.println(adcValue);
  }
}

void setup()
{
  Serial.begin(115200);

  if (USE_WEB_MONITOR) {
    if (!SPIFFS.begin()){
      Serial.println("An Error has occurred while mounting SPIFFS");
      return;
    }
  
    WiFi.begin(SSID, PASSWORD);
    while (WiFi.status() != WL_CONNECTED){
      delay(1000);
      Serial.println("Connecting to WiFi..");
    }
  
    Serial.println("");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
  
    server.on("/", HTTP_GET, [](AsyncWebServerRequest *request)
              { request->send(SPIFFS, "/index.html"); });
  
    server.begin();
    webSocket.begin();
  }

  pinMode(DISABLE_SAMPLING_PIN, INPUT_PULLUP);
}

void loop(){
//  if (USE_WEB_MO NITOR) { webSocket.loop(); }
//  if (!digitalRead(DISABLE_SAMPLING_PIN)) { buildSample(); }
  if (digitalRead(DISABLE_SAMPLING_PIN)) { return; }
//  delay(samplePeriod);
  adcValue = readADC();
//  updateWebMonitor(adcValue, i);
  Serial.println(adcValue);
}
