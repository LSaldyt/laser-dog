#include "BluetoothSerial.h"

#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled! Please run `make menuconfig` to and enable it
#endif

#ifdef ESP32
  #include <WiFi.h>
#else
  #include <ESP8266WiFi.h>
#endif

BluetoothSerial SerialBT;

void setup() {
Serial.begin(115200);
SerialBT.begin("SpotCameraESP32_test"); //Bluetooth device name
Serial.println(WiFi.macAddress());
Serial.println("The device started, now you can pair it with bluetooth!");
}

void loop() {
if (SerialBT.available()) {
SerialBT.write(0x0a);
}
delay(20);
}
