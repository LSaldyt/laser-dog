//***********
// Servo motor test
// SDA - pin next to GND - 21
// SCL - 22
//************

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <CommandParser.h>
#include "BluetoothSerial.h"

#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled! Please run `make menuconfig` to and enable it
#endif

BluetoothSerial SerialBT;

#define PWM_SERVO_ADDR    0x40
Adafruit_PWMServoDriver PWM = Adafruit_PWMServoDriver(PWM_SERVO_ADDR);

typedef CommandParser<> MyCommandParser;
MyCommandParser servoCommandParser;
int servo_id = 0;

int leg = 0;
int shoulder = 0;
int knee = 0;
bool cmd = false;

// the setup function runs once when you press reset or power the board
void setup() {

  // 12-Bit 16 Channel PWM Module

  PWM.begin();
  PWM.setPWMFreq(50);

  Serial.begin(115200);
  Serial.println("Starting!!!");
  SerialBT.begin("SpotESP32"); //Bluetooth device name
  Serial.println("The device started, now you can pair it with bluetooth!");

  while (!Serial);
  ;
  bool test = servoCommandParser.registerCommand("bl", "uuu", &cmd_bl);
  Serial.println(test);
  test = servoCommandParser.registerCommand("br", "uuu", &cmd_br);
  Serial.println(test);
  test = servoCommandParser.registerCommand("fl", "uuu", &cmd_fl);
  Serial.println(test);
  test = servoCommandParser.registerCommand("fr", "uuu", &cmd_fr);
  Serial.println(test);
  
  servoCommandParser.registerCommand("sleep", "", &servosSleep);
  Serial.println("registered command: sleep ");
  servoCommandParser.registerCommand("wake", "", &servosWake);
  Serial.println("registered command: wake ");
  Serial.println("");
}

// the loop function runs over and over again forever
void loop() {
  if (checkBluetoothInput())
    writePWM();
}

void load_cmd(MyCommandParser::Argument *args, char *response) {
  leg      = args[0].asInt64;
  shoulder = args[1].asInt64;
  knee     = args[2].asInt64;
  
  strlcpy(response, "success", MyCommandParser::MAX_RESPONSE_SIZE);
}

void cmd_bl(MyCommandParser::Argument *args, char *response) {
  load_cmd(args, response);
  servo_id = 6;
}

void cmd_br(MyCommandParser::Argument *args, char *response) {
  load_cmd(args, response);
  servo_id = 9;
}

void cmd_fl(MyCommandParser::Argument *args, char *response) {
  load_cmd(args, response);
  servo_id = 0;
}

void cmd_fr(MyCommandParser::Argument *args, char *response) {
  load_cmd(args, response);
  servo_id = 3;
}

bool checkBluetoothInput(){
  if (SerialBT.available()) {
    char line[256];
    size_t lineLength = SerialBT.readBytesUntil('\n', line, 255);
    line[lineLength] = '\0'; // Yikes

    Serial.println(line);
    char response[MyCommandParser::MAX_RESPONSE_SIZE];
    servoCommandParser.processCommand(line, response);
    Serial.println(response);

    if (strcmp(response, "success") == 0)
      return true;
  }

  return false;
}

void servosSleep(MyCommandParser::Argument *args, char *response) {
  PWM.sleep();
}

void servosWake(MyCommandParser::Argument *args, char *response) {
  PWM.wakeup();
}

void writePWM(){
  PWM.setPWM(servo_id, 0, leg);
  PWM.setPWM(servo_id + 1, 0, shoulder);
  PWM.setPWM(servo_id + 2, 0, knee);
  
}
