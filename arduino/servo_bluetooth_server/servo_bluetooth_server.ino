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
int pulse = 170;
int servo_id = 0;

int all_servos[12] = {200, 360, 270, 320, 280, 300, 700, 310, 220, 460, 200};

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
  
  bool result = servoCommandParser.registerCommand("s", "uu", &cmd_servo);
  Serial.println(result);
  Serial.println("registered command: s <uint64> <uint64> ");
  Serial.println("example: s 0 150");
  bool test = servoCommandParser.registerCommand("c", "uuuuuu", &cmd_servos);
  Serial.println(test);
  Serial.println("registered command: c <uint64>*12");
  Serial.println("example: c 120 120 120 120 120 120");
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

void cmd_servo(MyCommandParser::Argument *args, char *response) {
  servo_id = args[0].asInt64;
  pulse    = args[1].asInt64;
  strlcpy(response, "success", MyCommandParser::MAX_RESPONSE_SIZE);
}

void cmd_servos(MyCommandParser::Argument *args, char *response) {
  for (int i = 0; i <= 11; i++) {
      all_servos[i] = args[i].asInt64;
  }
  strlcpy(response, "success", MyCommandParser::MAX_RESPONSE_SIZE);
}


bool checkBluetoothInput(){
  if (SerialBT.available()) {
    char line[256];
    size_t lineLength = SerialBT.readBytesUntil('\n', line, 255);
    line[lineLength - 1] = '\0'; // Yikes

    Serial.printf("\"");
    Serial.println(line);
    Serial.printf("\"");
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
  Serial.printf("Servo %d: Pulse %d\n", servo_id, pulse);
  PWM.setPWM(servo_id, 0, pulse);
  
}
