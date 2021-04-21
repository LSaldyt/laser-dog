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

  // The following are the initial positions of the servos
  // This comes from emperical calibration.
  PWM.setPWM(1,  0, 360);
  PWM.setPWM(2,  0, 270);
  PWM.setPWM(3,  0, 320);
  PWM.setPWM(4,  0, 380);
  PWM.setPWM(5,  0, 300);
  PWM.setPWM(6,  0,  70);
  PWM.setPWM(7,  0, 310);
  PWM.setPWM(8,  0, 220);
  PWM.setPWM(9,  0, 460);
  PWM.setPWM(10, 0, 330);
  PWM.setPWM(11, 0, 200);
  
  servoCommandParser.registerCommand("s", "uu", &cmd_servo);
  Serial.println("registered command: s <uint64> <uint64> ");
  Serial.println("example: s 0 150");
  servoCommandParser.registerCommand("sleep", "", &servosSleep);
  Serial.println("registered command: sleep ");
  servoCommandParser.registerCommand("wake", "", &servosWake);
  Serial.println("registered command: wake ");
  Serial.println("");
}

// the loop function runs over and over again forever
void loop() {
  delay(100);                       // wait
  if (checkBluetoothInput())
    writePWM();
}

void cmd_servo(MyCommandParser::Argument *args, char *response) {
  servo_id = args[0].asInt64;
  pulse    = args[1].asInt64;
  strlcpy(response, "success", MyCommandParser::MAX_RESPONSE_SIZE);
}

bool checkBluetoothInput(){
  if (SerialBT.available()) {
    char line[128];
    size_t lineLength = SerialBT.readBytesUntil('\n', line, 127);
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
