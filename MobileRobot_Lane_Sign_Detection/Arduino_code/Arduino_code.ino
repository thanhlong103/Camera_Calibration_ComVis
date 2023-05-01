//**********Include libraries**********//
#include "Motor_Driver.h"

//**********6 Channels L298N Motor Driver**********//
#define ENA A0
#define ENB A1
#define IN1 4
#define IN2       
#define IN3 6
#define IN4 7


//------Declare variables for calculatePID------//
float P, I, D, PIDval;
const float Kp = 40;
const float Ki = 0;
const float Kd = 45;
const int init_speed = 150;
const int min_speed = 0;
const int max_speed = 220;
const int turn_speed = 170;

//------Declare variables for directions------//
int delay_90deg = 1000;
int right = 0;
int left = 1;
int state;
float error, prev_error;

//------DECLARE CLASSES FOR INCLUDED LIBRARIES------//
Motor_Driver driver = Motor_Driver(ENA, ENB, IN1, IN2, IN3, IN4);


//---------------------------VOID SETUP---------------------------//
void setup() {

  Serial.begin(9600);

  //------PINMODE------//
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);

  //------VARIABLES------//
  P = I = D = PIDval = 0;

  //-----Connect to Raspberry Pi-----//
  // while (!Serial);
  // Serial.println("Opencv Lane Detect Autonomous Robot")
}


//---------------------------FUNCTIONS---------------------------//
void setPIDspeed()
{
  P = error;
  I = I + error;
  D = error - prev_error;

  PIDval = (Kp * P) + (Ki * I) + (Kd * D);
  prev_error = error;

  int left_speed = init_speed - PIDval; 
  int right_speed = init_speed + PIDval;

  // The motor speed should not exceed the max PWM value
  left_speed = constrain(left_speed, min_speed, max_speed);
  right_speed = constrain(right_speed, min_speed, max_speed);

  driver.setSpeed(left_speed, right_speed);
}

//---------------------------VOID LOOP---------------------------//
void loop() {
  if (Serial.available() > 0)
  {
    state = Serial.read();
    Serial.print("State: ");
    Serial.println(state);

    switch(state)
    {
      case 'a':
        error = 0;      
        setPIDspeed();
        driver.forward();
        delay(50);
        driver.stopCar();
        prev_error = error;
        break;

      case 'b':
        error = -2;
        setPIDspeed();
        driver.forward();
        delay(50);
        driver.stopCar();
        prev_error = error;
        break;
      
      case 'c':
        error = -4;
        setPIDspeed();
        driver.forward();
        delay(50);
        driver.stopCar();
        prev_error = error;
        break;

      case 'd':
        error = 2;
        setPIDspeed();
        driver.forward();
        delay(50);
        driver.stopCar();
        prev_error = error;
        break;

      case 'e':
        error = 4;
        setPIDspeed();
        driver.forward();
        delay(50);
        driver.stopCar();
        prev_error = error;
        break;
        
      case 'f':
        driver.setSpeed(turn_speed, turn_speed);
        driver.turnRight();
        delay(100);
        driver.stopCar();
        delay(50);
        prev_error = error;
        break;
      
      case 'g':
        driver.setSpeed(turn_speed, turn_speed);
        driver.turnLeft();
        delay(100);
        driver.stopCar();
        delay(50);
        prev_error = error;
        break;

      case 'h':
        driver.setSpeed(turn_speed, turn_speed);
        driver.turnLeft();
        delay(100);
        driver.stopCar();
        delay(50);
        prev_error = error;
        break;
        
      case 'i':
        driver.setSpeed(turn_speed, turn_speed);
        driver.turnRight();
        delay(100);
        driver.stopCar();
        delay(50);
        prev_error = error;
        break;     

      case 'j':
        driver.stopCar();
        delay(5000);
        prev_error = error;
        break;    

      case 'k':
        driver.turn_90deg(right, delay_90deg);
        prev_error = error;
        break;
      
      case 'l':
        driver.turn_90deg(left, delay_90deg);
        prev_error = error;
        break;
    }
  }
}





  