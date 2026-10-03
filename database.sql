CREATE DATABASE maintainx_ai;
USE maintainx_ai;
CREATE TABLE users(
id INT AUTO_INCREMENT PRIMARY KEY,
name VARCHAR(100),
email VARCHAR(100) UNIQUE,
password VARCHAR(255),
role VARCHAR(20) DEFAULT 'user',
status VARCHAR(20) DEFAULT 'active'
);
CREATE TABLE machines(
machineID INT PRIMARY KEY,
model VARCHAR(50),
age INT
);
CREATE TABLE telemetry(
id INT AUTO_INCREMENT PRIMARY KEY,
machineID INT,
datetime DATETIME,
volt FLOAT,
rotate FLOAT,
pressure FLOAT,
vibration FLOAT
);
CREATE TABLE errors(
id INT AUTO_INCREMENT PRIMARY KEY,
machineID INT,
datetime DATETIME,
errorID VARCHAR(20)
);
CREATE TABLE maint(
id INT AUTO_INCREMENT PRIMARY KEY,
machineID INT,
datetime DATETIME,
comp VARCHAR(50)
);
CREATE TABLE prediction_history(
id INT AUTO_INCREMENT PRIMARY KEY,
user_id INT,
machineID INT,
health_score FLOAT,
risk_level VARCHAR(50),
failure_probability FLOAT,
prediction_time DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE maintenance_schedule(
id INT AUTO_INCREMENT PRIMARY KEY,
machineID INT,
date DATE,
engineer VARCHAR(100)
);