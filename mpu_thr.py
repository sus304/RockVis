# -*- coding: utf-8 -*-
import numpy as np
import RockVis as vis

def gyro_convert(digit):
  rps = np.radians(digit / 65.5) # [rad/s]
  return rps
def acc_convert(digit):
  acc = digit / 2100.0 * 9.80665 # [m/s2]
  return acc
def press_convert(digit, digit_init):
  Pa = 102.28 - ((digit - digit_init) * 4.148) # [Pa]
  return Pa
def temp_convert(digit):
  degC = ((digit / 4096.0) * 5000.0 * 0.5 - 424.0) / 6.25 # [degC]
  K = degC + 273.15
  return K


################## User Input ###################
logfile = 'test_log.csv'
frequency = 100 # [Hz]
elevation = 78 # [deg]
azimuth = 18 # [deg]
latitude = 40.0 # [deg]
longitude = 140.0 # [deg]
length_rail = 5.0 # [m]
file_row = {'temp':0, 'pressure':1, 'acc_axis':4, 'acc_lug':3, 'acc_side':2}
#################################################  

flightpath = vis.FlightPath(frequency, elevation, azimuth, 0.0, latitude, longitude, length_rail)

all_log_original = np.loadtxt(logfile, delimiter=',')
Tair_log = temp_convert(all_log_original[:, file_row['temp']]) # [K]
Pair_log = press_convert(all_log_original[:, file_row['pressure']], all_log_original[0, 1]) # [Pa]
acc_x_log = acc_convert(all_log_original[:, file_row['acc_axis']]) # [m/s2]
acc_y_log = acc_convert(all_log_original[:, file_row['acc_side']]) # [m/s2]
acc_z_log = acc_convert(all_log_original[:, file_row['acc_lug']]) # [m/s2]
gyro_x_log = gyro_convert(all_log_original[:, 3+file_row['acc_axis']])  # [rad/s]
gyro_y_log = gyro_convert(all_log_original[:, 3+file_row['acc_side']]) # [rad/s]
gyro_z_log = gyro_convert(all_log_original[:, 3+file_row['acc_lug']])  # [rad/s]

flightpath.parse_flight(Tair_log, Pair_log, acc_x_log, acc_y_log, acc_z_log, gyro_x_log, gyro_y_log, gyro_z_log)
flightpath.analysis()
flightpath.plot()

