# -*- coding: utf-8 -*-
'''
MIT License
Copyright (c) 2017 Susumu Tanaka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURposE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

import coordinate as coord
import environment as env

class FlightPath:
  def __init__(self, freq, elevation, azimuth, roll, latitude, longitude, l_lancher):
    self.freq = freq # [Hz]
    self.dt = 1.0 / freq # [s]
    self.elevation_init = np.radians(elevation)
    self.azimuth_init = np.radians(azimuth)
    self.roll_init = np.radians(roll)
    latitude_init = latitude
    longitude_init = longitude
    self.pos_LLH_init = np.array([latitude_init, longitude_init, 10.0])
    self.l_lancher = l_lancher
    
    self.pos_ECEF_init = coord.LLH2ECEF(self.pos_LLH_init)
    self.quat_init = coord.euler2quat(np.degrees(self.azimuth_init), np.degrees(self.elevation_init), np.degrees(self.roll_init))

  def seek_liftoff(self, acc_axis_log):
    # liftoff_acc = -100000.5# * 9.80665 # [m/s^2] わりとてきとうでよい
    liftoff_acc = 1.1 * 9.80665 # [m/s^2] わりとてきとうでよい
    index_liftoff = (acc_axis_log > liftoff_acc).argmax()# - self.freq
    return index_liftoff
    
  def parse_flight(self, Tair, Pair, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z):
    index_liftoff = self.seek_liftoff(acc_x)
    self.Tair_log = Tair[index_liftoff:]
    self.Pair_log = Pair[index_liftoff:]
    self.acc_body_log = acc_x[index_liftoff:]
    self.acc_body_log = np.c_[self.acc_body_log, acc_y[index_liftoff:]]
    self.acc_body_log = np.c_[self.acc_body_log, acc_z[index_liftoff:]]
    self.gyro_body_log = gyro_x[index_liftoff:]
    self.gyro_body_log = np.c_[self.gyro_body_log, gyro_y[index_liftoff:]]
    self.gyro_body_log = np.c_[self.gyro_body_log, gyro_z[index_liftoff:]]

    self.log_length = len(self.Pair_log)
  
  def analysis(self):
    # def kinematic_equation(quat, gyro_body):
    #   # 座標系の回転Ver. for クォータニオン計算便利ノート
    #   p = gyro_body[0]
    #   q = gyro_body[1]
    #   r = gyro_body[2]
    #   tensor = np.array([[0.0, -p , -q , -r ],
    #                     [p  , 0.0, r  , -q ],
    #                     [q  , -r , 0.0, p  ],
    #                     [r  , q  , -p , 0.0]])
    #   return 0.5 * tensor.dot(quat)
    # def kinematic_equation(quat, gyro_body):
    #   # 機体座標から慣性座標（地上とは書いてる）Ver. for MATLABによるクォータニオン&クォータニオンによるキネマティクス表示
    #   p = gyro_body[0]
    #   q = gyro_body[1]
    #   r = gyro_body[2]
    #   tensor = np.array([[0.0, r  , -q , p  ],
    #                      [-r , 0.0, p  , q  ],
    #                      [q  , -p , 0.0, r  ],
    #                      [-p , -q , -r , 0.0]])
    #   return 0.5 * tensor.dot(quat)
    def kinematic_equation(quat, gyro_body):
      # 位置ベクトルの回転Ver. for クォータニオン計算便利ノート
      p = gyro_body[0]
      q = gyro_body[1]
      r = gyro_body[2]
      tensor = np.array([[0.0, -p , -q , -r ],
                         [p  , 0.0, -r , q  ],
                         [q  , r  , 0.0, -p ],
                         [r  , -q , p  , 0.0]])
      return 0.5 * tensor.dot(quat)
    
    def delta(array_new, array_pre, dt):
      return 0.5 * (array_new + array_pre) * dt

    self.vel_body_log = np.empty((self.log_length, 3))
    self.dynamicpressure_log = np.empty((self.log_length))
    self.Mach_log = np.empty((self.log_length))
    self.acc_NED_log = np.empty((self.log_length, 3))
    self.vel_NED_log = np.empty((self.log_length, 3))
    self.pos_NED_log = np.empty((self.log_length, 3))
    self.pos_ECEF_log = np.empty((self.log_length, 3))
    self.pos_LLH_log = np.empty((self.log_length, 3))

    self.dquat_log = np.empty((self.log_length, 4))
    self.quat_log = np.empty((self.log_length, 4))
    self.attitude_rad_log = np.empty((self.log_length, 3))

    self.vel_body_log[0, :] = np.array((0.0, 0.0, 0.0))
    self.dynamicpressure_log[0] = 0.0
    self.Mach_log[0] = 0.0
    self.acc_NED_log[0, :] = np.array((0.0, 0.0, 0.0))
    self.vel_NED_log[0, :] = np.array((0.0, 0.0, 0.0))
    self.pos_NED_log[0, :] = np.array((0.0, 0.0, 0.0))
    self.pos_ECEF_log[0, :] = self.pos_ECEF_init
    self.pos_LLH_log[0, :] = self.pos_LLH_init

    self.dquat_log[0, :] = np.array((0.0, 0.0, 0.0, 0.0))
    self.quat_log[0, :] = self.quat_init
    self.attitude_rad_log[0, :] = np.array((self.azimuth_init, self.elevation_init, self.roll_init))

    for i in range(1, self.log_length):
      # altitude = self.pos_LLH_log[i-1, 2]
      altitude = -self.pos_NED_log[i-1, 2]
      DCM_ECEF2NED = coord.DCM_ECEF2NED(self.pos_LLH_log[i-1, 0], self.pos_LLH_log[i-1, 1])
      DCM_NED2ECEF = DCM_ECEF2NED.transpose()
      DCM_NED2body = coord.DCM_NED2body_quat(self.quat_log[i-1, :])
      DCM_body2NED = DCM_NED2body.transpose()
      self.attitude_rad_log[i, :] = coord.quat2euler(DCM_NED2body)

      g0 = 9.80665
      g = np.array([0.0, 0.0, env.gravity(altitude)])
      Tair, Pair, rho, Cs = env.std_atmo(altitude)
      
      # transration
      self.vel_body_log[i, :] = self.vel_body_log[i-1, :] + delta(self.acc_body_log[i, :], self.acc_body_log[i-1, :], self.dt)
      self.dynamicpressure_log[i] = 0.5 * rho * np.linalg.norm(self.vel_body_log[i, :]) ** 2
      self.Mach_log[i] = np.linalg.norm(self.vel_body_log[i, :]) / Cs
      self.acc_NED_log[i, :] = DCM_body2NED.dot(self.acc_body_log[i, :]) + g
      self.vel_NED_log[i, :] = self.vel_NED_log[i-1, :] + delta(self.acc_NED_log[i, :], self.acc_NED_log[i-1, :], self.dt)
      self.pos_NED_log[i, :] = self.pos_NED_log[i-1, :] + delta(self.vel_NED_log[i, :], self.vel_NED_log[i-1, :], self.dt)
      self.pos_ECEF_log[i, :] = DCM_NED2ECEF.dot(self.pos_NED_log[i, :])
      self.pos_LLH_log[i, :] = coord.ECEF2LLH(self.pos_ECEF_log[i, :])

      # rotation
      self.dquat_log[i, :] = kinematic_equation(self.quat_log[i-1, :], self.gyro_body_log[i, :])
      self.quat_log[i, :] = self.quat_log[i-1, :] + delta(self.dquat_log[i, :], self.dquat_log[i-1, :], self.dt)
      # self.quat_log[i, :] = self.quat_log[i-1, :] + self.dquat_log[i, :] * self.dt
      self.quat_log[i, :] = coord.quat_normalize(self.quat_log[i, :])

      # Liftoff = 0.0 sec
      # self.index_liftoff
      # index_liftoff = (acc_axis_log > liftoff_acc).argmax() - self.freq
      
      # print(self.acc_body_log[i-1, :], i-1, 'acc_body')
      # print(self.acc_NED_log[i-1, :], i-1, 'acc_NED')
      # print(self.vel_NED_log[i-1, :], i-1, 'vel')
      # print(self.pos_NED_log[i-1, :], i-1, 'pos_NED')
      # print(self.pos_ECEF_log[i-1, :], i-1, 'pos_ECEF')
      # print(self.pos_LLH_log[i-1, :], i-1, 'pos_LLH')
      # print(altitude, g)
      # input()

  
  def plot(self):
    plt.figure(0)
    plt.plot(self.pos_NED_log[:, 0], label='North')
    plt.plot(self.pos_NED_log[:, 1], label='East')
    plt.plot(self.pos_NED_log[:, 2], label='Down')
    plt.ylabel('pos_NED')
    plt.grid()
    plt.legend()

    plt.figure(1)
    plt.plot(np.degrees(self.attitude_rad_log[:, 0]), label='azimuth')
    plt.plot(np.degrees(self.attitude_rad_log[:, 1]), label='elevation')
    plt.plot(np.degrees(self.attitude_rad_log[:, 2]), label='roll')
    plt.ylabel('attitude')    
    plt.grid()
    plt.legend()

    plt.figure(2)
    plt.plot(self.gyro_body_log[:, 0], label='roll')
    plt.plot(self.gyro_body_log[:, 1], label='pitch')
    plt.plot(self.gyro_body_log[:, 2], label='yaw')
    plt.ylabel('gyro')    
    plt.grid()
    plt.legend()

    plt.figure(3)
    plt.plot(self.acc_body_log[:, 0], label='axis')
    plt.plot(self.acc_body_log[:, 1], label='side')
    plt.plot(self.acc_body_log[:, 2], label='lug')
    plt.ylabel('acc_body')
    plt.grid()
    plt.legend()

    plt.figure(4)
    plt.plot(self.acc_NED_log[:, 0], label='North')
    plt.plot(self.acc_NED_log[:, 1], label='East')
    plt.plot(self.acc_NED_log[:, 2], label='Down')
    plt.ylabel('acc_NED')
    plt.grid()
    plt.legend()

    plt.show()
    
    

class Engine:
  def __init__(self, Mox, Mfb, Mfa):
    f = 0.0






def kml_make(name,Launch_LLH):
  Log = np.loadtxt('position_log.csv',delimiter=",",skiprows=1)
  array = Log[:,1]
  array_len = len(array)
  print(":")
  position_ENU = np.zeros((array_len,3))
  position_ENU[:,0] = np.array(Log[:,0])
  position_ENU[:,1] = np.array(Log[:,1])
  position_ENU[:,2] = np.array(Log[:,2])
  
  position_ecef = np.zeros((array_len,3))
  position_LLH = np.zeros((array_len,3))
  print(":")
  for i in range(array_len):
    position_ecef[i,:] = ENU2ECEF(position_ENU[i,:],Launch_LLH)
    position_LLH[i,:] = ECEF2LLH(position_ecef[i,:])
  print(":")
  
  header = 'Latitude,Longitude,Height'
  np.savetxt("Result Log 1.csv",position_LLH,fmt = '%.5f',delimiter = ',',header = header)
  
  kml = simplekml.Kml(open=1)
  Log_LLH = []
  for i in range(array_len):
    if 0 == i % 10000:
      Log_LLH.append((position_LLH[i,1],position_LLH[i,0],position_LLH[i,2]))
  print(":")
  line = kml.newlinestring(name = name)
  line.style.linestyle.width = 5
  line.style.linestyle.color = simplekml.Color.red
  line.extrude = 1
  line.altitudemode = simplekml.AltitudeMode.absolute
  line.coords = Log_LLH
  line.style.linestyle.colormode = simplekml.ColorMode.random
  kml.save(name + ".kml")