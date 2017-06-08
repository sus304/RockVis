# -*- coding: utf-8 -*-
import numpy as np

def DCM_ECI2ECEF(t):
  # param t : time [sec]
  omega = 7.292115e-5 # Earth Angular Velocity on Equator [rad/s]
  theta = omega * t # [rad]

  DCM_ECI2ECEF = np.array([[np.cos(theta), np.sin(theta), 0.0], 
                          [-np.sin(theta), np.cos(theta), 0.0], 
                          [0.0           , 0.0          , 1.0]])
  return DCM_ECI2ECEF

def DCM_ECEF2NED(lat, lon):

  DCM_ECEF2NED = np.array([[-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat) ],
                           [-np.sin(lon)              , np.cos(lon)               , 0.0         ],
                           [-np.cos(lat) * np.cos(lon), -np.cos(lat) * np.sin(lon), -np.sin(lat)]])  
  return DCM_ECEF2NED

def vector_ECI2ECEF_NEDframe(DCM_ECI2NED, vector_ECI, pos_ECI):
  # ECIの速度をECEF(NED frame)へ変換
  omega = 7.292115e-5 # Earth Angular Velocity on Equator [rad/s]
  tensor_ECI2ECEF = np.array([[0.0  , -omega, 0.0],
                              [omega, 0.0   , 0.0],
                              [0.0  , 0.0   , 0.0]])
  vector_ECEF_NEDframe = DCM_ECI2NED.dot(vector_ECI - tensor_ECI2ECEF.dot(pos_ECI))
  return vector_ECEF_NEDframe

def vector_NED2ECI(DCM_NED2ECI, vector_NED, pos_ECI):
  # NEDの速度をECIへ変換
  omega = 7.292115e-5 # Earth Angular Velocity on Equator [rad/s]
  tensor_ECI2ECEF = np.array([[0.0  , -omega, 0.0],
                              [omega, 0.0   , 0.0],
                              [0.0  , 0.0   , 0.0]])
  return DCM_NED2ECI.dot(vector_NED) + tensor_ECI2ECEF.dot(pos_ECI)

def ECEF2LLH(vector_ECEF):
  # ECEF座標から緯度経度高度に変換
  # Latitude-Longitude-Height
  # Vector_ECEF : [x, y, z]
  
  # WGS84 Constant
  a = 6378137.0
  f = 1.0 / 298.257223563
  b = a * (1.0 - f)
  e_sq = 2.0 * f - (f * f)
  e2_sq = (e_sq * a * a) / (b * b)
  
  p = np.sqrt(np.power(vector_ECEF[0], 2) + np.power(vector_ECEF[1], 2))
  theta = np.arctan2(vector_ECEF[2] * a, p * b)
  vector_LLH = np.zeros(3)
  vector_LLH[0] = np.degrees(np.arctan2(vector_ECEF[2] + e2_sq * b * np.power(np.sin(theta), 3),p - e_sq * a * np.power(np.cos(theta), 3)))
  vector_LLH[1] = np.degrees(np.arctan2(vector_ECEF[1], vector_ECEF[0]))
  N = a / np.sqrt(1.0 - e_sq * np.power(np.sin(np.radians(vector_LLH[0])), 2))
  vector_LLH[2] = (p / np.cos(np.radians(vector_LLH[0]))) - N

  return vector_LLH

def LLH2ECEF(vector_LLH):
  # Vector_LLH : [latitude, longitude, height] = [deg, deg, m]
  
  # WGS84 Constant
  a = 6378137.0
  f = 1.0 / 298.257223563
  e_sq = 2.0 * f - (f * f)
  N = a / np.sqrt(1.0 - e_sq * np.power(np.sin(np.radians(vector_LLH[0])), 2))
  vector_ECEF = np.zeros(3)  
  vector_ECEF[0] = (N + vector_LLH[2]) * np.cos(np.radians(vector_LLH[0])) * np.cos(np.radians(vector_LLH[1]))
  vector_ECEF[1] = (N + vector_LLH[2]) * np.cos(np.radians(vector_LLH[0])) * np.sin(np.radians(vector_LLH[1]))
  vector_ECEF[2] = (N * (1.0 - e_sq) + vector_LLH[2]) * np.sin(np.radians(vector_LLH[0]))
  
  return vector_ECEF

def DCM_body2air(alpha, beta):  
  DCM_body2air = np.array([[np.cos(alpha) * np.cos(beta) , np.sin(beta), np.sin(alpha) * np.cos(beta) ],
                           [-np.cos(alpha) * np.sin(beta), np.cos(beta), -np.sin(alpha) * np.sin(beta)],
                           [-np.sin(alpha)               , 0.0         , np.cos(alpha)                ]])
  return DCM_body2air

def DCM_NED2body_euler(azimuth, elevation, roll):
  DCM_NED2body_euler = np.array([[np.cos(azimuth) * np.cos(elevation), np.sin(azimuth) * np.cos(elevation), -np.sin(elevation)],
                                 [np.cos(azimuth) * np.sin(elevation) * np.sin(roll) - np.sin(azimuth) * np.cos(roll), np.sin(azimuth) * np.sin(elevation) * np.sin(roll) + np.cos(azimuth) * np.cos(roll), np.cos(elevation) * np.sin(roll)],
                                 [np.cos(azimuth) * np.sin(elevation) * np.cos(roll) + np.sin(azimuth) * np.sin(roll), np.sin(azimuth) * np.sin(elevation) * np.cos(roll) - np.cos(azimuth) * np.sin(roll), np.cos(elevation) * np.cos(roll)]])
  return DCM_NED2body_euler

def quat_normalize(quat):
  norm = np.linalg.norm(quat)
  quat = quat / norm
  return quat

def DCM_NED2body_quat(quat):
  q0 = quat[0]
  q1 = quat[1]
  q2 = quat[2]
  q3 = quat[3]

  DCM_NED2body_quat = np.array([[q0*q0 + q1*q1 - q2*q2 - q3*q3, 2.0 * (q1 * q2 + q0 * q3)    , 2.0 * (q1 * q3 - q0 * q2)],
                                [2.0 * (q1 * q2 - q0 * q3)    , q0*q0 - q1*q1 + q2*q2 - q3*q3, 2.0 * (q2 * q3 + q0 * q1)],
                                [2.0 * (q1 * q3 + q0 * q2)    , 2.0 * (q2 * q3 - q0 * q1)    , q0*q0 - q1*q1 - q2*q2 + q3*q3]])
  return DCM_NED2body_quat

def euler2quat(azimuth, elevation, roll):
  azimuth2 = np.radians(azimuth * 0.5)
  elevation2 = np.radians(elevation * 0.5)
  roll2 = np.radians(roll * 0.5)

  q0 = np.cos(azimuth2) * np.cos(elevation2) * np.cos(roll2) + np.sin(azimuth2) * np.sin(elevation2) * np.sin(roll2)
  q1 = np.cos(azimuth2) * np.cos(elevation2) * np.sin(roll2) - np.sin(azimuth2) * np.sin(elevation2) * np.cos(roll2)
  q2 = np.cos(azimuth2) * np.sin(elevation2) * np.cos(roll2) + np.sin(azimuth2) * np.cos(elevation2) * np.sin(roll2)
  q3 = np.sin(azimuth2) * np.cos(elevation2) * np.cos(roll2) - np.cos(azimuth2) * np.sin(elevation2) * np.sin(roll2)

  quat = np.array([q0, q1, q2, q3])
  quat = quat_normalize(quat)

  return quat

def quat2euler(DCM_NED2body_quat):
  azimuth = np.arctan2(DCM_NED2body_quat[0,1], DCM_NED2body_quat[0,0])
  elevation = np.arcsin(-DCM_NED2body_quat[0,2])
  # elevation = np.arctan2(-DCM_NED2body_quat[0,2], np.sqrt(DCM_NED2body_quat[1,2]**2 + DCM_NED2body_quat[2,2]**2))
  roll = np.arctan2(DCM_NED2body_quat[1,2], DCM_NED2body_quat[2,2])
  
  return np.array((azimuth, elevation, roll))