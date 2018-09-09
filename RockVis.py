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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import codecs
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import cumtrapz
import pymap3d as pm
import simplekml


class RocketVisualizer:
    def __init__(self, sample_freq, g0, diameter, result_name):
        self.freq = sample_freq
        self.g0 = g0
        self.d = diameter
        self.A = 0.25 * self.d ** 2 * np.pi  # [m2]
        self.result_name = result_name
        self.R_air = 287.1
        self.gamma_air = 1.4
        self.temp_slope = 6.49  # [K/km] < 11 km alt

    def search_liftoff(self, acc_body_axis_log):
        # 機軸加速度を頭から読んでthreshold_time[sec]間threshold_acc[G]が持続したindexが離床タイミング
        threshold_time = 0.2  # [sec]
        threshold_acc = 1.5  # [G]
        for i in range(len(acc_body_axis_log)):
            if np.sum(acc_body_axis_log[i:i+int(threshold_time * self.freq)] > threshold_acc * self.g0) >= int(threshold_time * self.freq):
                index_liftoff = i
                return index_liftoff

    def INS_flight_path_analysis(self, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, elv0, azi0, roll0):
        '''
        Input coordinate (strap down INS sensor)\n
        +X:Side\n
        +Y:Launch direction\n
        -Y:Launcher\n
        +Z:Altitude(Body Axis)\n
        -Z:Ground\n
        Acc[G-Abs]/Gyro[rad/s]
        '''
        coordinate = Coordinate()

        self.acc_body_x_G_log = acc_z  # [G]
        self.acc_body_y_G_log = -acc_x
        self.acc_body_z_G_log = -acc_y 

        acc_body_x_log = acc_z * self.g0  # [m/s2]
        acc_body_y_log = -acc_x * self.g0
        acc_body_z_log = -acc_y * self.g0
        gyro_body_x_log = gyro_z - np.average(gyro_z[:int(self.freq)])  # [rad/s]
        gyro_body_y_log = -gyro_x + np.average(gyro_x[:int(self.freq)])
        gyro_body_z_log = -gyro_y + np.average(gyro_y[:int(self.freq)])

        self.index_liftoff = self.search_liftoff(acc_body_x_log)
        time_log = np.linspace(-self.index_liftoff / self.freq, (len(acc_body_x_log) - self.index_liftoff) / self.freq, len(acc_body_x_log))

        plt.figure()
        plt.plot(time_log, self.acc_body_x_G_log, label='X:Body Axis')
        plt.plot(time_log, self.acc_body_y_G_log, label='Y:Body Side')
        plt.plot(time_log, self.acc_body_z_G_log, label='Z:Body Side')
        plt.xlabel('Time [s]')
        plt.ylabel('Acceleration Body [G]')
        plt.grid()
        plt.legend()
        plt.savefig(self.result_name + '_Acc_body_G.png')
        plt.figure()
        plt.plot(time_log, acc_body_x_log, label='X:Body Axis')
        plt.plot(time_log, acc_body_y_log, label='Y:Body Side')
        plt.plot(time_log, acc_body_z_log, label='Z:Body Side')
        plt.xlabel('Time [s]')
        plt.ylabel('Acceleration Body [m/s2]')
        plt.grid()
        plt.legend()
        plt.savefig(self.result_name + '_Acc_body.png')
        plt.figure()
        plt.plot(time_log, np.degrees(gyro_body_x_log), label='X:Roll')
        plt.plot(time_log, np.degrees(gyro_body_y_log), label='Y:Pitch(initial)')
        plt.plot(time_log, np.degrees(gyro_body_z_log), label='Z:Yaw(initial)')
        plt.xlabel('Time [s]')
        plt.ylabel('Angler Velocity Body [deg/s]')
        plt.grid()
        plt.legend()
        plt.savefig(self.result_name + '_Gyro_body.png')

        self.time_log = time_log[self.index_liftoff:]
        self.acc_body_x_log = acc_body_x_log[self.index_liftoff:]
        self.acc_body_y_log = acc_body_y_log[self.index_liftoff:]
        self.acc_body_z_log = acc_body_z_log[self.index_liftoff:]
        self.acc_body_log = np.c_[self.acc_body_x_log, self.acc_body_y_log, self.acc_body_z_log]
        self.gyro_body_x_log = gyro_body_x_log[self.index_liftoff:]
        self.gyro_body_y_log = gyro_body_y_log[self.index_liftoff:]
        self.gyro_body_z_log = gyro_body_z_log[self.index_liftoff:]
        self.gyro_body_log = np.c_[self.gyro_body_x_log, self.gyro_body_y_log, self.gyro_body_z_log]

        gyro_x_polate = interpolate.interp1d(self.time_log, self.gyro_body_x_log, kind='linear', bounds_error=False, fill_value=(0.0, 0.0))
        gyro_y_polate = interpolate.interp1d(self.time_log, self.gyro_body_y_log, kind='linear', bounds_error=False, fill_value=(0.0, 0.0))
        gyro_z_polate = interpolate.interp1d(self.time_log, self.gyro_body_z_log, kind='linear', bounds_error=False, fill_value=(0.0, 0.0))

        def kinematic(quat, t):
            p = gyro_x_polate(t)
            q = gyro_y_polate(t)
            r = gyro_z_polate(t)
            # quat = coordinate.quat_normalize(quat)
            tersor_0 = [0.0, r, -q, p]
            tersor_1 = [-r, 0.0, p, q]
            tersor_2 = [q, -p, 0.0, r]
            tersor_3 = [-p, -q, -r, 0.0]
            tersor = np.array([tersor_0, tersor_1, tersor_2, tersor_3])
            quatdot = 0.5 * tersor.dot(quat)
            return quatdot

        quat_init = coordinate.euler2quat(azi0, elv0, roll0)
        self.quat_log = odeint(kinematic, quat_init, self.time_log)
        DCM_ENU2Body_log = np.array(list(map(coordinate.DCM_ENU2Body_quat, self.quat_log)))
        DCM_Body2ENU_log = np.array([DCM.transpose() for DCM in DCM_ENU2Body_log])
        self.attitude_log = np.array([coordinate.quat2euler(DCM) for DCM in DCM_ENU2Body_log])  # [deg]

        self.gravity_body_x_log = DCM_ENU2Body_log.dot([0.0, 0.0, self.g0])[:, 0]

        self.vel_body_x_log = cumtrapz(self.acc_body_x_log, self.time_log, initial=0.0)
        self.vel_body_y_log = cumtrapz(self.acc_body_y_log, self.time_log, initial=0.0)
        self.vel_body_z_log = cumtrapz(self.acc_body_z_log, self.time_log, initial=0.0)
        self.vel_body_log = np.c_[self.vel_body_x_log, self.vel_body_y_log, self.vel_body_z_log]

        self.acc_ENU_log = np.array([DCM.dot(acc_body) for DCM, acc_body in zip(DCM_Body2ENU_log, np.array(self.acc_body_log))])
        self.acc_ENU_x_log = self.acc_ENU_log[:, 0]
        self.acc_ENU_y_log = self.acc_ENU_log[:, 1]
        self.acc_ENU_z_log = self.acc_ENU_log[:, 2] - self.g0  # ToDo:重力加速度可変?

        self.vel_ENU_x_log = cumtrapz(self.acc_ENU_x_log, self.time_log, initial=0.0)
        self.vel_ENU_y_log = cumtrapz(self.acc_ENU_y_log, self.time_log, initial=0.0)
        self.vel_ENU_z_log = cumtrapz(self.acc_ENU_z_log, self.time_log, initial=0.0)
        self.vel_ENU_log = np.c_[self.vel_ENU_x_log, self.vel_ENU_y_log, self.vel_ENU_z_log]

        self.pos_ENU_x_log = cumtrapz(self.vel_ENU_x_log, self.time_log, initial=0.0)
        self.pos_ENU_y_log = cumtrapz(self.vel_ENU_y_log, self.time_log, initial=0.0)
        self.pos_ENU_z_log = cumtrapz(self.vel_ENU_z_log, self.time_log, initial=0.0)
        self.pos_ENU_log = np.c_[self.pos_ENU_x_log, self.pos_ENU_y_log, self.pos_ENU_z_log]

        self.index_landing = int(20.0 * self.freq)
        # self.index_landing = len(self.pos_ENU_z_log)

        output_array = np.c_[self.time_log, self.acc_body_log, self.vel_body_log, self.acc_ENU_log, self.vel_ENU_log, self.pos_ENU_log, np.rad2deg(self.gyro_body_log), self.quat_log, self.attitude_log]
        header = 'time[s]' ',acc_body_axial[m/s2],acc_body_side[m/s2],acc_body_upper[m/s2]' \
                ',vel_body_axial[m/s],vel_body_side[m/s],vel_body_upper[m/s]'\
                ',acc_ENU_East[m/s2],acc_ENU_North[m/s2],acc_ENU_Up[m/s2]'\
                ',vel_ENU_East[m/s],vel_ENU_North[m/s],vel_ENU_Up[m/s]'\
                ',pos_ENU_East[m],pos_ENU_North[m],pos_ENU_Up[m]'\
                ',gyro_body_axial[deg/s],gyro_body_side[deg/s],gyro_body_upper[deg/s]'\
                ',quatrnion1[-],quatrnion2[-],quatrnion3[-],quatrnion4[-]'\
                ',roll[deg],elevation[deg],yaw[deg]'
        np.savetxt(self.result_name + '_flight_log.csv', output_array, delimiter=',', fmt='%0.5f', header=header, comments='')

        plt.figure()
        plt.plot(self.time_log[:self.index_landing], self.vel_body_x_log[:self.index_landing], label='X:Body Axis')
        plt.plot(self.time_log[:self.index_landing], self.vel_body_y_log[:self.index_landing], label='Y:Body Side')
        plt.plot(self.time_log[:self.index_landing], self.vel_body_z_log[:self.index_landing], label='Z:Body Upper')
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity Body [m/s]')
        plt.grid()
        plt.legend()
        plt.savefig(self.result_name + '_Vel_body.png')

        plt.figure()
        plt.plot(self.time_log[:self.index_landing], self.quat_log[:self.index_landing, 0], label='p1')
        plt.plot(self.time_log[:self.index_landing], self.quat_log[:self.index_landing, 1], label='p2')
        plt.plot(self.time_log[:self.index_landing], self.quat_log[:self.index_landing, 2], label='p3')
        plt.plot(self.time_log[:self.index_landing], self.quat_log[:self.index_landing, 3], label='p4')
        plt.xlabel('Time [s]')
        plt.ylabel('Quatrnion [-]')
        plt.grid()
        plt.legend()
        plt.savefig(self.result_name + '_Quatrnion.png')

        plt.figure()
        plt.plot(self.time_log[:self.index_landing], self.attitude_log[:self.index_landing, 0], label='Azimuth')
        plt.plot(self.time_log[:self.index_landing], self.attitude_log[:self.index_landing, 1], label='Elevation')
        plt.plot(self.time_log[:self.index_landing], self.attitude_log[:self.index_landing, 2], label='Roll')
        plt.xlabel('Time [s]')
        plt.ylabel('Angle [deg]')
        plt.grid()
        plt.legend()
        plt.savefig(self.result_name + '_Aittitude.png')

        plt.figure()
        plt.plot(self.time_log[:self.index_landing], self.acc_ENU_x_log[:self.index_landing], label='X:East')
        plt.plot(self.time_log[:self.index_landing], self.acc_ENU_y_log[:self.index_landing], label='Y:North')
        plt.plot(self.time_log[:self.index_landing], self.acc_ENU_z_log[:self.index_landing], label='Z:Up')
        plt.xlabel('Time [s]')
        plt.ylabel('Acceleration ENU [m/s2]')
        plt.grid()
        plt.legend()
        plt.savefig(self.result_name + '_Acc_ENU.png')

        plt.figure()
        plt.plot(self.time_log[:self.index_landing], self.vel_ENU_x_log[:self.index_landing], label='X:East')
        plt.plot(self.time_log[:self.index_landing], self.vel_ENU_y_log[:self.index_landing], label='Y:North')
        plt.plot(self.time_log[:self.index_landing], self.vel_ENU_z_log[:self.index_landing], label='Z:Up')
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity ENU [m/s]')
        plt.grid()
        plt.legend()
        plt.savefig(self.result_name + '_Vel_ENU.png')

        plt.figure()
        plt.plot(self.time_log[:self.index_landing], self.pos_ENU_x_log[:self.index_landing], label='X:East')
        plt.plot(self.time_log[:self.index_landing], self.pos_ENU_y_log[:self.index_landing], label='Y:North')
        plt.plot(self.time_log[:self.index_landing], self.pos_ENU_z_log[:self.index_landing], label='Z:Up')
        plt.xlabel('Time [s]')
        plt.ylabel('Position ENU [m]')
        plt.grid()
        plt.legend()
        plt.savefig(self.result_name + '_Pos_ENU.png')

    def complementaly_fileter(self, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, elv0, azi0, roll0):
        '''
        Input coordinate (strap down INS sensor)\n
        +X:Side\n
        +Y:Launch direction\n
        -Y:Launcher\n
        +Z:Altitude(Body Axis)\n
        -Z:Ground\n
        Acc[G-Abs]/Gyro[rad/s]
        '''
        coordinate = Coordinate()

        self.acc_body_x_G_log = acc_z  # [G]
        self.acc_body_y_G_log = -acc_x
        self.acc_body_z_G_log = -acc_y 

        acc_body_x_log = acc_z * self.g0  # [m/s2]
        acc_body_y_log = -acc_x * self.g0
        acc_body_z_log = -acc_y * self.g0
        gyro_body_x_log = gyro_z - np.average(gyro_z[:int(self.freq)])  # [rad/s]
        gyro_body_y_log = -gyro_x + np.average(gyro_x[:int(self.freq)])
        gyro_body_z_log = -gyro_y + np.average(gyro_y[:int(self.freq)])

        self.index_liftoff = self.search_liftoff(acc_body_x_log)
        time_log = np.linspace(-self.index_liftoff / self.freq, (len(acc_body_x_log) - self.index_liftoff) / self.freq, len(acc_body_x_log))

        self.time_log = time_log[self.index_liftoff:]
        self.acc_body_x_log = acc_body_x_log[self.index_liftoff:]
        self.acc_body_y_log = acc_body_y_log[self.index_liftoff:]
        self.acc_body_z_log = acc_body_z_log[self.index_liftoff:]
        self.acc_body_log = np.c_[self.acc_body_x_log, self.acc_body_y_log, self.acc_body_z_log]
        self.gyro_body_x_log = gyro_body_x_log[self.index_liftoff:]
        self.gyro_body_y_log = gyro_body_y_log[self.index_liftoff:]
        self.gyro_body_z_log = gyro_body_z_log[self.index_liftoff:]
        self.gyro_body_log = np.c_[self.gyro_body_x_log, self.gyro_body_y_log, self.gyro_body_z_log]

        gyro_x_polate = interpolate.interp1d(self.time_log, self.gyro_body_x_log, kind='linear', bounds_error=False, fill_value=(0.0, 0.0))
        gyro_y_polate = interpolate.interp1d(self.time_log, self.gyro_body_y_log, kind='linear', bounds_error=False, fill_value=(0.0, 0.0))
        gyro_z_polate = interpolate.interp1d(self.time_log, self.gyro_body_z_log, kind='linear', bounds_error=False, fill_value=(0.0, 0.0))

        def kinematic(quat, t):
            p = gyro_x_polate(t)
            q = gyro_y_polate(t)
            r = gyro_z_polate(t)
            # quat = coordinate.quat_normalize(quat)
            tersor_0 = [0.0, r, -q, p]
            tersor_1 = [-r, 0.0, p, q]
            tersor_2 = [q, -p, 0.0, r]
            tersor_3 = [-p, -q, -r, 0.0]
            tersor = np.array([tersor_0, tersor_1, tersor_2, tersor_3])
            quatdot = 0.5 * tersor.dot(quat)
            return quatdot

        quat_init = coordinate.euler2quat(azi0, elv0, roll0)
        self.quat_log = odeint(kinematic, quat_init, self.time_log)
        DCM_ENU2Body_log = np.array(list(map(coordinate.DCM_ENU2Body_quat, self.quat_log)))
        DCM_Body2ENU_log = np.array([DCM.transpose() for DCM in DCM_ENU2Body_log])
        self.attitude_log = np.array([coordinate.quat2euler(DCM) for DCM in DCM_ENU2Body_log])  # [deg]

        self.gravity_body_x_log = DCM_ENU2Body_log.dot([0.0, 0.0, self.g0])[:, 0]

        self.vel_body_x_log = cumtrapz(self.acc_body_x_log, self.time_log, initial=0.0)
        self.vel_body_y_log = cumtrapz(self.acc_body_y_log, self.time_log, initial=0.0)
        self.vel_body_z_log = cumtrapz(self.acc_body_z_log, self.time_log, initial=0.0)
        self.vel_body_log = np.c_[self.vel_body_x_log, self.vel_body_y_log, self.vel_body_z_log]

        self.acc_ENU_log = np.array([DCM.dot(acc_body) for DCM, acc_body in zip(DCM_Body2ENU_log, np.array(self.acc_body_log))])
        self.acc_ENU_x_log = self.acc_ENU_log[:, 0]
        self.acc_ENU_y_log = self.acc_ENU_log[:, 1]
        self.acc_ENU_z_log = self.acc_ENU_log[:, 2] - self.g0  # ToDo:重力加速度可変?

        self.vel_ENU_x_log = cumtrapz(self.acc_ENU_x_log, self.time_log, initial=0.0)
        self.vel_ENU_y_log = cumtrapz(self.acc_ENU_y_log, self.time_log, initial=0.0)
        self.vel_ENU_z_log = cumtrapz(self.acc_ENU_z_log, self.time_log, initial=0.0)
        self.vel_ENU_log = np.c_[self.vel_ENU_x_log, self.vel_ENU_y_log, self.vel_ENU_z_log]

        self.pos_ENU_x_log = cumtrapz(self.vel_ENU_x_log, self.time_log, initial=0.0)
        self.pos_ENU_y_log = cumtrapz(self.vel_ENU_y_log, self.time_log, initial=0.0)
        self.pos_ENU_z_log = cumtrapz(self.vel_ENU_z_log, self.time_log, initial=0.0)
        self.pos_ENU_log = np.c_[self.pos_ENU_x_log, self.pos_ENU_y_log, self.pos_ENU_z_log]


    def extend_flight_path_earth(self, launch_point_LLH):
        lat, lon, h = pm.enu2geodetic(self.pos_ENU_log[:self.index_landing,0], self.pos_ENU_log[:self.index_landing,1], self.pos_ENU_log[:self.index_landing,2], launch_point_LLH[0], launch_point_LLH[1], launch_point_LLH[2])  # lat, lon, h
        self.pos_LLH_log = np.c_[lat, lon, h]
        ecef_x, ecef_y, ecef_z = pm.enu2ecef(self.pos_ENU_log[:self.index_landing,0], self.pos_ENU_log[:self.index_landing,1], self.pos_ENU_log[:self.index_landing,2], launch_point_LLH[0], launch_point_LLH[1], launch_point_LLH[2])
        self.pos_ECEF_log = np.c_[ecef_x, ecef_y, ecef_z]

        plt.figure()
        plt.plot(self.time_log[:self.index_landing], self.pos_ECEF_log[:self.index_landing, 0] /1e3, label='X')
        plt.plot(self.time_log[:self.index_landing], self.pos_ECEF_log[:self.index_landing, 1] /1e3, label='Y')
        plt.plot(self.time_log[:self.index_landing], self.pos_ECEF_log[:self.index_landing, 2] /1e3, label='Z')
        plt.xlabel('Time [s]')
        plt.ylabel('Position ECEF [km]')
        plt.grid()
        plt.legend()
        plt.savefig(self.result_name + '_Pos_ECEF.png')

        kml = simplekml.Kml(open=1)
        Log_LLH = []
        for i in range(len(self.pos_LLH_log[:,0])):
            if 0 == i % 10:
                Log_LLH.append([self.pos_LLH_log[i,1], self.pos_LLH_log[i,0], self.pos_LLH_log[i,2]])
        line = kml.newlinestring()
        line.style.linestyle.width = 4
        line.style.linestyle.color = simplekml.Color.red
        line.extrude = 1
        line.altitudemode = simplekml.AltitudeMode.absolute
        line.coords = Log_LLH
        line.style.linestyle.colormode = simplekml.ColorMode.random
        kml.save(self.result_name + '_trajectory.kml')


    def extend_pressure_altitude_analysis(self, Pair_log, Pair_0, Tair_0):
        '''
        Input: Pair_log[kPa], Pair_0[kPa],Tair_0[degC]
        '''
        dPair_log = Pair_log - np.average(Pair_log[:int(self.freq*0.5)])
        self.Pair_log = Pair_0 + dPair_log[self.index_liftoff:]
        self.alt_pressure_log = (((Pair_0 / self.Pair_log) ** (1.0 / 5.25607)) - 1) * (Tair_0 + 273.15) / (self.temp_slope / 1e3)

        plt.figure()
        plt.plot(self.time_log[:self.index_landing], self.alt_pressure_log[:self.index_landing])
        plt.xlabel('Time [s]')
        plt.ylabel('Altitude [m]')
        plt.grid()
        plt.savefig(self.result_name + '_PressureAlt.png')

        np.savetxt(self.result_name + '_PressureAlt_log.csv', np.c_[self.time_log, self.Pair_log, self.alt_pressure_log], delimiter=',', comments='', fmt='%0.5f', header='time[sec],air pressure[kPa],altitude[m]')

    def extend_thrust_analysis(self, mass_log, mach_array_interpolate, Cd_array_interpolate, Tair_0):
        self.Tair_log = (Tair_0 + 273.15) - self.pos_ENU_z_log * (self.temp_slope / 1e3)  # [K]
        self.rho_log = self.Pair_log * 1e3 / (self.R_air * self.Tair_log)
        self.Cs_log = np.sqrt(self.gamma_air * self.R_air * self.Tair_log)
        self.mach_log = self.vel_body_x_log / self.Cs_log

        Cd = interpolate.interp1d(mach_array_interpolate, Cd_array_interpolate, kind='linear', bounds_error=False, fill_value=(Cd_array_interpolate[0], Cd_array_interpolate[-1]))
        self.Cd_log = Cd(self.mach_log)

        self.drag_log = 0.5 * self.rho_log * self.vel_body_x_log ** 2 * self.Cd_log * self.A
        self.mg_axial_log = mass_log * self.gravity_body_x_log
        self.F_axial = self.acc_body_x_log * mass_log
        self.thrust = self.F_axial + self.drag_log - self.mg_axial_log

        output_array = np.c_[self.time_log, self.mach_log, self.Cd_log, self.drag_log, self.F_axial, self.thrust]
        header = 'time[s],mach[-],Cd[-],drag[N],axial[N],thrust[N]'
        np.savetxt(self.result_name + '_force_log.csv', output_array, delimiter=',', fmt='%0.5f', header=header, comments='')

        plt.figure()
        plt.plot(self.time_log[:self.index_landing], self.mach_log[:self.index_landing])
        plt.xlabel('Time [s]')
        plt.ylabel('Mach number [-]')
        plt.grid()
        plt.legend()
        plt.savefig(self.result_name + '_Mach.png')

        plt.figure()
        plt.plot(self.time_log[:self.index_landing], self.drag_log[:self.index_landing], label='Drag')
        plt.plot(self.time_log[:self.index_landing], self.thrust[:self.index_landing], label='Thrust')
        plt.plot(self.time_log[:self.index_landing], self.F_axial[:self.index_landing], label='Axial')
        plt.xlabel('Time [s]')
        plt.ylabel('Force [N]')
        plt.grid()
        plt.legend()
        plt.savefig(self.result_name + '_Force.png')

        plt.figure()
        plt.plot(self.time_log[:self.index_landing], self.thrust[:self.index_landing], label='Thrust')
        plt.xlabel('Time [s]')
        plt.ylabel('Thrust [N]')
        plt.grid()
        plt.legend()
        plt.savefig(self.result_name + '_Thrust.png')

class Coordinate:
    def DCM_ENU2Body_euler(self, azimuth, elevation, roll):
        '''
        Input: Azimuth[rad],Elevation[rad],Roll[rad]
        '''
        DCM_0 = [np.cos(azimuth) * np.cos(elevation), np.sin(azimuth) * np.cos(elevation), -np.sin(elevation)]
        DCM_1 = [-np.sin(azimuth) * np.cos(roll) + np.cos(azimuth) * np.sin(elevation) * np.sin(roll), np.cos(azimuth) * np.cos(roll) + np.sin(azimuth) * np.sin(elevation) * np.sin(roll), np.cos(elevation) * np.sin(roll)]
        DCM_2 = [np.sin(azimuth) * np.sin(roll) + np.cos(azimuth) * np.sin(elevation) * np.cos(roll), -np.cos(azimuth) * np.sin(roll) + np.sin(azimuth) * np.sin(elevation) * np.cos(roll), np.cos(elevation) * np.cos(roll)]
        DCM_ENU2Body_euler = np.array([DCM_0, DCM_1, DCM_2])
        return DCM_ENU2Body_euler

    def quat_normalize(self, quat):
        norm = np.linalg.norm(quat)
        quat = quat / norm
        return quat

    def DCM_ENU2Body_quat(self, quat):
        q0 = quat[0]
        q1 = quat[1]
        q2 = quat[2]
        q3 = quat[3]

        DCM_0 = [q0 * q0 - q1*q1 - q2*q2 + q3*q3, 2.0 * (q0 * q1 + q2 * q3)    , 2.0 * (q0 * q2 - q1 * q3)]
        DCM_1 = [2.0 * (q0 * q1 - q2 * q3)    , q1*q1 - q0*q0 - q2*q2 + q3*q3, 2.0 * (q1 * q2 + q0 * q3)]
        DCM_2 = [2.0 * (q0 * q2 + q1 * q3)    , 2.0 * (q1 * q2 - q0 * q3)    , q2*q2 - q0*q0 - q1*q1 + q3*q3]
        DCM_ENU2Body_quat = np.array([DCM_0, DCM_1, DCM_2])
        return DCM_ENU2Body_quat

    def euler2quat(self, azimuth, elevation, roll=0.0):
        '''
        Input: Azimuth[deg],Elevation[deg],Roll[deg]
        '''
        azimuth = np.radians(azimuth)
        elevation = np.radians(elevation)
        roll = np.radians(roll)

        DCM = self.DCM_ENU2Body_euler(azimuth, elevation, roll)
        q0 = 0.5 * np.sqrt(1.0 + DCM[0,0] - DCM[1,1] - DCM[2,2])
        q1 = 0.5 * np.sqrt(1.0 - DCM[0,0] + DCM[1,1] - DCM[2,2])
        q2 = 0.5 * np.sqrt(1.0 - DCM[0,0] - DCM[1,1] + DCM[2,2])
        q3 = 0.5 * np.sqrt(1.0 + DCM[0,0] + DCM[1,1] + DCM[2,2])

        quat_max_index = np.argmax([q0, q1, q2, q3])
        if quat_max_index == 0:
            q0 = 0.5 * np.sqrt(1.0 + DCM[0, 0] - DCM[1,1] - DCM[2,2])
            q1 = (DCM[0, 1] + DCM[1, 0]) / (4.0 * q0)
            q2 = (DCM[2, 0] + DCM[0, 2]) / (4.0 * q0)
            q3 = (DCM[1, 2] - DCM[2, 1]) / (4.0 * q0)
        elif quat_max_index == 1:
            q1 = 0.5 * np.sqrt(1.0 - DCM[0, 0] + DCM[1,1] - DCM[2,2])
            q0 = (DCM[0, 1] + DCM[1, 0]) / (4.0 * q1)
            q2 = (DCM[1, 2] + DCM[2, 1]) / (4.0 * q1)
            q3 = (DCM[2, 0] - DCM[0, 2]) / (4.0 * q1)
        elif quat_max_index == 2:
            q2 = 0.5 * np.sqrt(1.0 - DCM[0, 0] - DCM[1,1] + DCM[2,2])
            q0 = (DCM[2, 0] + DCM[0, 2]) / (4.0 * q2)
            q1 = (DCM[1, 2] + DCM[2, 1]) / (4.0 * q2)
            q3 = (DCM[0, 1] - DCM[1, 0]) / (4.0 * q2)
        elif quat_max_index == 3:
            q3 = 0.5 * np.sqrt(1.0 + DCM[0, 0] + DCM[1,1] + DCM[2,2])
            q0 = (DCM[1, 2] - DCM[2, 1]) / (4.0 * q3)
            q1 = (DCM[2, 0] - DCM[0, 2]) / (4.0 * q3)
            q2 = (DCM[0, 1] - DCM[1, 0]) / (4.0 * q3)           

        quat = np.array([q0, q1, q2, q3])
        quat = self.quat_normalize(quat)

        return quat

    def quat2euler(self, DCM_NED2Body):
        DCM = DCM_NED2Body
        azimuth = np.rad2deg(np.arctan2(DCM[0, 1], DCM[0, 0]))
        elevation = np.rad2deg(-np.arcsin(DCM[0, 2]))
        roll = np.rad2deg(np.arctan2(DCM[1, 2], DCM[2, 2]))

        return azimuth, elevation, roll

