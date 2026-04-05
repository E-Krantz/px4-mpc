############################################################################
#
#   Copyright (C) 2024 PX4 Development Team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
############################################################################

from acados_template import AcadosModel
import casadi as ca
import numpy as np
import px4_mpc.utils.rotations as R

class SpacecraftWrenchCWModel():
    def __init__(self, orbital_period=90.0):
        self.name = 'spacecraft_wrench_cw_model'

        # constants
        self.mass = 17.8
        self.inertia = np.diag([0.315]*3)
        self.max_thrust = 2 * 1.5 * 2/3
        self.max_torque = 4 * 0.12 * 1.5 * 1/3

        orbital_n = 2.0 * np.pi / (orbital_period * 60.0)

        self.cw_A_p = np.array([[3*orbital_n**2, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]])
        self.cw_A_v = np.array([[0, 2*orbital_n, 0],
                              [-2*orbital_n, 0, 0],
                              [0, 0, 0]])

        self.create_model()

    def create_model(self):
        # set up states & controls
        p      = ca.MX.sym('p', 3)
        v      = ca.MX.sym('v', 3)
        q      = ca.MX.sym('q', 4)
        w      = ca.MX.sym('w', 3)

        self.x = ca.vertcat(p, v, q, w)
        self.u = ca.MX.sym('u', 6)

        F = self.u[0:3]
        tau = self.u[3:6]

        # xdot
        p_dot      = ca.MX.sym('p_dot', 3)
        v_dot      = ca.MX.sym('v_dot', 3)
        q_dot      = ca.MX.sym('q_dot', 4)
        w_dot      = ca.MX.sym('w_dot', 3)

        self.xdot = ca.vertcat(p_dot, v_dot, q_dot, w_dot)

        # dynamics
        self.f_expl = ca.vertcat(v,
                                 R.v_dot_q_cs(F, q) / self.mass + self.cw_A_p @ p + self.cw_A_v @ v,
                                 1.0 / 2 * ca.mtimes(R.skew_symmetric_cs(w), q),
                                 ca.inv(self.inertia) @ (tau - ca.cross(w, self.inertia @ w))
                                 )
        self.dynamics = ca.Function('f', [self.x, self.u], [self.f_expl])
    
    def get_acados_model(self) -> AcadosModel:
        model = AcadosModel()
        f_impl = self.xdot - self.f_expl

        model.f_impl_expr = f_impl
        model.f_expl_expr = self.f_expl
        model.x = self.x
        model.xdot = self.xdot
        model.u = self.u
        model.name = self.name
        return model







    