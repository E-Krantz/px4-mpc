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
import casadi as cs
import numpy as np
import px4_mpc.utils.rotations as R

class SpacecraftPropellerModel():
    def __init__(self):
        self.name = 'spacecraft_propeller_model'

        # constants
        self.mass = 17.8
        self.inertia = np.diag([0.315]*3)
        self.min_thrust = -1.5
        self.max_thrust = 1.5
        self.torque_arm_length = 0.105

    def get_acados_model(self) -> AcadosModel:
        model = AcadosModel()

        # set up states & controls
        p      = cs.MX.sym('p', 3)
        v      = cs.MX.sym('v', 3)
        q      = cs.MX.sym('q', 4)
        w      = cs.MX.sym('w', 3)

        x = cs.vertcat(p, v, q, w)

        u = cs.MX.sym('u', 4)
        D_mat = cs.MX.zeros(2, 4)
        D_mat[0, 0] = -1
        D_mat[0, 1] = 1
        D_mat[1, 2] = -1
        D_mat[1, 3] = 1

        # L mat
        L_mat = cs.MX.zeros(1, 4)
        L_mat[0, 0] = 1
        L_mat[0, 1] = 1
        L_mat[0, 2] = 1
        L_mat[0, 3] = 1
        L_mat = L_mat * self.torque_arm_length

        F_2d = cs.mtimes(D_mat, u)
        tau_1d = cs.mtimes(L_mat, u)

        F = cs.vertcat(F_2d[0, 0], F_2d[1, 0], 0.0)
        tau = cs.vertcat(0.0, 0.0, tau_1d)

        # xdot
        p_dot      = cs.MX.sym('p_dot', 3)
        v_dot      = cs.MX.sym('v_dot', 3)
        q_dot      = cs.MX.sym('q_dot', 4)
        w_dot      = cs.MX.sym('w_dot', 3)

        xdot = cs.vertcat(p_dot, v_dot, q_dot, w_dot)

        q_normalized = q / cs.norm_2(q)
        a_thrust = R.v_dot_q_cs(F, q_normalized)/self.mass

        # dynamics with planar constraints: v_z = 0, w_x = 0, w_y = 0
        # Enforce v[2] = 0, w[0] = 0, w[1] = 0
        v_planar = cs.vertcat(v[0], v[1], 0.0)
        w_planar = cs.vertcat(0.0, 0.0, w[2])

        # Only propagate planar velocities and yaw
        a_thrust_planar = cs.vertcat(a_thrust[0], a_thrust[1], 0.0)
        w_dot_planar = cs.vertcat(0.0, 0.0, (1/self.inertia[2,2]) * (tau[2] - 0.0))

        # Quaternion derivative only for planar rotation (about z)
        q_dot_planar = 1 / 2 * cs.mtimes(R.skew_symmetric_cs(w_planar), q_normalized)

        f_expl = cs.vertcat(
            v_planar,                # p_dot
            a_thrust_planar,         # v_dot
            q_dot_planar,            # q_dot
            w_dot_planar             # w_dot
        )

        f_impl = xdot - f_expl

        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.name = self.name

        return model
