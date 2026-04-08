import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

class RobotInterface:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, robot_body_name="g1_29dof"):
        self.model = model
        self.data = data

        self.robot_body_name = robot_body_name
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_body_name)

        self.robot_root_name = "pelvis"
        self.robot_root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.robot_root_name)

        self.default_dof_pos = data.qpos.copy() # 默认pos

    # basic qpos/qvel/qacc
    def get_qpos(self): # 获取关节角度
        return self.data.qpos.copy()

    def get_qvel(self): # 获取关节角速度
        return self.data.qvel.copy()

    def get_qacc(self): # 获取关节角加速度
        return self.data.qacc.copy()

    # torques/limits
    def get_joint_torque(self): # 获取关节力矩
        return self.data.qfrc_actuator.copy()

    def get_motor_torque(self): # 获取电机力矩
        return self.data.actuator_force.copy()

    def get_joint_pos_limits(self): # 获取关节运动范围
        return self.model.jnt_range.copy()

    def get_motor_torque_limits(self): # 获取电机力矩范围
        """
        优先用 actuator_forcerange，否则退回 actuator_ctrlrange
        """
        # forcerange
        fr = None
        if hasattr(self.model, "actuator_forcerange") and self.model.actuator_forcerange is not None:
            fr = self.model.actuator_forcerange.copy()
            # 判定“有效”：上下限有限且上限>下限，且范围不是全 0
            if fr.size > 0 and np.all(np.isfinite(fr)):
                span = fr[:, 1] - fr[:, 0]
                if np.all(span > 1e-9) and np.max(np.abs(fr)) > 1e-9:
                    return fr

        # ctrlrange
        cr = self.model.actuator_ctrlrange.copy()

        return cr

    # poses
    @staticmethod
    def _quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
        return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)

    def get_body_pos(self, body_name): # 获取各body世界系位置
        body_names = [body_name] if isinstance(body_name, str) else body_name
        body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names]

        return np.array([self.data.xpos[bid].copy() for bid in body_ids])

    def get_root_quat_xyzw(self): # 获取根body姿态四元数
        q_wxyz = self.data.xquat[self.robot_root_id].copy()
        return self._quat_wxyz_to_xyzw(q_wxyz)

    def get_root_euler(self): # 获取根body欧拉角
        rot = R.from_quat(self.get_root_quat_xyzw())
        return rot.as_euler("XYZ").copy()

    def get_root_yaw_sincos(self): # 获取根body偏航角的sin和cos
        yaw = self.get_root_euler()[2]
        return np.array([np.sin(yaw), np.cos(yaw)], dtype=np.float64)

    def get_root_projected_gravity(self): # 获取根body重力向量
        rot_bw = R.from_quat(self.get_root_quat_xyzw())  # body->world（按常见语义）
        g_world = np.array([0.0, 0.0, -1.0], dtype=np.float64) # 世界坐标系重力方向
        g_body = rot_bw.inv().apply(g_world)
        return g_body.copy()

    # velocities
    def get_body_vel6(self, body_name: str, local: bool = True) -> np.ndarray:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        vel6 = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, bid, vel6, 1 if local else 0
        )
        return vel6.copy()  # [ang(3), lin(3)]  (rot:lin)

    def get_body_lin_vel(self, body_name: str, local: bool = True) -> np.ndarray: # 获取某body线速度
        v6 = self.get_body_vel6(body_name, local=local)
        return v6[3:6].copy()

    def get_body_ang_vel(self, body_name: str, local: bool = True) -> np.ndarray: # 获取某body角速度
        v6 = self.get_body_vel6(body_name, local=local)
        return v6[0:3].copy()

    def get_root_lin_vel(self, local: bool = True) -> np.ndarray: # 获取根body线速度
        return self.get_body_lin_vel(self.robot_root_name, local=local)

    def get_root_ang_vel(self, local: bool = True) -> np.ndarray: # 获取根body角速度
        return self.get_body_ang_vel(self.robot_root_name, local=local)

    def get_body_vel_batch6(self, body_names, local: bool = True) -> np.ndarray:
        body_names = [body_names] if isinstance(body_names, str) else body_names
        return np.stack([self.get_body_vel6(n, local=local) for n in body_names], axis=0)

    # contacts
    def get_body_floor_confrc_contactframe(self, body_name):
        body_names = [body_name] if isinstance(body_name, str) else body_name
        body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, bn) for bn in body_names]

        forces = np.zeros((len(body_ids), 6), dtype=np.float64)

        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1_body = self.model.geom_bodyid[con.geom1]
            g2_body = self.model.geom_bodyid[con.geom2]
            g1_root = self.model.body_rootid[g1_body]
            g2_root = self.model.body_rootid[g2_body]

            g1_is_robot = (g1_root == self.robot_root_id)
            g2_is_robot = (g2_root == self.robot_root_id)
            g1_is_floor = not g1_is_robot
            g2_is_floor = not g2_is_robot

            g1_is_target = g1_body in body_ids
            g2_is_target = g2_body in body_ids

            if (g1_is_floor and g2_is_target) or (g2_is_floor and g1_is_target):
                f = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, i, f)

                target_body = g1_body if g2_is_floor else g2_body
                idx = body_ids.index(target_body)
                forces[idx] += f

        return forces

    def contact_force_world_from_contact(self, con, f_contact3: np.ndarray) -> np.ndarray:
        A = np.array(con.frame, dtype=np.float64).reshape(3, 3) # rows are axes in world
        return (A.T @ f_contact3.reshape(3)).copy()

    # gait phase
    def compute_leg_phase(self, episode_step_count, dt): # 计算腿部相位
        period = 0.8 # 期望周期
        offset = 0.5
        phase = (episode_step_count * dt) % period / period
        return np.array([phase, (phase + offset) % 1.0], dtype=np.float64)

    # electronic skin
    def get_geom_contact_forces_raw(self, geom_name):
        geom_names = [geom_name] if isinstance(geom_name, str) else list(geom_name)
        geom_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in geom_names]
        geom_id_to_idx = {gid: idx for idx, gid in enumerate(geom_ids)}

        forces = np.zeros((len(geom_ids), 6), dtype=np.float64)

        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = con.geom1, con.geom2

            if g1 in geom_id_to_idx or g2 in geom_id_to_idx:
                f = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, i, f)

                if g1 in geom_id_to_idx:
                    forces[geom_id_to_idx[g1]] += f
                if g2 in geom_id_to_idx:
                    forces[geom_id_to_idx[g2]] += f

        return forces

    def get_geom_floor_contact_normal_forces(self, geom_name):
        raw_forces = self.get_geom_contact_forces_raw(geom_name)
        # mj_contactForce中第0维是法向力
        normal_forces = np.abs(raw_forces[:, 0])
        return normal_forces

    def get_geom_floor_contact_flags(self, geom_name, threshold=1e-6):
        normal_forces = self.get_geom_floor_contact_normal_forces(geom_name)
        return (normal_forces > threshold).astype(np.float64)

    def get_geom_floor_contact_force_matrix(self, geom_name, rows, cols):
        forces = self.get_geom_floor_contact_normal_forces(geom_name)
        return forces.reshape(rows, cols)

    def get_geom_floor_contact_flag_matrix(self, geom_name, rows, cols, threshold=1e-6):
        flags = self.get_geom_floor_contact_flags(geom_name, threshold)
        return flags.reshape(rows, cols)