import numpy as np
from envs.env_config import G1Cfg


class Reward:
    def __init__(self):
        self.cfg = G1Cfg.rewards
        self.scales = {k: v for k, v in vars(G1Cfg.rewards.scales).items() if not k.startswith("__")}

        # 初始化奖励函数所需数据
        self.root_lin_vel_z = 0.0
        self.root_ang_vel_xy = []
        self.projected_gravity_xy = []
        self.root_height = 0.0
        self.commands = []
        self.root_lin_vel_xy = []
        self.root_ang_vel_z = 0.0
        self.torques = []
        self.dof_vel = []  # 需剔除前6项
        self.dof_acc = []  # 需剔除前6项
        self.dof_pos = []  # 需剔除前7项
        self.pos_limits = []
        self.actions = []
        self.last_actions = []
        self.feet_ids = []
        self.feet_confrc = []
        self.feet_contact = []
        self.feet_pos = []
        self.feet_vel = []
        self.leg_phase = []
        self.default_dof_pos = []
        self.last_feet_contacts = np.zeros(0, dtype=np.float64)
        self.feet_air_time = np.zeros(0, dtype=np.float64)
        self.dt = 0.0

    def reset_episode(self, num_feet):
        self.last_feet_contacts = np.zeros(num_feet, dtype=np.float64)
        self.feet_air_time = np.zeros(num_feet, dtype=np.float64)

    def reward_lin_vel_z(self):  # √ 根body z轴线速度
        return self.root_lin_vel_z ** 2

    def reward_ang_vel_xy(self):  # √ 根body xy轴角速度
        return np.sum(self.root_ang_vel_xy ** 2)

    def reward_orientation(self):  # √ 根body xy轴的重力分量
        return np.sum(self.projected_gravity_xy ** 2)

    def reward_base_height(self):  # √ 根body偏离基准高度
        return (self.root_height - self.cfg.root_height_target) ** 2

    def reward_torques(self):  # √ 力矩
        return np.sum(self.torques ** 2)

    def reward_dof_vel(self):  # √ 关节速度
        return np.sum(self.dof_vel ** 2)

    def reward_dof_acc(self):  # √ 关节加速度
        return np.sum(self.dof_acc ** 2)

    def reward_dof_pos_limits(self):  # √ 关节运动超限量
        lower_violation = np.clip(self.pos_limits[:, 0] - self.dof_pos, a_min=0, a_max=None)
        upper_violation = np.clip(self.dof_pos - self.pos_limits[:, 1], a_min=0, a_max=None)
        return np.sum(lower_violation + upper_violation)

    def reward_action_rate(self):  # √ 动作变化幅度
        return np.sum((self.last_actions - self.actions) ** 2)

    def reward_hip_pos(self):  # √ 髋关节姿态
        return np.sum(self.dof_pos[[1, 2, 7, 8]] ** 2)

    def reward_termination(self, reset):  # √ 终止
        return reset

    def reward_tracking_lin_vel(self):  # √ 线速度指令跟踪程度
        err = np.sum((self.commands[:2] - self.root_lin_vel_xy) ** 2)
        return np.exp(-err / self.cfg.tracking_sigma)

    def reward_tracking_ang_vel(self):  # √ 角速度指令跟踪程度
        err = (self.commands[2] - self.root_ang_vel_z) ** 2
        return np.exp(-err / self.cfg.vel_sigma)

    def reward_forward_vel(self):  # 朝指令方向真正移动
        cmd = np.asarray(self.commands[:2], dtype=np.float64)
        vel = np.asarray(self.root_lin_vel_xy, dtype=np.float64)
        cmd_speed = np.linalg.norm(cmd)
        vel_speed = np.linalg.norm(vel)

        if cmd_speed < 0.05 or vel_speed < 1e-6:
            return 0.0

        alignment = np.dot(cmd, vel) / (cmd_speed * vel_speed + 1e-6)
        alignment = np.clip(alignment, 0.0, 1.0)
        speed_ratio = np.clip(vel_speed / (cmd_speed + 1e-6), 0.0, 1.2)
        return alignment * speed_ratio

    def reward_feet_swing_height(self):  # √ 摆动相位脚摆动高度
        res = 0.0
        for i in range(len(self.feet_ids)):
            desired_stance = self.leg_phase[i] < 0.55  # 期望相位判断
            pos_error = (self.feet_pos[i, 2] - self.cfg.feet_height_target) ** 2
            res += pos_error * (1 - desired_stance)
        return res

    def reward_feet_air_time(self):  # 鼓励迈步而不是原地硬扛
        num_feet = len(self.feet_ids)
        if self.last_feet_contacts.shape[0] != num_feet:
            self.reset_episode(num_feet)

        contact = np.asarray(self.feet_contact, dtype=np.float64) > 0.5
        first_contact = np.logical_and(contact, np.logical_not(self.last_feet_contacts > 0.5))

        rew = 0.0
        if np.linalg.norm(self.commands[:2]) > 0.1:
            air_time_bonus = np.clip(
                self.feet_air_time - self.cfg.min_feet_air_time,
                0.0,
                self.cfg.max_feet_air_time - self.cfg.min_feet_air_time,
            )
            rew = float(np.sum(air_time_bonus * first_contact.astype(np.float64)))

        self.feet_air_time += self.dt
        self.feet_air_time[contact] = 0.0
        self.last_feet_contacts = contact.astype(np.float64)
        return rew

    def reward_contact(self):  # √ 支撑相位脚接触
        res = 0.0
        for i in range(len(self.feet_ids)):
            desired_stance = self.leg_phase[i] < 0.55  # 期望相位判断
            contact = self.feet_contact[i] > 0.5  # 实际接触判断
            res += float(not (contact ^ desired_stance))
        return res

    def reward_contact_no_vel(self):  # √ 支撑相位脚速度
        res = 0.0
        for i in range(len(self.feet_ids)):
            desired_stance = self.leg_phase[i] < 0.55  # 期望相位判断
            in_contact = self.feet_contact[i] > 0.5
            res += np.sum(self.feet_vel[i, :2] ** 2) * desired_stance * in_contact
        return res

    def reward_feet_slip(self):  # √ 触地滑动惩罚（只看平面速度）
        res = 0.0
        for i in range(len(self.feet_ids)):
            in_contact = self.feet_contact[i] > 0.5
            if in_contact:
                res += np.sum(self.feet_vel[i, :2] ** 2)
        return res

    def reward_upper_body_pose(self):  # 稳住腰和手臂，减少无关自由度噪声
        if len(self.dof_pos) == 0 or len(self.default_dof_pos) != len(self.dof_pos):
            return 0.0

        upper_body_slice = slice(12, len(self.dof_pos))
        upper_body_error = self.dof_pos[upper_body_slice] - self.default_dof_pos[upper_body_slice]
        return np.sum(upper_body_error ** 2)

    def reward_alive(self):  # √ 依然活着
        return 1.0