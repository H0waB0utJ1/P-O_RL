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
        self.dof_vel = [] # 需剔除前6项
        self.dof_acc = [] # 需剔除前6项
        self.dof_pos = [] # 需剔除前7项
        self.pos_limits = []
        self.actions = []
        self.last_actions = []
        self.feet_ids = []
        self.feet_confrc = []
        self.feet_pos = []
        self.feet_vel = []
        self.leg_phase = []

    def reward_lin_vel_z(self): # √ 根body z轴线速度
        return self.root_lin_vel_z ** 2

    def reward_ang_vel_xy(self): # √ 根body xy轴角速度
        return np.sum(self.root_ang_vel_xy ** 2)

    def reward_orientation(self): # √ 根body xy轴的重力分量
        return np.sum(self.projected_gravity_xy ** 2)

    def reward_base_height(self): # √ 根body偏离基准高度
        return (self.root_height - self.cfg.root_height_target) ** 2

    def reward_torques(self): # √ 力矩
        return np.sum(self.torques ** 2)

    def reward_dof_vel(self): # √ 关节速度
        return np.sum(self.dof_vel ** 2)

    def reward_dof_acc(self): # √ 关节加速度
        return np.sum(self.dof_acc ** 2)

    def reward_dof_pos_limits(self): # √ 关节运动超限量
        lower_violation = np.clip(self.pos_limits[:, 0] - self.dof_pos, a_min=0, a_max=None)
        upper_violation = np.clip(self.dof_pos - self.pos_limits[:, 1], a_min=0, a_max=None)
        return np.sum(lower_violation + upper_violation)

    def reward_action_rate(self): # √ 动作变化幅度
        return np.sum((self.last_actions - self.actions) ** 2)

    def reward_hip_pos(self): # √ 髋关节姿态
        return np.sum(self.dof_pos[[1, 2, 7, 8]] ** 2)

    def reward_termination(self, reset): # √ 终止
        return reset

    def reward_tracking_lin_vel(self): # √ 线速度指令跟踪程度
        err = np.sum((self.commands[:2] - self.root_lin_vel_xy) ** 2)
        return np.exp(-err / self.cfg.tracking_sigma)

    def reward_tracking_ang_vel(self): # √ 角速度指令跟踪程度
        err = (self.commands[2] - self.root_ang_vel_z) ** 2
        return np.exp(-err / self.cfg.vel_sigma)

    def reward_feet_swing_height(self): # √ 摆动相位脚摆动高度
        res = 0.0
        for i in range(len(self.feet_ids)):
            desired_stance = self.leg_phase[i] < 0.55 # 期望相位判断
            pos_error = (self.feet_pos[i, 2] - self.cfg.feet_height_target) ** 2
            res += pos_error * (1 - desired_stance)
        return res
    
    def reward_contact(self): # √ 支撑相位脚接触
        res = 0.0
        for i in range(len(self.feet_ids)):
            desired_stance = self.leg_phase[i] < 0.55 # 期望相位判断
            contact = self.feet_confrc[i, 2] > 1 # 实际接触判断
            res += float(not (contact ^ desired_stance))
        return res

    def reward_contact_no_vel(self): # √ 支撑相位脚速度
        res = 0.0
        for i in range(len(self.feet_ids)):
            desired_stance = self.leg_phase[i] < 0.55 # 期望相位判断
            v = self.feet_vel[i]
            res += np.sum(v ** 2) * desired_stance
        return res

    def reward_feet_slip(self): # √ 触地滑动惩罚（只看平面速度）
        res = 0.0
        for i in range(len(self.feet_ids)):
            in_contact = self.feet_confrc[i, 2] > 5.0
            if in_contact:
                res += np.sum(self.feet_vel[i, :2] ** 2)
        return res

    def reward_alive(self): # √ 依然活着
        return 1.0
