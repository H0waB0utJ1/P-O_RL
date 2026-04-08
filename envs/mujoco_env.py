import mujoco
import numpy as np
from envs.robot_interface import RobotInterface
from envs.reward import Reward
from envs.env_config import G1Cfg
import time
import mujoco.viewer

class MujocoEnv: # 仿真环境接口
    #def __init__(self, xml_path: str, max_episode_step_count):
    def __init__(self, xml_path: str):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.cfg = G1Cfg
        self.normalization = G1Cfg.normalization
        self.obs_scales = G1Cfg.normalization.obs_scales
        self.commands_cfg = G1Cfg.commands
        self.env = G1Cfg.env
        self.control = G1Cfg.control
        self.init_state = G1Cfg.init_state
        self.commands_scale = np.array([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel])
        self.robot = RobotInterface(self.model, self.data, self.env.robot_body_name) # 加载机器人接口
        self.reward = Reward() # 加载奖励函数

        self.body_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) for body_id in range(self.model.nbody)]
        self.feet_names = [s for s in self.body_names if self.env.foot_name in s]
        self.feet_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in self.feet_names]
        self.jnt_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            for joint_id in range(self.model.njnt) if joint_id != self.env.free_jnt_id] # 剔除自由关节的关节名
        self.if_done = 0
        self.if_time_out = 0
        self.dt = self.model.opt.timestep # 仿真步长
        self.step_count = 0 # 全局仿真步数
        #self.max_episode_step_count = max_episode_step_count # 单轨迹最大步长
        if self.env.if_render: # 渲染
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.episode_env_init() # episode仿真环境初始化
        self.init_control() # 控制初始化
        self.prepare_reward_function() # 奖励函数准备

    def episode_env_init(self): # episode仿真环境初始化
        self.episode_step_count = 0 # episode步进计数
        self.start_time = time.time()

        self.obs = np.zeros(self.env.dim_obs, dtype=float) # 状态空间
        self.actions = np.zeros(self.env.dim_actions, dtype=float) # 动作空间
        self.last_actions = self.actions
        self.commands = np.zeros(self.commands_cfg.num_commands, dtype=float) # 指令
        self.leg_phase = np.zeros(self.env.num_feet, dtype=float) # 腿部相位

        self.resample_commands() # 更新指令
    
    def reset(self): # 重置仿真
        mujoco.mj_resetData(self.model, self.data)
        # 严格初始化到站立默认姿态
        if hasattr(self.robot, "default_dof_pos") and self.robot.default_dof_pos is not None:
            qpos0 = np.asarray(self.robot.default_dof_pos, dtype=np.float64)
            if qpos0.shape[0] == self.model.nq:
                self.data.qpos[:] = qpos0
        # 速度/加速度清零，避免上一回合残留
        if self.model.nv > 0:
            self.data.qvel[:] = 0.0
        if self.model.na > 0 and hasattr(self.data, "qacc"):
            self.data.qacc[:] = 0.0
        if hasattr(self.data, "ctrl") and self.data.ctrl is not None:
            self.data.ctrl[:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        
        self.episode_env_init() # episode仿真环境初始化

        self.episode_sums_rewards = {name: 0.0 for name in self.reward.scales.keys()} # episode累计值清零

        self.get_obs() # 获取状态空间
        return self.obs

    def init_control(self): # 控制初始化
        # 目标：action=0 时，PD 会把各关节拉回 cfg.init_state.default_joint_angles
        # 同时把 free joint 的 (pos, quat) 写入 default_qpos，避免“默认姿态”依赖 XML 初始 qpos。
        self.p_gains = np.zeros(self.env.num_dofs, dtype=np.float64)
        self.d_gains = np.zeros(self.env.num_dofs, dtype=np.float64)

        # 从当前模型的 qpos 拷贝一份作为 default_qpos 基础
        default_qpos = self.robot.default_dof_pos.copy()

        # free joint: [x, y, z, qw, qx, qy, qz]
        if hasattr(self.init_state, "pos") and len(self.init_state.pos) == 3:
            default_qpos[0:3] = np.array(self.init_state.pos, dtype=np.float64)
        default_qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # hinge joints: qpos[7:] 与 self.jnt_names 的顺序一致（free joint 已剔除）
        self.default_dof_pos = np.zeros(self.env.num_dofs, dtype=np.float64)
        for i in range(self.env.num_dofs):
            name = self.jnt_names[i]

            # 默认关节角：优先从配置读取；未配置的关节默认 0
            angle = 0.0
            if hasattr(self.init_state, "default_joint_angles") and name in self.init_state.default_joint_angles:
                angle = float(self.init_state.default_joint_angles[name])
            self.default_dof_pos[i] = angle
            default_qpos[7 + i] = angle

            # PD 增益：按关节名子串匹配（hip/knee/ankle 等）
            found = False
            for key in self.control.stiffness.keys():
                if key in name:
                    self.p_gains[i] = float(self.control.stiffness[key])
                    self.d_gains[i] = float(self.control.damping[key])
                    found = True
                    break

            # 未匹配到的关节给一个保守默认值（比置 0 更不容易“软塌/漂”）
            if not found:
                self.p_gains[i] = 500.0
                self.d_gains[i] = 10.0

        # 同步回 RobotInterface，用于 compute_torques() 里 target = action + default
        self.robot.default_dof_pos = default_qpos
    
    def step(self, actions): # 仿真步进
        clip_actions = self.normalization.clip_actions
        self.actions = np.clip(actions, -clip_actions, clip_actions) # 剪裁动作

        self.torques = self.compute_torques(self.actions) # 计算电机力矩
        self.data.ctrl[:] = self.torques # 写入控制
        mujoco.mj_step(self.model, self.data) # 模拟一步

        self.post_physics_step() # 计算观测、奖励、done
        
        if self.env.if_test: # 实时运行
            sim_t = self.data.time
            real_t = time.time() - self.start_time
            if sim_t > real_t: # 仿真时间更快，需等待现实时间同步
                time.sleep(sim_t - real_t)

        clip_obs = self.normalization.clip_observations
        self.obs = np.clip(self.obs, -clip_obs, clip_obs) # 剪裁观测值

        if self.env.if_render: # 渲染
            self.viewer.sync()
            #if self.if_done:
                #self.viewer.close()

        #print("cmd_xy:", self.commands[:2], "vel_xy:", self.reward.root_lin_vel_xy)

        return self.obs, self.rew, self.if_done, self.episode_sums_rewards
    
    def post_physics_step(self): # 仿真步进计算
        self.episode_step_count += 1 # 单轮仿真步数+1
        self.step_count += 1 # 全局仿真步数+1

        self.post_physics_step_callback() # 自定义回调（处理特定功能要求）

        self.check_termination() # 环境终止判断
        
        self.compute_reward() # 奖励计算
        self.get_obs() # 获取状态空间

        self.last_actions = self.actions

    def post_physics_step_callback(self): # 自定义回调函数
        self.leg_phase = self.robot.compute_leg_phase(self.episode_step_count, self.dt) # 更新腿部相位

        steps_per_resample = int(self.commands_cfg.resampling_time / self.dt)
        if self.episode_step_count % steps_per_resample == 0: # 判断是否需要更新指令
            self.resample_commands() # 更新指令
        if self.commands_cfg.heading_command: # 若为heading模式则需通过P控制器计算目标角速度***（后续是否应对微小角速度归0？）***
            error = self.commands[3] - self.robot.get_root_euler()[2] # 计算当前误差
            ang_vel_cmd = self.commands_cfg.heading_kp * error # P控制器计算目标角速度
            self.commands[2] = np.clip(ang_vel_cmd, -1.0, 1.0) # 将目标角速度限幅并记入指令

        self.update_reward_states() # 更新奖励计算所需状态值
    
    def compute_torques(self, actions): # PD控制器
        actions_scaled = actions * self.control.action_scale
        dof_pos = self.robot.get_qpos()[7:]
        dof_vel = self.robot.get_qvel()[6:]
        target = actions_scaled + self.robot.default_dof_pos[7:]
        torques = self.p_gains * (target - dof_pos) - self.d_gains * dof_vel

        #torques = actions * self.control.action_scale # end to end only

        lower_limits = self.robot.get_motor_torque_limits()[:, 0]
        upper_limits = self.robot.get_motor_torque_limits()[:, 1]

        return np.clip(torques, lower_limits, upper_limits)

    def update_reward_states(self): # 更新奖励计算所需状态值
        root_lin_vel_local = self.robot.get_root_lin_vel(local=True) # local=True -> 机体系表达
        root_ang_vel_local = self.robot.get_root_ang_vel(local=True) # local=True -> 机体系表达

        self.reward.root_lin_vel_z = float(root_lin_vel_local[2])
        self.reward.root_lin_vel_xy = root_lin_vel_local[:2].copy()
        self.reward.root_ang_vel_xy = root_ang_vel_local[:2].copy()
        self.reward.root_ang_vel_z = float(root_ang_vel_local[2])

        self.reward.projected_gravity_xy = self.robot.get_root_projected_gravity()[:2].copy()

        self.reward.root_height = float(self.robot.get_body_pos(self.env.robot_body_name)[0, 2])

        self.reward.commands = self.commands.copy()
        self.reward.torques = self.robot.get_motor_torque()
        self.reward.dof_vel = self.robot.get_qvel()[6:].copy() # 剔除前6项（free joint）
        self.reward.dof_acc = self.robot.get_qacc()[6:].copy()
        self.reward.dof_pos = self.robot.get_qpos()[7:].copy() # 剔除前7项（free joint pos+quat）
        self.reward.pos_limits = self.robot.get_joint_pos_limits()[1:].copy()
        self.reward.actions = self.actions.copy()
        self.reward.last_actions = self.last_actions.copy()

        self.reward.feet_pos = self.robot.get_body_pos(self.feet_names)

        feet_vel6_world = self.robot.get_body_vel_batch6(self.feet_names, local=False) # (N,6) rot:lin
        self.reward.feet_vel = feet_vel6_world[:, 3:6].copy()

        f6_cf = self.robot.get_body_floor_confrc_contactframe(self.feet_names) # (N,6) in contact frame
        fn = f6_cf[:, 0].copy()  # normal force
        feet_confrc = np.zeros((len(self.feet_names), 3), dtype=np.float64)
        feet_confrc[:, 2] = fn
        self.reward.feet_confrc = feet_confrc

        self.reward.leg_phase = self.leg_phase.copy()

    def get_obs(self): # 获取状态空间
        # 期望相位处理为连续值
        sin_phase = np.sin(2 * np.pi * self.leg_phase[0])
        cos_phase = np.cos(2 * np.pi * self.leg_phase[0])
        leg_phase_sincos = np.array([sin_phase, cos_phase], dtype=np.float64)

        # 获取观测值
        root_lin_vel_local = self.robot.get_root_lin_vel(local=True)
        root_ang_vel_local = self.robot.get_root_ang_vel(local=True)

        self.obs = np.concatenate([
            root_lin_vel_local * self.obs_scales.lin_vel,                                           # 根body线速度（机体系）
            root_ang_vel_local * self.obs_scales.ang_vel,                                           # 根body角速度（机体系）
            self.robot.get_root_projected_gravity(),                                                # 根body重力分量（机体系）
            (self.robot.get_qpos()[7:] - self.robot.default_dof_pos[7:]) * self.obs_scales.dof_pos, # 关节角
            self.robot.get_qvel()[6:] * self.obs_scales.dof_vel,                                    # 关节角速度
            self.robot.get_root_yaw_sincos(),                                                       # 根body偏航角 sin/cos
            self.commands[:3] * self.commands_scale,                                                # 指令
            self.actions,                                                                           # 历史动作
            leg_phase_sincos                                                                        # 期望相位 sin/cos
        ])

    def check_termination(self): # 终止检测
        # 接触力终止
        f6_cf = self.robot.get_body_floor_confrc_contactframe(self.env.terminate_after_contacts_on)
        fn = f6_cf[:, 0] # normal force

        contact_terminate = np.any(fn > 50.0)

        # 姿态终止
        rpy = self.robot.get_root_euler()
        orientation_terminate = (abs(rpy[1]) > 0.5) or (abs(rpy[0]) > 0.5)

        # 总终止条件
        self.if_done = bool(contact_terminate or orientation_terminate)

    def resample_commands(self): # 运动指令生成
        lin_x = np.random.uniform(*self.commands_cfg.ranges.lin_vel_x)
        lin_y = np.random.uniform(*self.commands_cfg.ranges.lin_vel_y)
        if np.linalg.norm([lin_x, lin_y]) <= 0.1: # 如果过小则直接设为0，防止微抖
            lin_x, lin_y = 0.0, 0.0
        self.commands[0] = lin_x
        self.commands[1] = lin_y

        if self.commands_cfg.heading_command: # 以角度为目标指令
            self.commands[3] = np.random.uniform(*self.commands_cfg.ranges.heading)
        else: # 以角速度为目标指令
            self.commands[2] = np.random.uniform(*self.commands_cfg.ranges.ang_vel_yaw)
            self.commands[3] = 0.0 # 未使用 heading 时置零

    def prepare_reward_function(self): # 准备scale非0奖励函数列表
        self.reward.feet_ids = self.feet_ids

        for key in list(self.reward.scales.keys()): # 清除scale为0的奖励并乘dt
            scale = self.reward.scales[key]
            if scale == 0:
                self.reward.scales.pop(key)
            elif key != "termination":
                self.reward.scales[key] *= self.dt

        self.reward_functions = [] # 准备奖励函数列表
        self.reward_names = []
        for name, scale in self.reward.scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            func_name = "reward_" + name
            self.reward_functions.append(getattr(self.reward, func_name))
        self.rew = 0.0 # 单步奖励
        self.episode_sums_rewards = {name: 0.0 for name in self.reward.scales.keys()} # episode累计值

    def compute_reward(self): # 计算单步总奖励并累加
        self.rew = 0.0
        
        for i in range(len(self.reward_functions)): # 逐个奖励项累加
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward.scales[name]
            self.rew += rew
            self.episode_sums_rewards[name] += rew
            #print(name, rew)

        if self.reward.cfg.only_positive_rewards: # 如果需要只保留正奖励
            self.rew = max(self.rew, 0.0)
        
        if "termination" in self.reward.scales: # 处理termination奖励
            #rew = self.reward.reward_termination(self.if_done, self.if_time_out) * self.reward.scales["termination"]
            rew = self.reward.reward_termination(self.if_done) * self.reward.scales["termination"]
            self.rew += rew
            self.episode_sums_rewards["termination"] += rew
            #print("termination", rew)