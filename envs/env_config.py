class G1Cfg():
    class env:
        # sim parameter
        send_timeouts = True
        if_render = False
        if_test = False

        # model parameter
        robot_body_name = "g1_29dof"
        foot_name = "ankle_roll"
        terminate_after_contacts_on = ["pelvis"]
        robot_root_name = "pelvis"
        num_dofs = 29
        free_jnt_id = 0
        num_feet = 2
        dim_actions = 29
        dim_obs = 103

    class init_state():
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,       
           'torso_joint' : 0.
        }

    class rewards:
        only_positive_rewards = False
        tracking_sigma = 0.25
        swing_sigma = 0.25
        vel_sigma = 0.25

        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.

        root_height_target = 0.78
        feet_height_target = 0.08
        max_contact_force = 100.

        class scales:
            # torso stability
            lin_vel_z = -1.0
            ang_vel_xy = -0.2
            orientation = -4.0
            base_height = -10.0

            # motion regularization
            torques = -2e-4
            dof_vel = -1e-4
            dof_acc = -2.5e-7
            dof_pos_limits = -10.0
            action_rate = -0.02
            hip_pos = -0.4

            # safety
            termination = -150.0

            # command tracking (main learning signal)
            tracking_lin_vel = 30.0
            tracking_ang_vel = 3.0

            # gait shaping
            feet_swing_height = -30.0
            contact = 1.2
            contact_no_vel = -2.0
            feet_slip = -2.0

            # survive bonus
            alive = 1.0

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 5.0

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # lin_vel_x, lin_vel_y, ang_vel_yaw, heading
        resampling_time = 2.0
        heading_command = False
        heading_kp = 0.5

        class ranges:
            lin_vel_x = [0.2, 0.6]
            lin_vel_y = [-0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class control:
        stiffness = {
            'hip_yaw': 600,
            'hip_roll': 600,
            'hip_pitch': 600,
            'knee': 900,
            'ankle': 240,
        }
        damping = {
            'hip_yaw': 12,
            'hip_roll': 12,
            'hip_pitch': 12,
            'knee': 24,
            'ankle': 12,
        }
        action_scale = 0.12
        # action_scale = 500
