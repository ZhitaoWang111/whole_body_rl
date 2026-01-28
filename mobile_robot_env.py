"""
Clean MuJoCo environment for Piper robot arm fruit picking task.
This environment uses only state observations (joint angles + end effector pose + object position).
"""
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
import os
from scipy.spatial.transform import Rotation as R
import time
# from mujoco_sim2real.viewer.gs_render.gaussian_renderer import GSRenderer
from viewer.gs_render.gaussian_renderer import GSRenderer
import torch
from car_controller.base_controller import SwerveBaseController


class PiperEnv(gym.Env):
    """
    Piper robot arm environment for fruit picking task.
    
    Task: Left arm grabs hot dog and places it in basket
    """
    def __init__(
        self,
        visualization: bool = False,
        gs_render: bool = True,
        max_episode_length: int = 80,
        use_top_rgb: bool = True,
        use_left_wrist_rgb: bool = True,
        use_right_wrist_rgb: bool = False,
    ):
        super(PiperEnv, self).__init__()
        self.assets_dir = '/home/wzt/wzt/mycode/whole_body_rl/model_assets'
        xml_path = os.path.join(self.assets_dir, 'fw_mini_single_piper', 'fw_mini_single_piper.xml')

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = 0.002
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.visualization = bool(visualization)

        if self.visualization:
            self.handle = mujoco.viewer.launch_passive(self.model, self.data)   # 创建一个被动渲染窗口(GUI)，可以实时查看仿真过程
            self.handle.cam.distance = 3                                        # 相机与目标的距离为 3
            self.handle.cam.azimuth = 100                                       # 方位角为 0 度
            self.handle.cam.elevation = -60 
        else:
            self.handle = None

        # Joint limits for single arm (7 joints: 6 arm + 1 gripper)
        self.joint_limits = np.array([
            (-2.618, 2.618),    # joint1
            (0, 3.14),          # joint2 
            (-2.697, 0),        # joint3
            (-1.832, 1.832),    # joint4
            (-1.22, 1.22),      # joint5
            (-3.14, 3.14),      # joint6
            (0, 0.045),         # gripper
        ])

        # Single arm action space (7 DOF)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(10,))

        # Arm joint naming (from XML)
        self.arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
        self.state_joint_names = self.arm_joint_names[:6]

        # Task/body/site names
        self.ee_site_name = "ee_site"
        self.grasp_site_name = "grasp_site"
        self.bottle_body_name = "bottle"
        self.desk_body_name = "desk"
        self.desk_mesh_name = "desk_seg"

        # Robot body names for collision penalties
        self.robot_body_names = ["base_link", "base_arm_link"] + [f"link{i}" for i in range(1, 9)]

        # Desk bounds from mesh (local) + desk body pose (world)
        desk_pos = np.asarray(self.data.body(self.desk_body_name).xpos, dtype=np.float32)
        self.desk_top_z = 0.86
        self.desk_xy_bounds = (
            desk_pos[0] - 0.25,  # x_min
            desk_pos[0] - 0.15,  # x_max
            desk_pos[1] - 0.2,   # y_min
            desk_pos[1] + 0.2,   # y_max
        )
        # Bottle tipping threshold (cosine of max tilt angle from upright)
        self.bottle_upright_cos = float(np.cos(np.deg2rad(45.0)))

        # Reward scaling/limits
        self.reach_ee_max_dist = 0.5    # ee to grasp (m)
        self.reach_ee_exp_k = 4.0       # ee 距离奖励指数衰减系数
        self.reach_ee_scale = 2.0       # ee 接近奖励放大系数（0~2）
        self.reach_ee_thresh = 0.05
        self.gripper_close_thresh = 0.01
        self.lift_start = 0.0
        self.lift_target = 0.05
        self.lift_hold_thresh = 0.06
        self.hold_steps = 10
        # 抬起阶段奖励参数
        self.lift_reward_scale = 1.0
        self.lift_success_bonus = 1.0
        # 抓取阶段门控参数（分阶段速度限制）
        self.base_v_max_far = 0.05
        self.base_omega_max_far = 0.10
        self.base_v_max_near = 0.03
        self.base_omega_max_near = 0.05
        self.gripper_open_threshold = 0.035
        self.gripper_close_threshold = 0.02
        self.ee_pos_tol = 0.03
        self.ee_orient_dist = 0.12  # 夹爪朝向奖励启用距离
        self.ee_orient_scale = 0.3  # 夹爪朝向奖励系数
        self.grasp_ready_bonus = 0.5
        self.grasp_contact_steps = 5
        self.grasp_progress_scale = 0.5
        # 惩罚参数
        self.penalty_object_fell = 5.0
        self.penalty_object_tipped = 3.0
        self.penalty_table_contact = 0.2
        self.penalty_early_close = 0.02
        self.penalty_base_motion = 0.02
        self.penalty_time = 0.001
        self.table_contact_terminate_steps = 10  # 累计接触桌面次数达到即终止
        self.table_contact_count = 0
        # 底盘到达区（相对瓶子位置的矩形范围）
        self.base_reach_x_range = (-0.5, -0.25)
        self.base_reach_y_range = (-0.25, 0.25)
        # 底盘朝向侧向分量阈值（|sin|）
        self.base_facing_cos = float(np.cos(np.deg2rad(30.0)))
        # 底盘位置奖励参数
        self.base_reach_max_dist = 0.5
        self.base_reach_exp_k = 4.0
        self.base_progress_scale = 1.0

        # Bottle rest height offset relative to desk top
        # bottle_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.bottle_body_name)
        # bottle_joint_id = self.model.body_jntadr[bottle_body_id]
        # bottle_qposadr = self.model.jnt_qposadr[bottle_joint_id]
        # bottle_init_z = float(self.model.qpos0[bottle_qposadr + 2])
        # self.bottle_z_offset = max(0.02, bottle_init_z - self.desk_top_z)
        # import pdb; pdb.set_trace()

        # Base command scaling (m/s, m/s, rad/s)
        # Keep as 1.0 to match cmd_vel semantics (vx, vy, yaw)
        self.base_cmd_scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        
        self.camera_width = 128
        self.camera_height = 128

        self.use_top_rgb = use_top_rgb
        self.use_left_wrist_rgb = use_left_wrist_rgb
        self.use_right_wrist_rgb = use_right_wrist_rgb

        # Simplified observation space: always left arm
        obs_dict = {
            # state = [base_x, base_y, joint1..joint6, gripper(0-1)]
            'state': spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32),
            # delta between ee_site and grasp_site (world xyz)
            'target': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        }

        if self.use_top_rgb:
            obs_dict['top_rgb'] = spaces.Box(
                low=0,
                high=255,
                shape=(self.camera_height, self.camera_width, 3),
                dtype=np.uint8,
            )
        if self.use_left_wrist_rgb:
            obs_dict['left_wrist_rgb'] = spaces.Box(
                low=0,
                high=255,
                shape=(self.camera_height, self.camera_width, 3),
                dtype=np.uint8,
            )
        if self.use_right_wrist_rgb:
            obs_dict['right_wrist_rgb'] = spaces.Box(
                low=0,
                high=255,
                shape=(self.camera_height, self.camera_width, 3),
                dtype=np.uint8,
            )
            
        self.observation_space = spaces.Dict(obs_dict)

        self.goal_reached = False
        self._reset_noise_scale = 0.0
        self.episode_len = max_episode_length  # Match rollout length to avoid incomplete episodes
        
        # Initialize joint positions for both arms (we still need to set both, but only control active one)
        self.init_qpos_left = np.zeros(7)
        self.init_qpos_left[0] = 0.0      # left_joint1dtype=torch.float32
        self.init_qpos_left[1] = 1.1      # left_joint2
        self.init_qpos_left[2] = -0.95    # left_joint3
        self.init_qpos_left[3] = 0.0      # left_joint4
        self.init_qpos_left[4] = 0.976    # left_joint5
        self.init_qpos_left[5] = 0.0      # left_joint6
        self.init_qpos_left[6] = 0.045    # left_gripper

        self.init_qpos_right = np.zeros(7)
        self.init_qpos_right[0] = 0.0      # right_joint1
        self.init_qpos_right[1] = 1.1      # right_joint2
        self.init_qpos_right[2] = -0.95    # right_joint3
        self.init_qpos_right[3] = 0.0      # right_joint4
        self.init_qpos_right[4] = 0.976    # right_joint5
        self.init_qpos_right[5] = 0.0      # right_joint6
        self.init_qpos_right[6] = 0.045    # right_gripper

        self.init_qvel = np.zeros(7)
        self.contact_streak = 0
        self.max_contact_streak = 0
        
        # Initialize persistent renderer for efficiency
        self._renderer = None

        self.gs_render = gs_render

        # Initialize GSRenderer for Gaussian rendering
        if self.gs_render:
            ### 创建 gs 渲染
            self.rgb_fovy = 65
            self.rgb_fovx = 90
            self.rgb_width = self.camera_width
            self.rgb_height = self.camera_height
            self.gs_model_dict = {}

            # 构造 model_assets 的路径
            self.asset_root = os.path.join(self.assets_dir, "3dgs_asserts")

            # 构造 gs_model_dict 的路径
            self.gs_model_dict["background"] = os.path.join(self.asset_root, "scene", "point_cloud_n.ply")
            
            # 本体部分 (移动底盘)
            # self.gs_model_dict["mobile_ai"] = os.path.join(self.asset_root, "robot", "chassis_new", "mobile_ai_v2.ply")
            
            # Left arm components (Piper arm links)
            self.gs_model_dict["base_arm_link"] = os.path.join(self.asset_root, "robot", "piper", "base_link_v2.ply")
            self.gs_model_dict["link1"] = os.path.join(self.asset_root, "robot", "piper", "link1_v2.ply")
            self.gs_model_dict["link2"] = os.path.join(self.asset_root, "robot", "piper", "link2_v2.ply")
            self.gs_model_dict["link3"] = os.path.join(self.asset_root, "robot", "piper", "link3_v2.ply")
            self.gs_model_dict["link4"] = os.path.join(self.asset_root, "robot", "piper", "link4_v2.ply")
            self.gs_model_dict["link5"] = os.path.join(self.asset_root, "robot", "piper", "link5_v2.ply")
            self.gs_model_dict["link6"] = os.path.join(self.asset_root, "robot", "piper", "link6_v2.ply")
            self.gs_model_dict["link7"] = os.path.join(self.asset_root, "robot", "piper", "link7_v2.ply")
            self.gs_model_dict["link8"] = os.path.join(self.asset_root, "robot", "piper", "link8_v2.ply")
            
            # Right arm components (Piper arm links) 
            # self.gs_model_dict["right_base_link"] = os.path.join(self.asset_root, "robot", "piper", "base_link_v2.ply")
            # self.gs_model_dict["right_link1"] = os.path.join(self.asset_root, "robot", "piper", "link1_v2.ply")
            # self.gs_model_dict["right_link2"] = os.path.join(self.asset_root, "robot", "piper", "link2_v2.ply")
            # self.gs_model_dict["right_link3"] = os.path.join(self.asset_root, "robot", "piper", "link3_v2.ply")
            # self.gs_model_dict["right_link4"] = os.path.join(self.asset_root, "robot", "piper", "link4_v2.ply")
            # self.gs_model_dict["right_link5"] = os.path.join(self.asset_root, "robot", "piper", "link5_v2.ply")
            # self.gs_model_dict["right_link6"] = os.path.join(self.asset_root, "robot", "piper", "link6_v2.ply")
            # self.gs_model_dict["right_link7"] = os.path.join(self.asset_root, "robot", "piper", "link7_v2.ply")
            # self.gs_model_dict["right_link8"] = os.path.join(self.asset_root, "robot", "piper", "link8_v2.ply")

            # objects
            self.gs_model_dict["bottle"] = os.path.join(self.asset_root, "object", "bottle", "bottle_seg.ply")
            self.gs_model_dict["desk"] = os.path.join(self.asset_root, "object", "desk", "desk.ply")


            # Update robot link lists to include full mobile robot
            # self.mobile_base_list = ["mobile_ai"]
            # self.left_arm_list = ["left_base_link", "left_link1", "left_link2", "left_link3", "left_link4", "left_link5", "left_link6", "left_link7", "left_link8"]
            # self.right_arm_list = ["right_base_link", "right_link1", "right_link2", "right_link3", "right_link4", "right_link5", "right_link6", "right_link7", "right_link8"]
            self.robot_link_list = ["base_arm_link", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "link8"]
            self.item_list = ["bottle", "desk"]

            self.gs_renderer = GSRenderer(self.gs_model_dict, self.rgb_width, self.rgb_height)
            self.gs_renderer.set_camera_fovy(self.rgb_fovy * np.pi / 180.)

        # Base (swerve) controller for the first 8 control actuators
        self._init_base_controller()

    def _init_base_controller(self) -> None:
        params = {
            "wheel_radius":  0.06545,
            "steer_track": 0.25,
            "wheel_base": 0.36,
            "max_steer_angle_parallel": 1.570,
            "min_turn_radius": 0.47644,
        }

        pid_params = None

        self.base_controller = SwerveBaseController(
            self.model,
            self.data,
            params,
            steer_act_names=["base_sfl", "base_sfr", "base_srl", "base_srr"],
            drive_act_names=["base_dfl", "base_dfr", "base_drl", "base_drr"],
            steer_joint_names=["steer_fl", "steer_fr", "steer_rl", "steer_rr"],
            drive_joint_names=["drive_fl", "drive_fr", "drive_rl", "drive_rr"],
            wheel_body_names=["Wheel3_Link", "Wheel4_Link", "Wheel1_Link", "Wheel2_Link"],
            base_body_name="base_link",
            cmd_vel_swap_xy=True,
            pid_params=pid_params,
            use_pid=True,
        )
        
    def _get_site_pos_ori(self, site_name: str) -> tuple[np.ndarray, np.ndarray]:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id == -1:
            raise ValueError(f"Site '{site_name}' not found")

        position = np.asarray(self.data.site(site_id).xpos, dtype=np.float32)
        xmat = np.asarray(self.data.site(site_id).xmat, dtype=np.float64)  # MuJoCo requires float64
        quaternion = np.zeros(4, dtype=np.float64)  # MuJoCo requires float64
        mujoco.mju_mat2Quat(quaternion, xmat)

        return position, quaternion.astype(np.float32)  # Convert back to float32

    def _get_mesh_aabb(self, mesh_name: str) -> tuple[np.ndarray, np.ndarray]:
        mesh_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MESH, mesh_name)
        if mesh_id == -1:
            raise ValueError(f"Mesh '{mesh_name}' not found")
        vert_adr = self.model.mesh_vertadr[mesh_id]
        vert_num = self.model.mesh_vertnum[mesh_id]
        verts = self.model.mesh_vert[vert_adr:vert_adr + vert_num * 3].reshape(-1, 3)
        return verts.min(axis=0), verts.max(axis=0)
    
    def update_gs_scene(self):
        # Update all robot components using regular body poses
        for name in self.robot_link_list:
            trans, quat_wxyz = self._get_body_pose(name)
            self.gs_renderer.set_obj_pose(name, trans, quat_wxyz)

        # Update environment objects
        for name in self.item_list:
            trans, quat_wxyz = self._get_body_pose(name)
            self.gs_renderer.set_obj_pose(name, trans, quat_wxyz)

        def multiple_quaternion_vector3d(qwxyz, vxyz):
            qw = qwxyz[..., 0]
            qx = qwxyz[..., 1]
            qy = qwxyz[..., 2]
            qz = qwxyz[..., 3]
            vx = vxyz[..., 0]
            vy = vxyz[..., 1]
            vz = vxyz[..., 2]
            qvw = -vx*qx - vy*qy - vz*qz
            qvx =  vx*qw - vy*qz + vz*qy
            qvy =  vx*qz + vy*qw - vz*qx
            qvz = -vx*qy + vy*qx + vz*qw
            vx_ =  qvx*qw - qvw*qx + qvz*qy - qvy*qz
            vy_ =  qvy*qw - qvz*qx - qvw*qy + qvx*qz
            vz_ =  qvz*qw + qvy*qx - qvx*qy - qvw*qz
            return torch.stack([vx_, vy_, vz_], dim=-1).cuda().requires_grad_(False)
        
        def multiple_quaternions(qwxyz1, qwxyz2):
            q1w = qwxyz1[..., 0]
            q1x = qwxyz1[..., 1]
            q1y = qwxyz1[..., 2]
            q1z = qwxyz1[..., 3]

            q2w = qwxyz2[..., 0]
            q2x = qwxyz2[..., 1]
            q2y = qwxyz2[..., 2]
            q2z = qwxyz2[..., 3]

            qw_ = q1w * q2w - q1x * q2x - q1y * q2y - q1z * q2z
            qx_ = q1w * q2x + q1x * q2w + q1y * q2z - q1z * q2y
            qy_ = q1w * q2y - q1x * q2z + q1y * q2w + q1z * q2x
            qz_ = q1w * q2z + q1x * q2y - q1y * q2x + q1z * q2w

            return torch.stack([qw_, qx_, qy_, qz_], dim=-1).cuda().requires_grad_(False)

        if self.gs_renderer.update_gauss_data:
            self.gs_renderer.update_gauss_data = False
            self.gs_renderer.renderer.need_rerender = True
            self.gs_renderer.renderer.gaussians.xyz[self.gs_renderer.renderer.gau_env_idx:] = multiple_quaternion_vector3d(self.gs_renderer.renderer.gau_rot_all_cu[self.gs_renderer.renderer.gau_env_idx:], self.gs_renderer.renderer.gau_ori_xyz_all_cu[self.gs_renderer.renderer.gau_env_idx:]) + self.gs_renderer.renderer.gau_xyz_all_cu[self.gs_renderer.renderer.gau_env_idx:]
            self.gs_renderer.renderer.gaussians.rot[self.gs_renderer.renderer.gau_env_idx:] = multiple_quaternions(self.gs_renderer.renderer.gau_rot_all_cu[self.gs_renderer.renderer.gau_env_idx:], self.gs_renderer.renderer.gau_ori_rot_all_cu[self.gs_renderer.renderer.gau_env_idx:])



    def _get_body_pose(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"Body '{body_name}' not found")
        
        position = np.asarray(self.data.body(body_id).xpos, dtype=np.float32)
        quaternion = np.asarray(self.data.body(body_id).xquat, dtype=np.float32)
        
        return position, quaternion

    def map_action_to_joint_deltas(self, action: np.ndarray) -> np.ndarray:
        max_delta_per_step = np.array([
            0.05, 0.03, 0.03, 0.03, 0.03, 0.05, 0.005
        ], dtype=np.float32)
        
        # Ensure action is a numpy array with proper dtype
        action = np.asarray(action, dtype=np.float32)
        assert action.shape == (7,), f"Action must be 7D for single arm, got {action.shape}"
        
        # Process active arm
        delta_action = action * max_delta_per_step
        gripper_action = action[6]
        
        # Get current gripper position for left arm
        current_gripper_pos = self.data.joint(self.arm_joint_names[6]).qpos[0]
        
        if gripper_action < 0:
            # Close: set to 0
            delta_action[6] = 0.0 - current_gripper_pos
        else:  # gripper_action >= 0
            # Open: set to 0.045  
            delta_action[6] = 0.045 - current_gripper_pos
        
        return delta_action
    
    def apply_joint_deltas_with_limits(self, current_qpos: np.ndarray, delta_action: np.ndarray) -> np.ndarray:
        """Apply delta action to current joint positions with limits."""
        # Ensure arrays are proper numpy arrays with consistent dtype
        current_qpos = np.asarray(current_qpos, dtype=np.float32)
        delta_action = np.asarray(delta_action, dtype=np.float32)
        
        new_qpos = current_qpos + delta_action
        
        # Use single arm joint limits
        lower_bounds = self.joint_limits[:, 0].astype(np.float32)
        upper_bounds = self.joint_limits[:, 1].astype(np.float32)
        return np.clip(new_qpos, lower_bounds, upper_bounds)

    def _set_state(self, qpos_left=None, qvel_left=None, qpos_right=None, qvel_right=None, qpos_base=None, qvel_base=None):
        """Set the state for mobile robot components.
        
        The mobile robot has:
        - freejoint (7 DOF) for the base: indices 0-6 in qpos, 0-5 in qvel
        - left arm joints: indices 7-14 in qpos (8 joints including both gripper fingers)
        - right arm joints: indices 15-22 in qpos (8 joints including both gripper fingers)
        - wheel joints: indices 23+ in qpos
        
        Args:
            qpos_left: Left arm joint positions (8 values) or (7 values, will set joint8 automatically)
            qvel_left: Left arm joint velocities (8 values) or (7 values)
            qpos_right: Right arm joint positions (8 values) or (7 values)
            qvel_right: Right arm joint velocities (8 values) or (7 values)
            qpos_base: Base position and orientation [x, y, z, qw, qx, qy, qz] (7 values)
            qvel_base: Base linear and angular velocities [vx, vy, vz, wx, wy, wz] (6 values)
        """
        
        # Set base state (freejoint: 7 DOF in qpos, 6 DOF in qvel)
        if qpos_base is not None:
            qpos_base = np.asarray(qpos_base, dtype=np.float32)
            assert qpos_base.shape == (7,), f"qpos_base must have shape (7,), got {qpos_base.shape}"
            self.data.qpos[0:7] = np.copy(qpos_base)
            
        if qvel_base is not None:
            qvel_base = np.asarray(qvel_base, dtype=np.float32)
            assert qvel_base.shape == (6,), f"qvel_base must have shape (6,), got {qvel_base.shape}"
            self.data.qvel[0:6] = np.copy(qvel_base)
        
        # Set left arm state (8 joints: left_joint1 through left_joint8)
        if qpos_left is not None:
            qpos_left = np.asarray(qpos_left, dtype=np.float32)
            if qpos_left.shape == (7,):
                # Only set joints 1-7, let joint8 be handled by equality constraint
                self.data.qpos[7:14] = np.copy(qpos_left)
            elif qpos_left.shape == (8,):
                # Set all 8 joints explicitly
                self.data.qpos[7:15] = np.copy(qpos_left)
            else:
                raise ValueError(f"qpos_left must have shape (7,) or (8,), got {qpos_left.shape}")
                
        if qvel_left is not None:
            qvel_left = np.asarray(qvel_left, dtype=np.float32)
            if qvel_left.shape == (7,):
                # Only set velocities for joints 1-7 (dof_addr 6-12)
                self.data.qvel[6:13] = np.copy(qvel_left)
            elif qvel_left.shape == (8,):
                # Set all 8 joint velocities (dof_addr 6-13)
                self.data.qvel[6:14] = np.copy(qvel_left)
            else:
                raise ValueError(f"qvel_left must have shape (7,) or (8,), got {qvel_left.shape}")
        
        # Set right arm state (8 joints: right_joint1 through right_joint8)
        if qpos_right is not None:
            qpos_right = np.asarray(qpos_right, dtype=np.float32)
            if qpos_right.shape == (7,):
                # Only set joints 1-7, let joint8 be handled by equality constraint
                self.data.qpos[15:22] = np.copy(qpos_right)
            elif qpos_right.shape == (8,):
                # Set all 8 joints explicitly
                self.data.qpos[15:23] = np.copy(qpos_right)
            else:
                raise ValueError(f"qpos_right must have shape (7,) or (8,), got {qpos_right.shape}")
                
        if qvel_right is not None:
            qvel_right = np.asarray(qvel_right, dtype=np.float32)
            if qvel_right.shape == (7,):
                # Only set velocities for joints 1-7 (dof_addr 14-20)
                self.data.qvel[14:21] = np.copy(qvel_right)
            elif qvel_right.shape == (8,):
                # Set all 8 joint velocities (dof_addr 14-21)
                self.data.qvel[14:22] = np.copy(qvel_right)
            else:
                raise ValueError(f"qvel_right must have shape (7,) or (8,), got {qvel_right.shape}")
        
        # Update kinematics without advancing simulation
        mujoco.mj_forward(self.model, self.data)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        
        # Initialize base position using values from model.qpos0
        qpos_base = self.model.qpos0[0:7].copy()    # Use initial qpos from XML: [-0.85, 1.35, -0.133, 0.68949843, 0.0, 0.0, 0.72428717]
        qvel_base = np.zeros(6)                     # [vx, vy, vz, wx, wy, wz]
        
        # Initialize both arms to default position
        qpos_left = np.zeros(7)
        qvel_left = self.init_qvel.copy()
        qpos_right = np.zeros(7)
        qvel_right = self.init_qvel.copy()

        self._set_state(qpos_left=qpos_left, qvel_left=qvel_left, 
                       qpos_base=qpos_base, qvel_base=qvel_base,
                       qpos_right=qpos_right, qvel_right=qvel_right)
        self._reset_object_pose()
        
        obs = self._get_observation()
        
        self.step_number = 0
        self.goal_reached = False
        self.lifted = False
        self.contact_streak = 0
        self.max_contact_streak = 0
        self.table_contact_count = 0
        self.prev_dist_to_box = None

        return obs, {}  # Gymnasium API returns (observation, info)
    
    def set_goal_pose(self, goal_body_name, position, quat_wxyz):
        """Set target pose for a body."""
        goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, goal_body_name)
        if goal_body_id == -1:
            raise ValueError(f"Body '{goal_body_name}' not found")

        goal_joint_id = self.model.body_jntadr[goal_body_id]
        goal_qposadr = self.model.jnt_qposadr[goal_joint_id]

        if goal_qposadr + 7 <= self.model.nq:
            self.data.qpos[goal_qposadr: goal_qposadr + 3] = position
            self.data.qpos[goal_qposadr + 3: goal_qposadr + 7] = quat_wxyz
    
    def _reset_object_pose(self):
        def reset_object(name, center_x, center_y, height=0.72, max_radius=0.1):
            """Reset object to random position in circle around center."""
            # Get initial quaternion from XML (not current quaternion)
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            joint_id = self.model.body_jntadr[body_id]
            qposadr = self.model.jnt_qposadr[joint_id]
            quat = self.model.qpos0[qposadr + 3: qposadr + 7]  # Initial quaternion from XML
            
            if max_radius == 0:
                # Use exact position (no randomization)
                x = center_x
                y = center_y
            else:
                # Use random position in circle around center
                theta = np.random.uniform(0, 2 * np.pi)
                rho = max_radius * np.sqrt(np.random.uniform(0, 1))
                x = rho * np.cos(theta) + center_x
                y = rho * np.sin(theta) + center_y
            
            self.set_goal_pose(name, [x, y, height], quat)
            
            # Reset velocity to zero
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id != -1:
                joint_id = self.model.body_jntadr[body_id]
                qveladr = self.model.jnt_dofadr[joint_id]
                self.data.qvel[qveladr:qveladr + 6] = 0.0
            
            return x, y, height

        # Reset bottle on the desk within mesh bounds
        x_min, x_max, y_min, y_max = self.desk_xy_bounds
        # import pdb; pdb.set_trace()
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        z = self.desk_top_z
        reset_object(self.bottle_body_name, x, y, z, 0.0)


    def _get_rgb_observation(self, camera_name):
        """Get RGB camera observation."""
        # Get camera ID
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id == -1:
            raise ValueError(f"Camera '{camera_name}' not found")
        
        # Use persistent renderer for efficiency
        if self._renderer is None:
            try:
                self._renderer = mujoco.Renderer(self.model, height=self.camera_height, width=self.camera_width)
            except Exception as e:
                print(f"Warning: Could not create renderer: {e}")
                # Return dummy RGB array if rendering fails
                return np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        
        try:
            self._renderer.update_scene(self.data, camera=camera_id)
            rgb_array = self._renderer.render()
            return rgb_array.astype(np.uint8)
        except Exception as e:
            print(f"Warning: Rendering failed: {e}")
            # Return dummy RGB array if rendering fails
            return np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
    
    def _get_state_observation(self):
        """Get state: base (x,y) + 6 arm joints + gripper normalized to 0-1."""
        base_xy = np.asarray(self.data.body("base_link").xpos[:2], dtype=np.float32)
        joint_positions = np.asarray(
            [self.data.joint(name).qpos[0] for name in self.state_joint_names],
            dtype=np.float32,
        )
        gripper_qpos = float(self.data.joint(self.arm_joint_names[6]).qpos[0])
        gripper_norm = np.clip(gripper_qpos / 0.045, 0.0, 1.0)
        return np.concatenate([base_xy, joint_positions, np.array([gripper_norm], dtype=np.float32)], axis=0)

    def _get_observation(self):
        """Get mixed observation: RGB cameras + state for left arm."""
        state_obs = self._get_state_observation()
        ee_pos, _ = self._get_site_pos_ori(self.ee_site_name)
        grasp_pos, _ = self._get_site_pos_ori(self.grasp_site_name)
        target = grasp_pos - ee_pos
        
        obs = {
            'state': state_obs,
            'target': target,
        }

        if self.gs_render:
            if self.use_top_rgb:
                obs['top_rgb'] = self.get_img("top")
            if self.use_left_wrist_rgb:
                obs['left_wrist_rgb'] = self.get_img("wrist_cam")
            if self.use_right_wrist_rgb:
                obs['right_wrist_rgb'] = self.get_img("right_wrist_cam")

            # Only show cv2 windows when rendering is enabled
            if self.visualization:
                if self.use_top_rgb:
                    bgr_img_3rd = cv2.cvtColor(obs['top_rgb'], cv2.COLOR_RGB2BGR)
                    cv2.imshow("3rd Person View", bgr_img_3rd)
                if self.use_left_wrist_rgb:
                    bgr_img_wrist = cv2.cvtColor(obs['left_wrist_rgb'], cv2.COLOR_RGB2BGR)
                    cv2.imshow("Left Wrist Camera View", bgr_img_wrist)
                if self.use_right_wrist_rgb:
                    bgr_img_right = cv2.cvtColor(obs['right_wrist_rgb'], cv2.COLOR_RGB2BGR)
                    cv2.imshow("Right Wrist Camera View", bgr_img_right)
                cv2.waitKey(1)
        else:
            if self.use_top_rgb:
                obs['top_rgb'] = self._get_rgb_observation("top")
            if self.use_left_wrist_rgb:
                obs['left_wrist_rgb'] = self._get_rgb_observation("wrist_cam")
            if self.use_right_wrist_rgb:
                obs['right_wrist_rgb'] = self._get_rgb_observation("right_wrist_cam")

        return obs

    def _check_contact_between_bodies(self, body1_name: str, body2_name: str) -> tuple[bool, float]:
        """Check contact between two bodies."""
        body1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_name)
        body2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_name)
        
        if body1_id == -1 or body2_id == -1:
            return False, 0.0
            
        total_force = 0.0
        contact_found = False
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            geom1_body = self.model.geom_bodyid[geom1_id]
            geom2_body = self.model.geom_bodyid[geom2_id]
            
            if ((geom1_body == body1_id and geom2_body == body2_id) or 
                (geom1_body == body2_id and geom2_body == body1_id)):
                contact_found = True
                contact_force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, contact_force)
                force_magnitude = np.linalg.norm(contact_force[:3])
                total_force += force_magnitude
                
        return contact_found, total_force

    def _check_robot_table_contact(self) -> tuple[bool, float]:
        """Check contact between robot bodies and the desk."""
        total_force = 0.0
        contact_found = False
        for name in self.robot_body_names:
            contact, force = self._check_contact_between_bodies(name, self.desk_body_name)
            if contact:
                contact_found = True
                total_force += force
        return contact_found, total_force

    def _check_gripper_contact_with_object(self, object_name) -> bool:
        """Check if gripper fingers contact the object."""
        contact_found = 0
        for link_name in ["link7", "link8"]:
            collision_contact, _ = self._check_contact_between_bodies(link_name, object_name)
            if collision_contact:
                contact_found += 1
        return contact_found >= 2

    def _check_object_fell_off_table(self, object_name) -> bool:
        """Check if object left the desk bounds or fell below the table."""
        pos, _ = self._get_body_pose(object_name)
        x_min, x_max, y_min, y_max = self.desk_xy_bounds
        if pos[2] < (self.desk_top_z - 0.05):
            return True
        if pos[0] < x_min or pos[0] > x_max or pos[1] < y_min or pos[1] > y_max:
            return True
        return False

    def _check_object_tipped(self, object_name) -> bool:
        """Check if object is tipped beyond the upright threshold."""
        _, quat_wxyz = self._get_body_pose(object_name)
        # scipy expects xyzw
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
        rot = R.from_quat(quat_xyzw).as_matrix()
        up_world = rot @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return float(up_world[2]) < self.bottle_upright_cos

    def _compute_gripper_action_penalty(self, action):
        """No penalty since gripper actions are now binary (open/close only)."""
        return 0.0

    def _compute_reward(self):
        """Staged reward: reach -> lift (all non-negative)."""
        reward = 0.0

        # --- Stage 1: Reach (base + ee) ---
        ee_pos, _ = self._get_site_pos_ori(self.ee_site_name)
        grasp_pos, _ = self._get_site_pos_ori(self.grasp_site_name)
        bottle_pos, _ = self._get_body_pose(self.bottle_body_name)
        # print('=========== bottle_z:', bottle_pos[2],"===========")
        base_xy = np.asarray(self.data.body("base_link").xpos[:2], dtype=np.float32)
        target_xy = grasp_pos[:2]
        

        # ===== 1. reaching 奖励（矩形 + 朝向）=====
        x_min = target_xy[0] + self.base_reach_x_range[0]
        x_max = target_xy[0] + self.base_reach_x_range[1]
        y_min = target_xy[1] + self.base_reach_y_range[0]
        y_max = target_xy[1] + self.base_reach_y_range[1]

        # 1.1 到矩形到达区的最短距离（在区内为 0）
        dx = max(x_min - base_xy[0], 0.0, base_xy[0] - x_max)
        dy = max(y_min - base_xy[1], 0.0, base_xy[1] - y_max)
        dist_to_box = float(np.hypot(dx, dy))
        base_in_box = (x_min <= base_xy[0] <= x_max) and (y_min <= base_xy[1] <= y_max)         # 判断底盘是否在到达区内

        # 1.2 底盘位置奖励：越接近到达区越高（指数型）
        exp_max = float(np.exp(-self.base_reach_exp_k * self.base_reach_max_dist))
        base_pos_reward = (np.exp(-self.base_reach_exp_k * dist_to_box) - exp_max) / (1.0 - exp_max + 1e-6)
        base_pos_reward = float(np.clip(base_pos_reward, 0.0, 1.0))                              # wandb

        # 1.3 底盘进步奖励：距离持续变小才给
        if self.prev_dist_to_box is None:
            base_progress_reward = 0.0
        else:
            base_progress_reward = max(self.prev_dist_to_box - dist_to_box, 0.0) * self.base_progress_scale
        self.prev_dist_to_box = dist_to_box

        # 1.4 底盘朝向奖励：当前朝向计算下，越接近 0 越“正对”
        to_target = target_xy - base_xy
        to_target_norm = float(np.linalg.norm(to_target))
        if to_target_norm < 1e-6:
            base_angle_reward = 1.0
            facing_cos = 1.0
        else:
            base_xmat = np.asarray(self.data.body("base_link").xmat, dtype=np.float32).reshape(3, 3)
            heading_xy = base_xmat[:2, 0]  # 取底盘 x 轴作为朝向
            heading_norm = float(np.linalg.norm(heading_xy)) + 1e-6
            heading_xy = heading_xy / heading_norm
            dir_xy = to_target / to_target_norm
            facing_cos = float(np.dot(heading_xy, dir_xy))
            base_angle_reward = 1.0 - abs(facing_cos)
            base_angle_reward = float(np.clip(base_angle_reward, 0.0, 1.0))                    # wandb

        # 1.5 底盘 reaching 奖励：位置 + 朝向 + 进步
        reach_base = base_pos_reward * (0.5 + 0.5 * base_angle_reward) + base_progress_reward  # wandb

        # 1.6 夹爪接近奖励：距离越小越高（指数型且连续）
        dist_ee = float(np.linalg.norm(ee_pos - grasp_pos))
        exp_max = float(np.exp(-self.reach_ee_exp_k * self.reach_ee_max_dist))
        reach_ee = (np.exp(-self.reach_ee_exp_k * dist_ee) - exp_max) / (1.0 - exp_max + 1e-6)
        reach_ee = float(np.clip(reach_ee, 0.0, 1.0))
        # 1.7 到达前保持夹爪打开：未到达时用张开比例调节奖励
        gripper_qpos = float(self.data.joint(self.arm_joint_names[6]).qpos[0])
        gripper_open_ratio = float(np.clip(gripper_qpos / 0.045, 0.0, 1.0))
        if dist_ee > self.reach_ee_thresh:
            reach_ee *= gripper_open_ratio
        reach_ee *= self.reach_ee_scale

        # 1.8 夹爪朝向奖励：接近时让夹爪 z 轴指向瓶子
        reach_ee_orient = 0.0
        if dist_ee < self.ee_orient_dist:
            ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name)
            if ee_site_id != -1:
                ee_xmat = np.asarray(self.data.site(ee_site_id).xmat, dtype=np.float32).reshape(3, 3)
                ee_z_axis = ee_xmat[:, 2]
                ee_z_axis = ee_z_axis / (float(np.linalg.norm(ee_z_axis)) + 1e-6)
                to_grasp = grasp_pos - ee_pos
                to_grasp_norm = float(np.linalg.norm(to_grasp))
                if to_grasp_norm > 1e-6:
                    dir_xyz = to_grasp / to_grasp_norm
                    orient_cos_z = float(np.dot(ee_z_axis, dir_xyz))
                    reach_ee_orient = self.ee_orient_scale * max(orient_cos_z, 0.0)
                    # print('=========== ee_orient_cos_z:', orient_cos_z,"===========")


        # 1.9 总 reaching 奖励
        reaching = reach_base + reach_ee + reach_ee_orient  # [0, 2]                             # wandb
        reward += reaching


        # ===== 2. grasp_ready 奖励（矩形 + 朝向）=====

        v_base = float(np.linalg.norm(self.data.qvel[0:3]))
        omega_base = float(np.linalg.norm(self.data.qvel[3:6]))

        # 末端接近时更严格，远距离时更宽松
        v_max = self.base_v_max_near            if dist_ee < 0.05 else self.base_v_max_far
        omega_max = self.base_omega_max_near    if dist_ee < 0.05 else self.base_omega_max_far
        v_ok = v_base < v_max
        omega_ok = omega_base < omega_max
        base_ready = base_in_box and v_ok and omega_ok  # wandb

        gripper_open = gripper_qpos > self.gripper_open_threshold                                       # wandb    
        ee_ready = dist_ee < self.ee_pos_tol
 
        base_ready_reward = 0.0                                                                          # wandb
        ee_ready_reward = self.grasp_ready_bonus if (base_ready and ee_ready) else 0.0                   # wandb
        reward += ee_ready_reward

        grasp_ready = base_ready and gripper_open and ee_ready

        # debug check

        if self.step_number % 10 == 0:
            print(
                f"[grasp_ready_debug] step={self.step_number} "
                f"base_ready={base_ready} ee_ready={ee_ready} gripper_open={gripper_open} "
                f"dist_ee={dist_ee:.3f} gripper_qpos={gripper_qpos:.3f} "
                f"base_in_box={base_in_box} v_ok={v_ok} omega_ok={omega_ok} "
                f"v_base={v_base:.3f}/{v_max:.3f} omega_base={omega_base:.3f}/{omega_max:.3f}"
            )
        if self.step_number % 10 == 0:
            print(
                f"[base_ready_debug] step={self.step_number} "
                f"base_xy=({base_xy[0]:.3f},{base_xy[1]:.3f}) "
                f"box_x=[{x_min:.3f},{x_max:.3f}] box_y=[{y_min:.3f},{y_max:.3f}] "
                f"base_in_box={base_in_box} v_ok={v_ok} omega_ok={omega_ok} "
                f"v_base={v_base:.3f}/{v_max:.3f} omega_base={omega_base:.3f}/{omega_max:.3f}"
            )

        # ===== 3. grasp 成功与过程奖励（需要 base_ready + ee_ready） =====
        both_fingers_contact = self._check_gripper_contact_with_object(self.bottle_body_name)
        if base_ready and ee_ready and both_fingers_contact:
            self.contact_streak += 1
        else:
            self.contact_streak = 0
        self.max_contact_streak = max(self.max_contact_streak, self.contact_streak)

        gripper_closed = gripper_qpos < self.gripper_close_threshold                                                            # 夹爪是否闭合到阈值
        contact_ratio = float(np.clip(self.contact_streak / self.grasp_contact_steps, 0.0, 1.0))                                # 接触步数归一化
        close_ratio = float(np.clip((self.gripper_close_threshold - gripper_qpos) / self.gripper_close_threshold, 0.0, 1.0))    # 闭合程度归一化
        grasp_progress = contact_ratio * close_ratio                                                                            # 接触+闭合的进度
        catch_reward = self.grasp_progress_scale * grasp_progress if (base_ready and ee_ready) else 0.0                         # 仅门控通过才给奖励
        reward += catch_reward

        catch_success = (
            base_ready
            and ee_ready
            and both_fingers_contact
            and gripper_closed
            and (self.contact_streak >= self.grasp_contact_steps)
        )

        # ===== 4. lifting 奖励（需要 catch_success）=====
        dz = bottle_pos[2] - self.desk_top_z
        # 4.1 连续抬起奖励（仅在抓稳后）
        lift_progress = float(np.clip((dz - self.lift_start) / (self.lift_target - self.lift_start), 0.0, 1.0))
        lifting = self.lift_reward_scale * lift_progress if catch_success else 0.0
        reward += lifting

        # 4.2 成功抬起奖励（每步给，需抓稳）
        lift_success_bonus = self.lift_success_bonus if (catch_success and dz >= self.lift_target) else 0.0
        reward += lift_success_bonus

        # Success: bottle lifted above target height
        if catch_success and (dz >= self.lift_target):
            self.goal_reached = True

        # ===== 5. 惩罚项 =====
        object_fell = self._check_object_fell_off_table(self.bottle_body_name)
        object_tipped = self._check_object_tipped(self.bottle_body_name)
        table_contact, _ = self._check_robot_table_contact()
        if table_contact:
            self.table_contact_count += 1
            # print('=== Robot-table contact count:', self.table_contact_count, '===')
        table_contact_terminated = self.table_contact_count >= self.table_contact_terminate_steps

        penalties = 0.0
        if object_fell:
            penalties -= self.penalty_object_fell
        if object_tipped:
            penalties -= self.penalty_object_tipped
        if table_contact:
            penalties -= self.penalty_table_contact

        # base_ready 前闭合夹爪惩罚
        if (not base_ready) and (gripper_qpos < self.gripper_close_threshold):
            penalties -= self.penalty_early_close
        # 抓取/抬起阶段底盘乱动惩罚
        if (grasp_ready or catch_success) and (not base_ready):
            penalties -= self.penalty_base_motion

        # 时间惩罚
        penalties -= self.penalty_time
        reward += penalties

        reward_dict = {
            "reaching": reaching,
            "reach_base": reach_base,
            "reach_base_pos": base_pos_reward,
            "reach_base_angle": base_angle_reward,
            "reach_base_progress": base_progress_reward,
            "reach_base_in_box": float(base_in_box),
            "reach_ee": reach_ee,
            "reach_ee_orient": reach_ee_orient,
            "gripper_open_ratio": gripper_open_ratio,
            "gripper_open": float(gripper_open),
            "grasp_ready": float(grasp_ready),
            "base_ready_reward": base_ready_reward,
            "ee_ready_reward": ee_ready_reward,
            "both_fingers_contact": float(both_fingers_contact),
            "gripper_closed": float(gripper_closed),
            "contact_streak": float(self.contact_streak),
            "grasp_progress": grasp_progress,
            "catch_reward": catch_reward,
            "catch_success": float(catch_success),
            "lifting": lifting,
            "lift_progress": lift_progress,
            "lift_success_bonus": lift_success_bonus,
            "object_fell": float(object_fell),
            "object_tipped": float(object_tipped),
            "table_contact": float(table_contact),
            "table_contact_count": float(self.table_contact_count),
            "table_contact_terminated": float(table_contact_terminated),
            "penalties": penalties,
        }
        return reward, reward_dict

    def step(self, action):
        """Execute one environment step for left arm only."""
        base_cmd = np.asarray(action[:3], dtype=np.float32) * self.base_cmd_scale   # 这里的动作范围都是 [-1, 1]
        delta_action = self.map_action_to_joint_deltas(action[3:])
        
        # Get current joint positions for left arm
        current_qpos = np.asarray(
            [self.data.joint(name).qpos[0] for name in self.arm_joint_names],
            dtype=np.float32,
        )
        control_indices = slice(8, 15)                                              # Control indices for single arm
        new_qpos = self.apply_joint_deltas_with_limits(current_qpos, delta_action)  # x_new = x_current + delta_x
        self.data.ctrl[control_indices] = new_qpos                                  # 机械臂控制目标
        
        for i in range(20):
            # Update base control each sub-step to keep PID stable.
            self.base_controller.apply(base_cmd[0], base_cmd[1], base_cmd[2])       # vx vy wz
            mujoco.mj_step(self.model, self.data)                                   # 模拟一步
            
            # Render if viewer is available
            if self.visualization and self.handle:
                self.handle.sync()

        if self.gs_render:
            self.update_gs_scene()

        self.step_number += 1
        
        observation = self._get_observation()
        reward, reward_dict = self._compute_reward()
        
        # Check termination conditions
        object_fell = bool(reward_dict.get("object_fell", 0.0))
        object_tipped = bool(reward_dict.get("object_tipped", 0.0))
        table_contact = bool(reward_dict.get("table_contact", 0.0))
        table_contact_terminated = bool(reward_dict.get("table_contact_terminated", 0.0))
        terminated = self.goal_reached or object_fell or object_tipped or table_contact_terminated
        truncated = self.step_number >= self.episode_len

        terminated_reasons = []
        if self.goal_reached:
            terminated_reasons.append("success")
        if object_fell:
            terminated_reasons.append("object_fell")
        if object_tipped:
            terminated_reasons.append("object_tipped")
        if table_contact_terminated:
            terminated_reasons.append("table_contact")


        info = {
            'is_success': self.goal_reached,
            'total_reward': reward,
            'step_number': self.step_number,
            'current_qpos': new_qpos.copy(),
            'delta_action': delta_action.copy(),
            'terminated_reasons': terminated_reasons,
            'reward_info': {
                'reaching': float(reward_dict.get("reaching", 0.0)),
                'reach_base': float(reward_dict.get("reach_base", 0.0)),
                'reach_base_pos': float(reward_dict.get("reach_base_pos", 0.0)),
                'reach_base_angle': float(reward_dict.get("reach_base_angle", 0.0)),
                'reach_base_progress': float(reward_dict.get("reach_base_progress", 0.0)),
                'reach_ee': float(reward_dict.get("reach_ee", 0.0)),
                'reach_ee_orient': float(reward_dict.get("reach_ee_orient", 0.0)),
                'gripper_open_ratio': float(reward_dict.get("gripper_open_ratio", 0.0)),
                'grasp_ready': bool(reward_dict.get("grasp_ready", 0.0)),
                'base_ready_reward': float(reward_dict.get("base_ready_reward", 0.0)),
                'ee_ready_reward': float(reward_dict.get("ee_ready_reward", 0.0)),
                'catch_reward': float(reward_dict.get("catch_reward", 0.0)),
                'catch_success': bool(reward_dict.get("catch_success", 0.0)),
                'lifting': float(reward_dict.get("lifting", 0.0)),
                'lift_success_bonus': float(reward_dict.get("lift_success_bonus", 0.0)),
            },
            'penalty_info': {
                'object_fell': object_fell,
                'object_tipped': object_tipped,
                'table_contact': table_contact,
                'table_contact_count': int(reward_dict.get("table_contact_count", 0)),
                'penalties': float(reward_dict.get("penalties", 0.0)),
            },
        }

        return observation, reward, terminated, truncated, info

    def get_img(self, camera_name):
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id == -1:
            raise ValueError(f"Camera '{camera_name}' not found")
        # print(f"Camera ID: {cam_id}, Name: {cam_name}")
        cam_pos = self.data.cam_xpos[camera_id]
        cam_rot = self.data.cam_xmat[camera_id].reshape((3, 3))
        cam_quat = R.from_matrix(cam_rot).as_quat()

        # 设置gs相机参数，渲染图像
        self.gs_renderer.set_camera_fovy(self.rgb_fovy * np.pi / 180.)
        self.gs_renderer.set_camera_pose(cam_pos, cam_quat)
        with torch.inference_mode():
            rgb_img = self.gs_renderer.render()


        if isinstance(rgb_img, torch.Tensor):
            rgb_img = rgb_img.detach().cpu().numpy()

        # 如果是 [3, H, W] 格式，转成 [H, W, 3]
        if rgb_img.shape[0] == 3 and len(rgb_img.shape) == 3:
            rgb_img = np.transpose(rgb_img, (1, 2, 0))

        # 确保值在 [0,1] 范围，并转换为 uint8
        rgb_img = np.clip(rgb_img, 0, 1)
        rgb_img = (rgb_img * 255).astype(np.uint8)
        return rgb_img

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def close(self):
        """Clean up resources"""
        if self._renderer is not None:
            try:
                self._renderer.close()
                self._renderer = None
            except Exception as e:
                print(f"Warning: Error closing renderer: {e}")
        
        # Close MuJoCo viewer if it exists
        if hasattr(self, 'handle') and self.handle is not None:
            try:
                self.handle.close()
                self.handle = None
                
                # Force GLFW cleanup to prevent segfaults when switching between GUI environments
                try:
                    import glfw
                    if glfw.get_current_context():
                        glfw.terminate()
                        # Small delay to ensure cleanup
                        import time
                        time.sleep(0.1)
                except ImportError:
                    # glfw not available, skip
                    pass
                except Exception as e:
                    print(f"Warning: GLFW cleanup error: {e}")
                    
            except Exception as e:
                print(f"Warning: Error closing MuJoCo viewer: {e}")
            
        super().close()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()


def make_env():
    """Factory function to create PiperEnv for left arm fruit picking."""
    return PiperEnv(visualization=False)

import cv2
import numpy as np
if __name__ == "__main__":

    env = PiperEnv(visualization=True)  # Left arm only
 

    print("=== Left Arm Fruit Picking Task ===")
    obs, info = env.reset()
    print(f"  Action space: {env.action_space}")
    print(f"Observation state shape: {obs['state'].shape}")

    while True:
        # action = env.action_space.sample()
        action = [0.0] * 10
        # print(f"Sampled action: {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        # print(f'state = {obs["state"]}')
        # print(f"reward={reward:.3f}")

    # print("Demo completed successfully!")
    # env.close()
