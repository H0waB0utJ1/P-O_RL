import torch

class PPOBuffer:
    def __init__(self, obs_len=1, act_len=1, gamma=0.99, lam=0.95, size=1):
        self.states = torch.zeros(size, obs_len, dtype=torch.float32)
        self.actions = torch.zeros(size, act_len, dtype=torch.float32)
        self.rewards = torch.zeros(size, 1, dtype=torch.float32)
        self.values = torch.zeros(size, 1, dtype=torch.float32)
        self.advantages = torch.zeros(size, 1, dtype=torch.float32)
        self.dones = torch.zeros(size, 1, dtype=torch.float32)
        self.log_probs = torch.zeros(size, 1, dtype=torch.float32)

        self.gamma, self.lam = gamma, lam
        self.ptr = 0
        self.traj_idx = [0] # 轨迹首位索引

    def __len__(self):
        return self.ptr

    def store(self, state, action, reward, value, done, log_prob):
        self.states[self.ptr]= state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.ptr += 1

    def finish_path(self, last_val): # GAE计算
        path_start = self.traj_idx[-1] # 取当前轨迹首位
        path_end   = self.ptr # 取下一轨迹首位
        self.traj_idx.append(self.ptr) # 记录下一轨迹首位

        rewards = self.rewards[path_start:path_end, 0] # 提取该轨迹每步的奖励值
        values  = self.values[path_start:path_end, 0] # 提取该轨迹每步的价值
        advantages = torch.zeros_like(rewards) # 创建容器存储该轨迹每步的GAE优势
        gae = 0 # 该轨迹每步的GAE优势
        
        for t in reversed(range(path_end - path_start)): # 反向循环 path_end - path_start 次，递推计算各项GAE优势
            next_value = last_val if t == (path_end - path_start - 1) else values[t + 1] # 轨迹的最后一步的last_val要进行特殊处理
            delta = rewards[t] + self.gamma * next_value - values[t] # TD残差
            gae = delta + self.gamma * self.lam * gae # GAE优势
            advantages[t] = gae
        
        self.advantages[path_start:path_end, 0] = advantages # 将该轨迹的优势值存入存储器对应位置

    def get_data(self):
        ep_lens = [j - i for i, j in zip(self.traj_idx, self.traj_idx[1:])] # 每条轨迹的长度
        ep_rewards = [float(sum(self.rewards[int(i):int(j)])) for i, j in zip(self.traj_idx, self.traj_idx[1:])] # 每条轨迹的总奖励值
        data = {
            'states': self.states[:self.ptr],         # 每一步的obs
            'actions': self.actions[:self.ptr],       # 每一步的action
            'rewards': self.rewards[:self.ptr],       # 每一步的reward
            'values': self.values[:self.ptr],         # 每一步的value
            'advantages': self.advantages[:self.ptr], # 每一步的advantage
            'dones': self.dones[:self.ptr],           # 每一步的done
            'log_probs': self.log_probs[:self.ptr],   # 每个动作的对数概率
            'traj_idx': torch.Tensor(self.traj_idx),  # 每一条轨迹的首位索引
            'ep_lens': torch.Tensor(ep_lens),         # 每一条轨迹的长度
            'ep_rewards': torch.Tensor(ep_rewards),   # 每一条轨迹的总奖励值
        }
        return data