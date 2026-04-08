import sys
import ray
import time
import datetime
import numpy as np

import torch
from pathlib import Path
from copy import deepcopy
import torch.optim as optim
from collections import defaultdict
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from rl.ppo.ppo_config import PPOCfg
from rl.ppo.ppobuffer import PPOBuffer
from rl.policies.critic import FF_V, LSTM_V
from rl.policies.actor import Gaussian_FF_Actor, Gaussian_LSTM_Actor

@ray.remote
def sample(env_fn, actor, critic, gamma, lam, max_traj_len, max_n_traj, deterministic): # 单环境采样
    env = env_fn() # 环境创建

    memory = PPOBuffer(actor.state_dim, actor.action_dim, gamma, lam, max_traj_len*max_n_traj) # 采样数据存储器创建
    n_traj = 0

    term_sum = defaultdict(float)
    term_count = 0

    while n_traj < max_n_traj:
        n_traj += 1
        state = torch.tensor(env.reset(), dtype=torch.float) # 单轨迹初始状态采样
        done = False # 单轨迹结束标识
        traj_len = 0 # 单轨迹长度

        # 隐藏状态初始化
        if hasattr(actor, 'init_hidden_state'): 
            actor.init_hidden_state()
        if hasattr(critic, 'init_hidden_state'):
            critic.init_hidden_state()

        info = None

        while not done and traj_len < max_traj_len: # 单轨迹采样
            with torch.no_grad():
                action, log_prob = actor(state, deterministic=deterministic, return_log_prob=True) # 动作生成
                value = critic(state) # 价值生成

            next_state, reward, done, info = env.step(action.detach().cpu().numpy()) # 环境交互
            reward = torch.tensor(reward, dtype=torch.float) # 转为张量
            memory.store(state, action, reward, value, float(done), log_prob) # 采样数据存储
            state = torch.tensor(next_state, dtype=torch.float) # 更新状态
            
            traj_len += 1

            '''
            with torch.no_grad():
                action, log_prob = actor(state, deterministic=deterministic, return_log_prob=True) # 动作生成

            step = 0
            while not done and step < 4:
                with torch.no_grad():
                    value = critic(state) # 价值生成

                next_state, reward, done, info = env.step(action.detach().cpu().numpy()) # 环境交互
                reward = torch.tensor(reward, dtype=torch.float) # 转为张量
                memory.store(state, action, reward, value, float(done), log_prob) # 采样数据存储
                state = torch.tensor(next_state, dtype=torch.float) # 更新状态
                
                step += 1
                traj_len += 1

            with torch.no_grad():
                action, _ = actor(state, deterministic=deterministic, return_log_prob=True) # 动作生成

            step = 0
            while not done and step < 4:
                with torch.no_grad():
                    value = critic(state)
                    
                    # 用当前 state 重新评估log_prob
                    dist = actor.distribution(state)
                    log_prob = dist.log_prob(action).sum(-1)

                next_state, reward, done, info = env.step(action.detach().cpu().numpy()) # 环境交互
                reward = torch.tensor(reward, dtype=torch.float) # 转为张量
                memory.store(state, action, reward, value, float(done), log_prob) # 采样数据存储
                state = torch.tensor(next_state, dtype=torch.float) # 更新状态

                step += 1
                traj_len += 1
            '''
            
        # 单轨迹结束，进行轨迹尾处理并计算GAE
        with torch.no_grad():
            value = critic(state) # 轨迹尾处理
        memory.finish_path(last_val=(not done) * value) # GAE计算

        if info is not None and traj_len > 0:
            for k, v in dict(info).items():
                term_sum[k] += float(v)
            term_count += traj_len

    stats = {"term_sum": dict(term_sum), "term_count": int(term_count)}
    return memory.get_data(), stats

class PPO:
    def __init__(self, env_fn, args):
        self.gamma          = args.gamma          # 折扣因子
        self.lam            = args.lam            # GAE的λ
        self.lr             = args.lr             # 学习率
        self.eps            = args.eps
        self.ent_coeff      = args.entropy_coeff  # 熵项系数
        self.clip           = args.clip           # 裁减系数
        self.minibatch_size = args.minibatch_size # 每批样本数（使用循环神经网络时为轨迹数）
        self.epochs         = args.epochs         # 一批采样数据的重复更新次数
        self.max_traj_len   = args.max_traj_len   # 单episode的最大步长
        self.max_n_traj     = args.max_n_traj     # 单环境采样轨迹数
        self.n_proc         = args.num_procs      # 并行环境数量
        self.grad_clip      = args.max_grad_norm  # 梯度裁减阈值
        self.eval_freq      = args.eval_freq      # 评估周期
        self.recurrent      = args.recurrent      # 是否适用循环神经网络

        self.total_steps = 0 # 计数器
        self.highest_reward = -np.inf # 历史最优成绩
        self.iteration_count = 0 # 迭代编号

        self.save_path = Path(args.logdir) # 保存路径
        Path.mkdir(self.save_path, parents=True, exist_ok=True) # 创建文件夹
        self.writer = SummaryWriter(log_dir=self.save_path, flush_secs=10) # 日志记录器

        obs_dim = PPOCfg.dim_obs # 状态空间维度
        action_dim = PPOCfg.dim_actions # 动作空间维度

        if args.continued: # 根据已有模型继续训练
            path_to_actor = args.continued # actor文件地址
            path_to_critic = Path(args.continued.parent, "critic" + str(args.continued).split('actor')[1]) # 根据actor文件地址解码出critic文件地址
            actor = torch.load(path_to_actor, weights_only=False) # 加载actor网络
            critic = torch.load(path_to_critic, weights_only=False) # 加载critic网络
            if args.learn_std: # actor方差设置
                actor.stds = torch.nn.Parameter(args.std_dev * torch.ones(action_dim))
            else:
                actor.register_buffer('stds', args.std_dev * torch.ones(action_dim))
            print("Loaded (pre-trained) actor from: ", path_to_actor)
            print("Loaded (pre-trained) critic from: ", path_to_critic)
        else: # 无基础模型
            if args.recurrent: # 采用循环神经网络
                actor = Gaussian_LSTM_Actor(obs_dim, action_dim, init_std=args.std_dev, learn_std=args.learn_std, normc_init=True, bounded=False)
                critic = LSTM_V(obs_dim, normc_init=True)
            else: # 采用非循环神经网络
                actor = Gaussian_FF_Actor(obs_dim, action_dim, init_std=args.std_dev, learn_std=args.learn_std, normc_init=True, bounded=False)
                critic = FF_V(obs_dim, normc_init=True)

        self.actor = actor
        self.critic = critic

    @staticmethod
    def save(nets, save_path, suffix=""):
        filetype = ".pt"
        for name, net in nets.items():
            path = Path(save_path, name + suffix + filetype)
            torch.save(net, path)
            print("Saved {} at {}".format(name, path))
        return
    
    def sample_parallel(self, *args, deterministic=False):
        worker_args = (self.gamma, self.lam, self.max_traj_len, self.max_n_traj, deterministic)
        args = args + worker_args

        worker = sample
        workers = [worker.remote(*args) for _ in range(self.n_proc)]
        results = ray.get(workers)

        # 兼容 worker 返回 (data, stats) 或仅 data
        if isinstance(results[0], (tuple, list)) and len(results[0]) == 2:
            data_list  = [r[0] for r in results]  # memory.get_data()
            stats_list = [r[1] for r in results]  # {"term_sum":..., "term_count":...}
        else:
            data_list  = results
            stats_list = None

        #keys = list(results[0].keys())
        keys = list(data_list[0].keys())
        aggregated = {}
        #state_counts = [r['states'].shape[0] for r in results] # 各环境采集的数据量
        state_counts = [r['states'].shape[0] for r in data_list] # 各环境采集的数据量
        offsets = [0] # 偏移列表
        for cnt in state_counts[:-1]:
            offsets.append(offsets[-1] + cnt) # 记录各环境偏移量

        for k in keys: # 数据拼接！！！
            if k == 'traj_idx': # 对轨迹首位索引进行偏移拼接
                corrected = []
                #for idx, (off, r) in enumerate(zip(offsets, results)):
                for idx, (off, r) in enumerate(zip(offsets, data_list)):
                    ti = r['traj_idx'].to(torch.long)
                    # 将原轨迹首位索引偏移为拼接后轨迹首位索引
                    if idx == 0:
                        corrected.append(ti + off)
                    else: # 上一环境的轨迹索引已经包含该环境第一条轨迹的起始索引
                        corrected.append(ti[1:] + off)
                aggregated[k] = torch.cat(corrected) # 拼接为一个张量（最后一个环境包含了一个多余的轨迹首位索引）
            else:
                try:
                    #aggregated[k] = torch.cat([r[k] for r in results], dim=0) # 拼接为一个张量
                    aggregated[k] = torch.cat([r[k] for r in data_list], dim=0) # 拼接为一个张量
                except Exception as e:
                    #if isinstance(results[0][k], list):
                    if isinstance(data_list[0][k], list):
                        #aggregated[k] = [elem for r in results for elem in r[k]] # 拼接为一个列表
                        aggregated[k] = [elem for r in data_list for elem in r[k]] # 拼接为一个列表
                    else:
                        raise

        # stats 聚合（每轮采样每项奖励单步均值）
        reward_terms_mean = None
        if stats_list is not None:
            global_sum = defaultdict(float)
            global_count = 0

            for st in stats_list:
                # st: {"term_sum": {name: sum_scaled}, "term_count": int}
                for name, v in st["term_sum"].items():
                    global_sum[name] += float(v)
                global_count += int(st["term_count"])

            reward_terms_mean = {name: global_sum[name] / max(1, global_count) for name in global_sum}
            aggregated["reward_terms_mean"] = reward_terms_mean # 放进aggregated

        class Data: # 创建数据类
            def __init__(self, data):
                for key, val in data.items():
                    setattr(self, key, val)
        return Data(aggregated)

    def update_policies(self, obs_batch, action_batch, return_batch, advantage_batch, old_log_prob_batch, mask): # 策略网络更新
        # 初始化隐藏层状态
        if hasattr(self.actor, "init_hidden_state"):
            self.actor.init_hidden_state(batch_size=obs_batch.shape[1])
        if hasattr(self.critic, "init_hidden_state"):
            self.critic.init_hidden_state(batch_size=obs_batch.shape[1])

        # 计算ratio
        pdf = self.actor.distribution(obs_batch) # 计算当前策略的动作概率分布（t, b, a）
        log_probs = pdf.log_prob(action_batch).sum(-1, keepdim=True) # 动作维度的对数概率相加（log空间，乘积变求和）（t, b, 1）
        
        old_log_probs = old_log_prob_batch # （t, b, 1）

        ratio = (log_probs - old_log_probs).exp() # 重要性采样权重：新旧策略概率比（log空间相除变相减）
        with torch.no_grad(): # 计算kl散度（不带梯度）
            log_ratio = log_probs - old_log_probs
            approx_kl_div = ((ratio - 1) - log_ratio)[mask > 0].mean()

        # 计算损失
        cpi_loss = ratio * advantage_batch * mask
        clip_loss = ratio.clamp(1.0 - self.clip, 1.0 + self.clip) * advantage_batch * mask
        values = self.critic(obs_batch) # 计算价值（带梯度）
        if isinstance(mask, torch.Tensor): # 采用循环神经网络
            mask_sum = mask.sum()
            actor_loss = -torch.min(cpi_loss, clip_loss).sum() / mask_sum
            entropy = pdf.entropy().mean(-1, keepdim=True)  # 平均每个关节
            entropy_penalty = -(entropy * mask).sum() / mask_sum # 熵项损失
            clip_fraction = (((torch.abs(ratio - 1) > self.clip).float() * mask).sum() / mask_sum).item() # clip触发比例计算
            critic_loss = ((return_batch - values) ** 2 * mask).sum() / mask_sum # 计算critic_loss
        else: # 采用非循环神经网络
            actor_loss = -torch.min(cpi_loss, clip_loss).mean()
            entropy = pdf.entropy().mean(-1)  # 平均每个关节
            entropy_penalty = -entropy.mean() # 熵项损失
            clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip).float()).item() # clip触发比例计算
            critic_loss = F.mse_loss(return_batch, values) # 计算critic_loss

        # 更新网络
        self.actor_optimizer.zero_grad() # 清空梯度
        (actor_loss + self.ent_coeff * entropy_penalty).backward() # 反向传播
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip) # 梯度裁减，防止爆炸
        self.actor_optimizer.step() # 网络参数更新

        self.critic_optimizer.zero_grad() # 清空梯度
        critic_loss.backward() # 反向传播
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip) # 梯度裁减，防止爆炸
        self.critic_optimizer.step() # 网络参数更新

        return (actor_loss, entropy_penalty, critic_loss, approx_kl_div, clip_fraction)
    
    def evaluate(self, env_fn, nets, itr, num_batches=5): # 策略评估
        for net in nets.values(): # 切换网络模式
            net.eval()

        eval_batches = []
        for _ in range(num_batches):
            batch = self.sample_parallel(env_fn, *nets.values(), deterministic=True)
            eval_batches.append(batch)

        self.save(nets, self.save_path, "_" + repr(itr))

        # 评估历史最佳并保存
        eval_ep_rewards = [float(i) for b in eval_batches for i in b.ep_rewards]
        avg_eval_ep_rewards = np.mean(eval_ep_rewards)
        if self.highest_reward < avg_eval_ep_rewards:
            self.highest_reward = avg_eval_ep_rewards
            self.save(nets, self.save_path)

        return eval_batches
    
    def train(self, env_fn, n_itr):
        # 创建优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.eps)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr, eps=self.eps)

        train_start_time = time.time() # 记录训练开始的时间

        for itr in range(n_itr):
            print("----------| Iteration{} |----------".format(itr))

            # 设置网络为训练模式
            self.actor.train() # 训练模式
            self.critic.train() # 训练模式

            self.iteration_count = itr # 迭代轮数，传入环境接口可用于课程学习

            sample_start_time = time.time() # 采样开始时间

            # 将网络参数序列化并发送到Ray共享内存
            actor_ref = ray.put(self.actor)
            critic_ref = ray.put(self.critic)
            
            batch = self.sample_parallel(env_fn, actor_ref, critic_ref) # 并行采样
            
            # 提取并转换数据
            observations = batch.states.float()
            actions = batch.actions.float()
            advantages = batch.advantages.float()
            values = batch.values.float()
            old_log_probs = batch.log_probs.float()

            # 记录每项奖励在本轮采样的单步平均
            if hasattr(batch, "reward_terms_mean") and batch.reward_terms_mean is not None:
                for name, mean_val in batch.reward_terms_mean.items(): # 日志记录
                    self.writer.add_scalar(f"RewardTerms/{name}", float(mean_val), itr)

            # 采样情况输出
            num_samples = len(observations) # 采样步数
            elapsed = time.time() - sample_start_time # 采样耗时
            print("Sampling took {:.2f}s for {} steps.".format(elapsed, num_samples))
            
            returns = advantages + values # 计算回报值
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps) # 优势归一化

            minibatch_size = self.minibatch_size or num_samples # 确定minibatch大小
            self.total_steps += num_samples # 更新总采样步数

            # 初始化记录列表
            optimizer_start_time = time.time()
            actor_losses = [] # 每次更新的actor_loss
            entropies = [] # 每次更新的策略熵
            critic_losses = [] # 每次更新的critic_loss
            kls = [] # 每次更新的kl散度
            clip_fractions = [] # 每次更新的clip触发比例

            for epoch in range(self.epochs):
                if self.recurrent: # 按轨迹随机取样
                    random_indices = SubsetRandomSampler(range(len(batch.traj_idx)-1))
                    sampler = BatchSampler(random_indices, minibatch_size, drop_last=False)
                else: # 直接随机取样
                    random_indices = SubsetRandomSampler(range(num_samples))
                    sampler = BatchSampler(random_indices, minibatch_size, drop_last=True)

                for indices in sampler: # 将采样到的所有数据进行分批处理
                    if self.recurrent: # 循环网络，将数据转为（t, b, ）
                        # 将该批所选中的轨迹排列成张量列表
                        obs_batch          = [observations[int(batch.traj_idx[i]):int(batch.traj_idx[i+1])] for i in indices]
                        action_batch       = [actions[int(batch.traj_idx[i]):int(batch.traj_idx[i+1])] for i in indices]
                        return_batch       = [returns[int(batch.traj_idx[i]):int(batch.traj_idx[i+1])] for i in indices]
                        advantage_batch    = [advantages[int(batch.traj_idx[i]):int(batch.traj_idx[i+1])] for i in indices]
                        old_log_prob_batch = [old_log_probs[int(batch.traj_idx[i]):int(batch.traj_idx[i+1])] for i in indices]
                        mask               = [torch.ones_like(r) for r in return_batch] # 生成每条轨迹的全1张量并形成列表
                        # 对齐长度并转换为三维张量（t, b, ）
                        obs_batch       = pad_sequence(obs_batch, batch_first=False)
                        action_batch    = pad_sequence(action_batch, batch_first=False)
                        return_batch    = pad_sequence(return_batch, batch_first=False)
                        advantage_batch = pad_sequence(advantage_batch, batch_first=False)
                        old_log_prob_batch = pad_sequence(old_log_prob_batch, batch_first=False)
                        mask            = pad_sequence(mask, batch_first=False) # 将各轨迹对应的mask对齐（短轨迹填0）
                    else: # 非循环网络，将数据转为（b，）
                        obs_batch          = observations[indices]
                        action_batch       = actions[indices]
                        return_batch       = returns[indices]
                        advantage_batch    = advantages[indices]
                        old_log_prob_batch = old_log_probs[indices]
                        mask               = 1

                    # 更新网络
                    actor_loss, entropy_penalty, critic_loss, approx_kl_div, clip_fraction = self.update_policies(obs_batch, 
                                                                                                                  action_batch, 
                                                                                                                  return_batch, 
                                                                                                                  advantage_batch, 
                                                                                                                  old_log_prob_batch,
                                                                                                                  mask)

                    # 更新训练相关状态值
                    actor_losses.append(actor_loss.item())
                    entropies.append(entropy_penalty.item())
                    critic_losses.append(critic_loss.item())
                    kls.append(approx_kl_div.item())
                    clip_fractions.append(clip_fraction)

            elapsed = time.time() - optimizer_start_time # 单轮网络更新耗时
            print("Optimizer took: {:.2f}s".format(elapsed))

            action_noise = torch.exp(self.actor.log_std.data).tolist()

            # 训练状态输出
            sys.stdout.write("-" * 37 + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Mean Eprew', "%8.5g" % torch.mean(batch.ep_rewards)) + "\n") # 单轨迹平均奖励值
            sys.stdout.write("| %15s | %15s |" % ('Mean Eplen', "%8.5g" % torch.mean(batch.ep_lens)) + "\n")    # 单轨迹平均长度
            sys.stdout.write("| %15s | %15s |" % ('Actor loss', "%8.3g" % np.mean(actor_losses)) + "\n")        # actor网络平均损失
            sys.stdout.write("| %15s | %15s |" % ('Critic loss', "%8.3g" % np.mean(critic_losses)) + "\n")      # critic网络平均损失
            sys.stdout.write("| %15s | %15s |" % ('Mean KL Div', "%8.3g" % np.mean(kls)) + "\n")                # 新老策略KL距离
            sys.stdout.write("| %15s | %15s |" % ('Mean Entropy', "%8.3g" % np.mean(entropies)) + "\n")         # 策略熵
            sys.stdout.write("| %15s | %15s |" % ('Clip Fraction', "%8.3g" % np.mean(clip_fractions)) + "\n")   # clip触发比例
            sys.stdout.write("| %15s | %15s |" % ('Mean noise std', "%8.3g" % np.mean(action_noise)) + "\n")    # 动作噪声
            sys.stdout.write("-" * 37 + "\n")
            sys.stdout.flush()

            # 训练时间输出
            elapsed = time.time() - train_start_time # 训练总耗时
            iter_avg = elapsed/(itr+1) # 每轮平均耗时
            ETA = round((n_itr - itr)*iter_avg) # 预计剩余时间
            print("Total time elapsed: {:.2f}s. Total steps: {} (fps={:.2f}. iter-avg={:.2f}s. ETA={})".format(
                elapsed, self.total_steps, self.total_steps/elapsed, iter_avg, datetime.timedelta(seconds=ETA)))

            if itr == 0 or (itr+1)%self.eval_freq == 0: # 是否应该执行评估
                nets = {"actor": self.actor, "critic": self.critic}

                # 策略评估
                evaluate_start = time.time()
                eval_batches = self.evaluate(env_fn, nets, itr)
                eval_time = time.time() - evaluate_start
                # 评估结果输出
                eval_ep_lens = [float(i) for b in eval_batches for i in b.ep_lens]
                eval_ep_rewards = [float(i) for b in eval_batches for i in b.ep_rewards]
                avg_eval_ep_lens = np.mean(eval_ep_lens)
                avg_eval_ep_rewards = np.mean(eval_ep_rewards)
                print("====EVALUATE EPISODE====")
                print("(Episode length:{:.3f}. Reward:{:.3f}. Time taken:{:.2f}s)".format(
                    avg_eval_ep_lens, avg_eval_ep_rewards, eval_time))

                self.writer.add_scalar("Eval/mean_reward", avg_eval_ep_rewards, itr)
                self.writer.add_scalar("Eval/mean_episode_length", avg_eval_ep_lens, itr)

            # TensorBoard训练指标记录
            self.writer.add_scalar("Loss/actor", np.mean(actor_losses), itr)                    # actor网络损失
            self.writer.add_scalar("Loss/critic", np.mean(critic_losses), itr)                  # critic网络损失
            self.writer.add_scalar("Train/mean_reward", torch.mean(batch.ep_rewards), itr)      # 单轨迹平均奖励值
            self.writer.add_scalar("Train/mean_episode_length", torch.mean(batch.ep_lens), itr) # 单轨迹平均长度 
            self.writer.add_scalar("Train/mean_noise_std", np.mean(action_noise), itr)          # 动作噪声