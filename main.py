import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import random

# ==========================================
# 1. CONFIGURATION & PHYSICS
# ==========================================
class Config:
    def __init__(self):
        # Dimensions & Limits
        self.Lf = 1.2; self.Lr = 1.2; self.L = 2.4
        self.dt = 0.05
        self.max_gamma = np.deg2rad(50.0)
        self.max_dgamma = np.deg2rad(90.0)
        self.max_v = 2.0

        # Power Estimation Physics
        self.mass = 500.0      # kg (Field Rover)
        self.g = 9.81
        self.crr = 0.05        # Coeff of Rolling Resistance (Field/Dirt)
        self.k_steer = 150.0   # Watts per rad/s (Actuator load)

        # Baseline Tunings
        self.pp_ld = 2.5
        self.kp_base = 0.45

# ==========================================
# 2. TERRAIN MODEL
# ==========================================
class Terrain:
    def __init__(self, name):
        self.name = name
        if name == "Tarmac":
            self.slip = 0.02; self.crr = 0.02
        elif name == "Rough":
            self.slip = 0.1; self.crr = 0.06
        elif name == "Mud":
            self.slip = 0.2; self.crr = 0.25 # High slip, High resistance

# ==========================================
# 3. VEHICLE MODEL
# ==========================================
class ElectricalVehicle:
    def __init__(self, config, x=0, y=0, yaw=0):
        self.c = config
        self.x = x; self.y = y; self.yaw = yaw; self.gamma = 0; self.v = 0
        self.energy = 0.0

    def update(self, v_target, target_gamma, terrain):
        # 1. Actuator Dynamics
        target_gamma = np.clip(target_gamma, -self.c.max_gamma, self.c.max_gamma)
        dgamma = np.clip(target_gamma - self.gamma, -self.c.max_dgamma * self.c.dt, self.c.max_dgamma * self.c.dt)
        self.gamma += dgamma

        accel = (v_target - self.v) * 0.5
        self.v += accel
        self.v = np.clip(self.v, 0.0, self.c.max_v)

        # 2. SLIP PHYSICS
        v_eff = self.v * (1.0 - terrain.slip)
        steer_eff = 1.0 - (terrain.slip * 0.8)

        # 3. Kinematics
        self.x += v_eff * math.cos(self.yaw) * self.c.dt
        self.y += v_eff * math.sin(self.yaw) * self.c.dt
        yaw_rate = (v_eff * math.sin(self.gamma) / self.c.L) * steer_eff
        self.yaw += yaw_rate * self.c.dt

        while self.yaw > math.pi: self.yaw -= 2*math.pi
        while self.yaw < -math.pi: self.yaw += 2*math.pi

        # 4. POWER CALCULATION
        f_roll = self.c.mass * self.c.g * terrain.crr
        f_accel = self.c.mass * max(0, accel / self.c.dt)

        # Power = Force * Velocity / Efficiency (Slip wastes power)
        traction_eff = max(0.1, 1.0 - terrain.slip)
        p_tract = ((f_roll + f_accel) * self.v) / traction_eff

        p_steer = self.c.k_steer * abs(dgamma / self.c.dt)
        self.energy += (p_tract + p_steer) * self.c.dt

def calc_state_vars(veh, cx, cy, cyaw):
    dists = np.hypot(np.array(cx) - veh.x, np.array(cy) - veh.y)
    idx = np.argmin(dists)
    if idx < len(cx)-1: dx=cx[idx+1]-cx[idx]; dy=cy[idx+1]-cy[idx]
    else: dx=cx[idx]-cx[idx-1]; dy=cy[idx]-cy[idx-1]
    cross = dx*(veh.y-cy[idx]) - dy*(veh.x-cx[idx])
    cte = dists[idx] if cross > 0 else -dists[idx]

    h_err = cyaw[idx] - veh.yaw
    while h_err > math.pi: h_err -= 2*math.pi
    while h_err < -math.pi: h_err += 2*math.pi

    look = min(idx + 30, len(cx)-2)
    curve = abs(math.atan2(cy[look+1]-cy[look], cx[look+1]-cx[look]) - math.atan2(dy, dx))
    return cte, h_err, curve, idx

# ==========================================
# 4. RL AGENT
# ==========================================
ACTION_SPACE = [
    (1.0, 2.0), # 0: Cruise
    (2.0, 1.5), # 1: Rough
    (4.0, 1.0), # 2: Mud Straight
    (6.0, 0.6), # 3: Mud Corner
    (8.0, 0.4)  # 4: Emergency Crawl
]

class QLearningAgent:
    def __init__(self, actions):
        self.q_table = {}
        self.actions = actions
        self.lr = 0.15; self.gamma = 0.95; self.epsilon = 1.0; self.min_epsilon = 0.01; self.decay = 0.997

    def get_state_key(self, cte, h_err, future_curve, slip):
        cte_bin = int(np.clip(cte / 0.3, -4, 4))
        h_bin = int(np.clip(h_err / 0.1, -5, 5))
        c_bin = 0
        if future_curve > 0.05: c_bin = 1
        if future_curve > 0.15: c_bin = 2
        s_bin = 0
        if slip > 0.1: s_bin = 1
        if slip > 0.4: s_bin = 2
        return (cte_bin, h_bin, c_bin, s_bin)

    def choose_action(self, state):
        if state not in self.q_table: self.q_table[state] = np.zeros(len(self.actions))
        if random.uniform(0, 1) < self.epsilon: return random.randint(0, len(self.actions) - 1)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        if state not in self.q_table: self.q_table[state] = np.zeros(len(self.actions))
        if next_state not in self.q_table: self.q_table[next_state] = np.zeros(len(self.actions))
        old = self.q_table[state][action]
        nxt = np.max(self.q_table[next_state])
        self.q_table[state][action] = old + self.lr * (reward + self.gamma * nxt - old)

    def decay_epsilon(self): self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

    def get_cmd(self, veh, cx, cy, yaw, terrain):
        cte, h_err, curve, _ = calc_state_vars(veh, cx, cy, yaw)
        state = self.get_state_key(cte, h_err, curve, terrain.slip)
        if state not in self.q_table: act_idx = 0
        else: act_idx = np.argmax(self.q_table[state])

        gain, speed = self.actions[act_idx]

        dists = np.hypot(np.array(cx) - veh.x, np.array(cy) - veh.y)
        idx = np.argmin(dists)
        dist_look = 0.0
        while dist_look < 2.5 and idx < len(cx)-1:
            dist_look += np.hypot(cx[idx+1]-cx[idx], cy[idx+1]-cy[idx])
            idx += 1
        tx, ty = cx[idx], cy[idx]
        alpha = math.atan2(ty - veh.y, tx - veh.x) - veh.yaw
        while alpha > math.pi: alpha -= 2*math.pi
        while alpha < -math.pi: alpha += 2*math.pi
        gamma_geo = math.atan2(2.0 * veh.c.L * math.sin(alpha), 2.5)

        gamma_cmd = gamma_geo - (veh.c.kp_base * gain * cte)
        return speed, gamma_cmd

# ==========================================
# 5. PATHS & TRAINING
# ==========================================
def generate_path(shape):
    step = 0.1
    if shape == "Sinusoidal":
        x = np.arange(0, 100, step); y = 5.0 * np.sin(x / 8.0)
        wx, wy = x, y
    elif shape == "Trapezoidal":
        kx = [0, 20, 25, 45, 50, 70, 75, 100]
        ky = [0, 0,  10, 10,  0,  0,  10, 10]
        t = np.arange(len(kx)); ti = np.linspace(0, len(kx)-1, 1000)
        wx = interp.PchipInterpolator(t, kx)(ti); wy = interp.PchipInterpolator(t, ky)(ti)
    elif shape == "Square":
        kx = [0, 20, 20, 45, 45, 70, 70, 95]
        ky = [0, 0,  10, 10, -5, -5,  5,  5]
        t = np.arange(len(kx)); ti = np.linspace(0, len(kx)-1, 1000)
        wx = interp.Akima1DInterpolator(t, kx)(ti); wy = interp.Akima1DInterpolator(t, ky)(ti)

    yaw = [math.atan2(wy[i+1]-wy[i], wx[i+1]-wx[i]) for i in range(len(wx)-1)]
    yaw.append(yaw[-1])
    return wx, wy, yaw

def train_agent(episodes=1500):
    agent = QLearningAgent(ACTION_SPACE)
    config = Config()
    paths = ["Sinusoidal", "Trapezoidal", "Square"]
    terrains = [Terrain("Tarmac"), Terrain("Rough"), Terrain("Mud"), Terrain("Mud")]

    # Store history for visualization
    reward_history = []

    print(f"Training RL Agent ({episodes} episodes)...")
    for ep in range(episodes):
        p_name = random.choice(paths); terr = random.choice(terrains)
        cx, cy, cyaw = generate_path(p_name)
        veh = ElectricalVehicle(config, x=cx[0], y=cy[0], yaw=cyaw[0])
        cte, h_err, curve, idx = calc_state_vars(veh, cx, cy, cyaw)
        state = agent.get_state_key(cte, h_err, curve, terr.slip)
        steps = 0

        episode_reward = 0 # Accumulate reward per episode

        while idx < len(cx)-20 and steps < 1500:
            act_idx = agent.choose_action(state)
            gain, speed = ACTION_SPACE[act_idx]

            # Geometric Control
            dists = np.hypot(np.array(cx) - veh.x, np.array(cy) - veh.y)
            pp_idx = min(np.argmin(dists) + 40, len(cx)-1)
            tx, ty = cx[pp_idx], cy[pp_idx]
            alpha = math.atan2(ty - veh.y, tx - veh.x) - veh.yaw
            while alpha > math.pi: alpha -= 2*math.pi
            while alpha < -math.pi: alpha += 2*math.pi
            gamma_geo = math.atan2(2.0 * config.L * math.sin(alpha), 2.5)

            gamma_cmd = gamma_geo - (config.kp_base * gain * cte)
            veh.update(speed, gamma_cmd, terr)

            n_cte, n_h, n_curve, idx = calc_state_vars(veh, cx, cy, cyaw)
            nxt_state = agent.get_state_key(n_cte, n_h, n_curve, terr.slip)

            r = - (abs(n_cte)*2.0 + abs(n_h)*2.5)
            if terr.slip > 0.4 and n_curve > 0.1 and veh.v > 0.8: r -= 20.0 # Penalty for speeding in mud
            if abs(n_cte) < 0.3: r += 2.0

            agent.learn(state, act_idx, r, nxt_state)
            state = nxt_state
            episode_reward += r
            steps += 1

        agent.decay_epsilon()
        reward_history.append(episode_reward)

    # --- VISUALIZE TRAINING PROGRESS ---
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, alpha=0.3, label="Raw Reward", color='gray')

    # Calculate Moving Average
    window_size = 50
    moving_avg = np.convolve(reward_history, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(reward_history)), moving_avg, color='red', linewidth=2, label="Moving Avg (50 eps)")

    plt.title("RL Training Progress: Reward Convergence")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return agent

# Standard Controllers
class PurePursuit:
    def get_cmd(self, veh, cx, cy, yaw, terrain):
        dists = np.hypot(np.array(cx) - veh.x, np.array(cy) - veh.y)
        idx = min(np.argmin(dists) + 40, len(cx)-1)
        tx, ty = cx[idx], cy[idx]
        alpha = math.atan2(ty - veh.y, tx - veh.x) - veh.yaw
        while alpha > math.pi: alpha -= 2*math.pi
        while alpha < -math.pi: alpha += 2*math.pi
        gamma = math.atan2(2.0 * 2.4 * math.sin(alpha), 2.5)
        return 2.0, gamma

class FueMPP:
    def get_cmd(self, veh, cx, cy, yaw, terrain):
        dists = np.hypot(np.array(cx) - veh.x, np.array(cy) - veh.y)
        idx = np.argmin(dists)
        pp_idx = min(idx + 40, len(cx)-1)
        tx, ty = cx[pp_idx], cy[pp_idx]
        alpha = math.atan2(ty - veh.y, tx - veh.x) - veh.yaw
        while alpha > math.pi: alpha -= 2*math.pi
        while alpha < -math.pi: alpha += 2*math.pi
        gamma = math.atan2(2.0 * 2.4 * math.sin(alpha), 2.5)
        if idx < len(cx)-1: dx=cx[idx+1]-cx[idx]; dy=cy[idx+1]-cy[idx]
        else: dx=cx[idx]-cx[idx-1]; dy=cy[idx]-cy[idx-1]
        cross = dx*(veh.y-cy[idx]) - dy*(veh.x-cx[idx])
        cte = dists[idx] if cross > 0 else -dists[idx]
        return 2.0, gamma - (0.45 * cte)

# ==========================================
# 6. EXECUTION & VISUALIZATION
# ==========================================
def run_comparison():
    agent = train_agent(3500)
    agent.epsilon = 0.0 # Test mode
    config = Config()

    terrains = [Terrain("Tarmac"), Terrain("Rough"), Terrain("Mud")]
    controllers = [("Pure Pursuit", PurePursuit()), ("Fue MPP", FueMPP()), ("RL Agent", agent)]

    cx, cy, cyaw = generate_path("Square") # Test on hardest path

    print("\n" + "="*90)
    print(f"{'TERRAIN':<10} | {'CONTROLLER':<15} | {'RMSE (m)':<10} | {'MAX ERR':<10} | {'ENERGY (kJ)':<10}")
    print("="*90)

    # 3 Terrains -> 3 Columns. Each column has 3 rows (Path, V, Gamma)
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))

    for i, terr in enumerate(terrains):
        # Column i is for Terrain i
        ax_path = axes[0, i]
        ax_vel = axes[1, i]
        ax_steer = axes[2, i]

        ax_path.plot(cx, cy, 'k--', linewidth=1, alpha=0.5, label="Target")
        ax_path.set_title(f"{terr.name}\n(Slip={terr.slip})")

        for name, ctrl in controllers:
            veh = ElectricalVehicle(config, x=cx[0], y=cy[0], yaw=cyaw[0])
            x_h, y_h, v_h, g_h, err_h = [], [], [], [], []
            t_h = []

            idx = 0; steps = 0
            while idx < len(cx)-10 and steps < 4000:
                v_cmd, g_cmd = ctrl.get_cmd(veh, cx, cy, cyaw, terr)
                veh.update(v_cmd, g_cmd, terr)

                x_h.append(veh.x); y_h.append(veh.y)
                v_h.append(veh.v); g_h.append(np.rad2deg(veh.gamma))
                t_h.append(steps * config.dt)

                cte, _, _, idx = calc_state_vars(veh, cx, cy, cyaw)
                err_h.append(abs(cte))
                if abs(cte) > 5.0: break # Failed
                steps += 1

            rmse = np.sqrt(np.mean(np.square(err_h)))
            max_e = np.max(err_h)
            energy = veh.energy / 1000.0

            res_str = f"{rmse:.4f}" if max_e < 4.5 else "FAILED"
            print(f"{terr.name:<10} | {name:<15} | {res_str:<10} | {max_e:.4f}     | {energy:.2f}")
            # Plotting
            ls = '-' if name == "RL Agent" else '--'
            lw = 2 if name == "RL Agent" else 1.5

            ax_path.plot(x_h, y_h, linestyle=ls, linewidth=lw, label=name)
            ax_vel.plot(t_h, v_h, linestyle=ls, linewidth=lw, label=name)
            ax_steer.plot(t_h, g_h, linestyle=ls, linewidth=lw, label=name)

        # Formatting
        ax_path.axis('equal'); ax_path.grid(True, alpha=0.3)
        ax_vel.set_ylabel("Speed (m/s)"); ax_vel.grid(True, alpha=0.3)
        ax_steer.set_ylabel("Articulation (deg)"); ax_steer.grid(True, alpha=0.3)
        ax_steer.set_xlabel("Time (s)")

        if i == 2: # Legend only on last column
            ax_path.legend(loc='lower right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_comparison()
