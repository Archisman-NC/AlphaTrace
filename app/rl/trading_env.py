import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class AlphaTraceTradingEnv(gym.Env):
    """
    Experimental RL Trading Environment for research purposes.
    
    Actions: 0 = HOLD, 1 = BUY (Long), 2 = SELL (Flat/Short)
    Position States: 0 = Flat, 1 = Long, -1 = Short
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, df_features: pd.DataFrame, df_prices: pd.Series, initial_balance=100000):
        super(AlphaTraceTradingEnv, self).__init__()
        
        self.df_features = df_features
        self.df_prices = df_prices
        self.initial_balance = initial_balance
        self.n_steps = len(df_features)
        
        # Action space: 0=HOLD, 1=BUY (LONG), 2=SELL (SHORT)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: features + current position state
        # Features: log_return, vol_20d, rsi_norm, macd_norm, bb_pct, log_vol_change
        n_features = df_features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features + 1,), dtype=np.float32
        )
        
        # State
        self.current_step = 0
        self.position = 0 # 0: Flat, 1: Long, -1: Short
        self.balance = initial_balance
        self.shares = 0
        self.net_worth = initial_balance
        self.prev_net_worth = initial_balance
        
        # Diagnostics
        self.total_trades = 0
        self.cumulative_reward = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Ensure action space is seeded for deterministic sampling
        if seed is not None:
            self.action_space.seed(seed)
        
        self.current_step = 0
        self.position = 0
        self.balance = self.initial_balance
        self.shares = 0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.total_trades = 0
        self.cumulative_reward = 0
        
        observation = self._get_observation()
        info = {}
        
        return observation, info

    def _get_observation(self):
        # Get feature vector for current step
        obs_features = self.df_features.iloc[self.current_step].values
        # Append current position state
        obs = np.append(obs_features, [self.position]).astype(np.float32)
        return obs

    def step(self, action):
        # 1. Update step
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        
        # 2. Get price data
        current_price = self.df_prices.iloc[self.current_step]
        
        # 3. Handle Trading Logic (Simplified)
        trade_occurred = False
        if action == 1: # BUY -> LONG
            if self.position != 1:
                self.position = 1
                self.total_trades += 1
                trade_occurred = True
        elif action == 2: # SELL -> SHORT
            if self.position != -1:
                self.position = -1
                self.total_trades += 1
                trade_occurred = True
        # action == 0 (HOLD) does nothing to position
        
        # 4. Calculate Net Worth
        # In this simplified model, we assume we are always 100% in the direction of the position
        # or flat. We use return-based net worth calculation for simplicity.
        daily_return = (current_price / self.df_prices.iloc[self.current_step - 1]) - 1
        step_return = self.position * daily_return
        
        self.net_worth = self.net_worth * (1 + step_return)
        
        # 5. Reward Engineering
        # Reward = % change in net worth - transaction cost penalty
        raw_reward = (self.net_worth / self.prev_net_worth) - 1
        
        # Transaction Penalty (0.1% per trade)
        transaction_penalty = 0.001 if trade_occurred else 0.0
        
        # Overtrading Penalty (to discourage high-frequency flipping if not profitable)
        overtrading_penalty = 0.0
        
        reward = raw_reward - transaction_penalty - overtrading_penalty
        
        self.cumulative_reward += reward
        self.prev_net_worth = self.net_worth
        
        # 6. Prepare output
        observation = self._get_observation()
        terminated = done
        truncated = False
        
        info = {
            "net_worth": self.net_worth,
            "total_trades": self.total_trades,
            "position": self.position,
            "step_return": step_return
        }
        
        return observation, reward, terminated, truncated, info

    def render(self):
        print(f"Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Position: {self.position}")
