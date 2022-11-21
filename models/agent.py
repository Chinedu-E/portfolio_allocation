import tensorflow as tf
from keras import backend as K 
from keras.layers import LeakyReLU, Dense, Conv1D, Input, Dropout, Conv2D, Reshape, LSTM, BatchNormalization, Bidirectional, Flatten, Concatenate
from keras.optimizers import Adam
from keras.models import Model
import numpy as np
from noise import OUActionNoise
from buffer import MemoryBuffer


class DDPGAgent:
    def __init__(self, window_length: int, n_stocks: int, actor_lr: float, critic_lr: float,
                 tau: float, discount_rate: float, buffer_capacity: int, batch_size: int, policy: str,
                 epsilon: float, epsilon_min: float, epsilon_decay: float, mode: str):
        
        assert policy in ["CNN", "LSTM"]
        assert mode in ["train", "test"]

        self.policy = policy
        self.mode = mode
        self.window_length = window_length
        self.n_stocks = n_stocks

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.gamma = discount_rate
        self.w_per = False
        
        self.noise = OUActionNoise(size=n_stocks)
        self.buffer = MemoryBuffer(buffer_size=buffer_capacity, with_per=self.w_per)
        self.batch_size = batch_size
        
        self.actor = self._build_actor()
        self.actor_target = self._build_actor()
        self.actor_target.set_weights(self.actor.get_weights())
        self.actor_optimizer = Adam(actor_lr)
        
        self.critic = self._build_critic()
        self.critic_target = self._build_critic()
        self.critic_target.set_weights(self.critic.get_weights())
        self.critic_optimizer = Adam(critic_lr)
          
    def _build_actor(self):
        #------ Starting stream 1-----#
        input1 = Input(shape=(self.window_length, self.n_stocks, 3))
        
        if self.policy == "LSTM":
            resh = Reshape(target_shape=(self.window_length, self.n_stocks*3))(input1)
            s1 = Bidirectional(LSTM(32, return_sequences=True))(resh)
            s1 = Dropout(0.3)(s1)
            s1 = LeakyReLU(0.2)(s1)
            s1 = Bidirectional(LSTM(32))(s1)
        else:
            s1 = Conv2D(64, kernel_size=3, strides=2)(input1)
            s1 = Dropout(0.3)(s1)
            s1 = LeakyReLU(0.2)(s1)
            s1 = Conv2D(64, kernel_size=3, strides=2)(s1)
            
        s1 = Dropout(0.3)(s1)
        s1 = LeakyReLU(0.2)(s1)
        s1 = Flatten()(s1)
        #------ Starting stream 2------#
        
        input2 = Input(shape=(self.n_stocks, self.n_stocks))
        
        s2 = Conv1D(16, kernel_size=3, strides=1)(input2)
        s2 = LeakyReLU(0.2)(s2)
        s2 = Dropout(0.3)(s2)
        s2 = Conv1D(32, kernel_size=5, strides=1)(s2)
        s2 = LeakyReLU(0.2)(s2)
        s2 = Dropout(0.3)(s2)
        s2 = Flatten()(s2)
        
        input3 = Input(shape=(self.n_stocks))
        #----- Concatenation ----- #
        
        merged = Concatenate()([s1, s2, input3])
        
        fc = Dense(128, kernel_regularizer="l2")(merged)
        fc = LeakyReLU()(fc)
        fc = Dropout(0.5)(fc)
        
        fc = Dense(128, kernel_regularizer="l2")(fc)
        fc = LeakyReLU()(fc)
        fc = Dropout(0.5)(fc)
        
        output = Dense(self.n_stocks, activation="softmax")(fc)
        
        model = Model(inputs=[input1, input2, input3], outputs=output)
        return model
        
    def _build_critic(self):
        #------ Starting stream 1-----#
        input1 = Input(shape=(self.window_length, self.n_stocks, 3))
        
        if self.policy == "LSTM":
            resh = Reshape(target_shape=(self.window_length, self.n_stocks*3))(input1)
            s1 = Bidirectional(LSTM(32, return_sequences=True))(resh)
            s1 = Dropout(0.3)(s1)
            s1 = LeakyReLU(0.2)(s1)
            s1 = Bidirectional(LSTM(50))(s1)
        else:
            s1 = Conv2D(80, kernel_size=3, strides=2)(input1)
            s1 = Dropout(0.3)(s1)
            s1 = LeakyReLU(0.2)(s1)
            s1 = Conv2D(80, kernel_size=6, strides=1)(s1)
            
        s1 = Dropout(0.3)(s1)
        s1 = LeakyReLU(0.2)(s1)
        s1 = Flatten()(s1)
        
        #------ Starting stream 2------#
        input2 = Input(shape=(self.n_stocks, self.n_stocks))
        
        s2 = Conv1D(20, kernel_size=3, strides=1)(input2)
        s2 = LeakyReLU(0.2)(s2)
        s2 = Dropout(0.3)(s2)
        s2 = Conv1D(40, kernel_size=5, strides=1)(input2)
        s2 = LeakyReLU(0.2)(s2)
        s2 = Dropout(0.3)(s2)
        s2 = Flatten()(s2)
        
        #-------Stream 3----------
        input3 = Input(shape=(self.n_stocks,))
        
        #-------Stream 4----------
        input4 = Input(shape=(self.n_stocks,))
        
        #----- Concatenation ----- #
        
        merged = Concatenate()([s1, s2, input3, input4])
        
        fc = Dense(256, kernel_regularizer="l2")(merged)
        fc = LeakyReLU()(fc)
        fc = Dropout(0.5)(fc)
        
        fc = Dense(256, kernel_regularizer="l2")(fc)
        fc = LeakyReLU()(fc)
        fc = Dropout(0.5)(fc)
        
        output = Dense(self.n_stocks)(fc)
        
        model = Model(inputs=[input1, input2, input3, input4], outputs=output)
        return model
        
    def make_action(self, obs: list, t: int, noise=True):
        if self.mode == "train":
            e = np.random.randn()
            if e < self.epsilon:
                action = np.random.random(size=self.n_stocks)
                action = np.array(tf.nn.softmax(action))
                return action
            else:
                return self.actor.predict(obs)[0]
        else:
            return self.actor.predict(obs)[0]
        #a = np.clip(action_ + self.noise.generate(t) if noise else 0, 0, 1)
        return action_
    
    def learn(self):
        if (self.buffer.size() <= self.batch_size): return 0,0
        # sample from buffer
        states, actions, rewards, dones, new_states, idx = self.sample_batch(self.batch_size)
        
        # get target q-value using target network

        q_vals = self.critic_target([*new_states, self.actor_target(new_states)])
        
        # bellman iteration for target critic value
        critic_target = np.array(q_vals)
        for i in range(q_vals.shape[0]):
            if dones[i]:
                critic_target[i] = rewards[i]
            else:
                critic_target[i] = self.gamma * q_vals[i] + rewards[i]
                
            if self.w_per:
                self.buffer.update(idx[i], abs(q_vals[i]-critic_target[i])[0])
                
        # train(or update) the actor & critic and target networks
        actor_loss, critic_loss = self.update_networks(states, actions, critic_target)
        return actor_loss, critic_loss
        
    def memorize(self, obs, action, reward, done, new_obs):
        if self.w_per:
            act = self.actor.predict(obs)
            qval = self.critic([*obs, act])
        
            next_act = self.actor.predict(new_obs)
            next_q_val = self.critic_target.predict([*new_obs, next_act])
            new_val = reward + self.gamma*next_q_val
            td_error = abs(new_val - qval)[0]
        else:
            td_error = 0
        
        self.buffer.memorize(*obs, action, reward, done, *new_obs, td_error)
    
    def Qgradient(self, obs, acts):
        acts = tf.convert_to_tensor(acts)
        with tf.GradientTape() as tape:
            tape.watch(acts)
            q_values = self.critic([*obs,acts])
            q_values = tf.squeeze(q_values)
        return tape.gradient(q_values, acts)
        
    def update_networks(self, obs, acts, critic_target):
        """ Train actor & critic from sampled experience
        """ 
        # update critic
        critic_loss = self.train_critic(obs, acts, critic_target)
        
        # update actor
        actor_loss = self.train_actor(obs, self.critic)
        
        return actor_loss, critic_loss
        
        
    def update_target_networks(self):
        # Start actor updates
        weights, weights_t = self.actor.get_weights(), self.actor_target.get_weights()
        for i in range(len(weights)):
            weights_t[i] = self.tau*weights[i] + (1-self.tau)*weights_t[i]
        self.actor_target.set_weights(weights_t)
        
        # Start critic updates
        weights, weights_t = self.critic.get_weights(), self.critic_target.get_weights()
        for i in range(len(weights)):
            weights_t[i] = self.tau*weights[i] + (1-self.tau)*weights_t[i]
        self.critic_target.set_weights(weights_t)
        
    def train_critic(self, obs: list, acts, target):
        with tf.GradientTape() as tape:
            q_values = self.critic([*obs, acts], training=True)
            td_error = q_values - target
            critic_loss = tf.reduce_mean(tf.math.square(td_error))
            self.critic_loss = float(critic_loss)
        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))
        return critic_loss
        
    def train_actor(self, obs: list, critic):
        with tf.GradientTape() as tape:
            actions = self.actor(obs)
            actor_loss = -tf.reduce_mean(critic([*obs,actions]))
        actor_grad = tape.gradient(actor_loss,self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad,self.actor.trainable_variables))
        return actor_loss
        
    def sample_batch(self, batch_size: int):
        """ Sampling from the batch
        """
        s1, s2, s3, a, r, d, ns1, ns2, ns3, idx =  self.buffer.sample_batch(batch_size)
        s1 = tf.convert_to_tensor(s1.reshape(self.batch_size, self.window_length, self.n_stocks, 3))
        s2 = tf.convert_to_tensor(s2.reshape(self.batch_size, self.n_stocks, self.n_stocks))
        s3 = tf.convert_to_tensor(s3.reshape(self.batch_size, self.n_stocks))
        
        ns1 = tf.convert_to_tensor(ns1.reshape(self.batch_size, self.window_length, self.n_stocks, 3))
        ns2 = tf.convert_to_tensor(ns2.reshape(self.batch_size, self.n_stocks, self.n_stocks))
        ns3 = tf.convert_to_tensor(ns3.reshape(self.batch_size, self.n_stocks))
        return [s1, s2, s3], a, r, d, [ns1, ns2, ns3], idx

    def decay(self):
        """Linear decay to the epsilon value"""
        self.epsilon = max(self.epsilon_min, self.epsilon-self.epsilon_decay)
        
    def save_weights(self, paths: list[str]):
        assert len(paths) == 4
        self.actor.save_weights(paths[0])
        self.actor_target.save_weights(paths[1])
        
        self.critic.save_weights(paths[2])
        self.critic_target.save_weights(paths[3])
        
    def load_weights(self, paths: list[str]):
        assert len(paths) == 4
        self.actor.load_weights(paths[0])
        self.actor.load_weights(paths[1])
        
        self.critic.load_weights(paths[2])
        self.actor.load_weights(paths[3])