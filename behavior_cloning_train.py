# 假設你已經有 states (N, 96, 96, 1) 和 actions (N,) 的 numpy array
import tensorflow as tf
import NeuralNets as NN

# 1. 準備資料集
batch_size = 64
dataset = tf.data.Dataset.from_tensor_slices((states, actions)).shuffle(10000).batch(batch_size)

# 2. 建立 policy network
policy_net = NN.policy_nn()  # 直接用你 PPO 的 policy_nn 結構

# 3. 編譯模型
policy_net.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. 訓練
policy_net.fit(dataset, epochs=10)

# 5. 儲存模型（可選）
policy_net.save_weights('imitation_policy_net.h5')