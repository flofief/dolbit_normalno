import tensorflow as tf

# Инициализируем модель
model = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(1000,)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Компилируем модель
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Обучаем модель
model.fit(x_train, y_train, epochs=10)

# Оцениваем модель
model.evaluate(x_test, y_test)
