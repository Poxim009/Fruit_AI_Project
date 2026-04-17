import tensorflow as tf
import numpy as np

data_dir = "dataset" 

img_height = 150
img_width = 150
batch_size = 16

print("--- กำลังเตรียมข้อมูลเข้าสมอง AI ---")

train_data = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

val_data = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2, 
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

class_names = train_data.class_names
print("สิ่งที่ AI ตรวจพบในโฟลเดอร์คือ:", class_names)

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(16, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(2)
])

model.compile(
  optimizer='adam', 
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
  metrics=['accuracy'] 
)

print("\n--- เริ่มฝึกสมอง AI (ใช้การ์ดจอ RTX 4060 ลุยเลย!) ---")
epochs = 10
history = model.fit(
  train_data,
  validation_data=val_data,
  epochs=epochs
)

print("เรียนจบแล้วจารย์!")

test_image_path = "test_fruit.jpg" 

print("\n--- กำลังประมวลผลข้อสอบลับ ---")

img = tf.keras.utils.load_img(
    test_image_path, target_size=(img_height, img_width)
)

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) 

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

predicted_class = class_names[np.argmax(score)]
confidence = 100 * np.max(score)

print(f"\n=====================================")
print(f"🍏🍊 ผลสอบออกมาแล้ว!!")
print(f"AI มั่นใจ {confidence:.2f}% ว่ารูปนี้คือ: {predicted_class}")
print(f"=====================================\n")
