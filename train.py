import tensorflow as tf

# 1. ระบุตำแหน่ง "หนังสือเรียน"
# เติมชื่อโฟลเดอร์หลักที่เก็บรูปผลไม้ทั้ง 2 ชนิดของเราไว้
data_dir = "dataset" # เปลี่ยนเป็นชื่อโฟลเดอร์ที่คุณใช้เก็บรูปภาพ

# 2. ตั้งกฎเกณฑ์การเรียนรู้
img_height = 150  # สั่งให้ AI ย่อทุกรูปให้เหลือขนาด 150x150 พิกเซลเท่ากันหมด
img_width = 150
batch_size = 16   # ป้อนรูปให้ AI ดูทีละ 16 รูป (กันการ์ดจอกินเมมโมรี่เกิน)

print("--- กำลังเตรียมข้อมูลเข้าสมอง AI ---")

# 3. เตรียมชุด "แบบฝึกหัด" (Training Data)
train_data = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2, # หักรูป 20% เอาไปแอบไว้ทำเป็น "ข้อสอบ"
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

# 4. เตรียมชุด "ข้อสอบ" (Validation Data)
val_data = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2, 
  subset="validation",  # นี่คือชุดข้อสอบที่ AI จะไม่เคยเห็นตอนเรียน
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

# ลองให้โปรแกรมปริ้นท์ชื่อผลไม้ที่มันตรวจเจอออกมาดู
class_names = train_data.class_names
print("สิ่งที่ AI ตรวจพบในโฟลเดอร์คือ:", class_names)

# ==========================================
# ส่วนที่ 2: สร้างสมองกลและสั่งเรียนรู้
# ==========================================

# 5. สร้างโครงข่ายประสาทเทียม (Linear Algebra ทำงานตรงนี้!)
model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255), # แปลงรหัสสีจาก 0-255 ให้เป็นเลข 0-1 (ทศนิยม)
  tf.keras.layers.Conv2D(16, 3, activation='relu'), # ดึงจุดเด่นของภาพ (หาขอบ หารูปทรง)
  tf.keras.layers.MaxPooling2D(), # ย่อขนาดภาพลงครึ่งนึงเพื่อให้คิดเลขไวขึ้น
  tf.keras.layers.Flatten(), # ตบเมทริกซ์ให้แบนเป็นเส้นก๋วยเตี๋ยว (Linear Algebra)
  tf.keras.layers.Dense(64, activation='relu'), # เส้นประสาท 64 เส้น
  tf.keras.layers.Dense(2) # ทางออก 2 ทาง (เพราะเรามีผลไม้ 2 ชนิด: ส้ม กับ แอปเปิ้ล)
])

# 6. ตั้งกฎการเรียนรู้ (Calculus ทำงานตรงนี้!)
model.compile(
  optimizer='adam', 
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
  metrics=['accuracy'] 
)

# 7. สั่งให้ AI ลงมือเรียน!!
print("\n--- เริ่มฝึกสมอง AI (ใช้การ์ดจอ RTX 4060 ลุยเลย!) ---")
epochs = 10 # ให้เรียนซ้ำๆ 10 รอบ
history = model.fit(
  train_data,
  validation_data=val_data,
  epochs=epochs
)

print("เรียนจบแล้วจารย์!")

# ==========================================
# ส่วนที่ 4: สอบปฏิบัติจริง (Testing / Prediction)
# ==========================================

# NumPy (อ่านว่า นัมพาย) คือเครื่องมือสุดโหดสำหรับงานคณิตศาสตร์และตัวเลขโดยเฉพาะ!
# AI มันมองทุกอย่างเป็น "ตารางตัวเลข (Array/Matrix)" เราจึงต้องใช้ NumPy มาช่วยบวก ลบ หรือหาค่าสูงสุดในตารางเหล่านั้น
# เราตั้งชื่อเล่นให้มันว่า np จะได้พิมพ์สั้นๆ
import numpy as np 

# 1. ชี้เป้ารูปที่เราต้องการทดสอบ (ข้อสอบลับ)
# ใส่ชื่อไฟล์รูปแอปเปิ้ลหรือส้มที่คุณเพิ่งเซฟมาไว้ในโฟลเดอร์เดียวกับโค้ด
test_image_path = "test_fruit.jpg" 

print("\n--- กำลังประมวลผลข้อสอบลับ ---")

# 2. โหลดรูปและย่อขนาด
# AI ของเราถูกตั้งกฎไว้ตอนแรกว่ารับรูปขนาด 150x150 เท่านั้น เราเลยต้องสั่งย่อรูปนี้ให้ตรงสเปคก่อน
img = tf.keras.utils.load_img(
    test_image_path, target_size=(img_height, img_width)
)

# 3. แปลงภาพเป็นคณิตศาสตร์ (Linear Algebra)
# บรรทัดนี้จะเปลี่ยนสีทุกพิกเซลในรูป ให้กลายเป็นตารางตัวเลข (Matrix) 
img_array = tf.keras.utils.img_to_array(img)

# ปกติ AI ถูกสอนมาให้ตรวจข้อสอบเป็นปึก (Batch) เช่น ปึกละ 16 รูป
# แต่รูปทดสอบเรามีแค่ "1 รูป" เราจึงต้องใช้คำสั่ง expand_dims เพื่อสร้าง "กล่องหลอกๆ" หุ้มมันไว้
# เปลี่ยนรูปแบบข้อมูลจาก (150, 150, 3) ให้กลายเป็น (1, 150, 150, 3) AI จะได้ไม่งง
img_array = tf.expand_dims(img_array, 0) 

# 4. ให้ AI ใช้สมองทำนายผล!
# ฟังก์ชัน .predict() คือการสั่งให้รูปภาพวิ่งผ่านโครงข่ายประสาทเทียมที่เราเทรนไว้
# ผลลัพธ์ที่ได้ (predictions) จะเป็นตัวเลขดิบๆ ทางคณิตศาสตร์ (Logits) เช่น [2.5, -1.2]
predictions = model.predict(img_array)

# 5. แปลงคะแนนดิบเป็นเปอร์เซ็นต์
# ใช้ฟังก์ชันคณิตศาสตร์ Softmax เพื่อบีบตัวเลขดิบๆ ในบรรทัดข้างบน ให้กลายเป็นสัดส่วนความน่าจะเป็นที่รวมกันได้ 100% (1.0)
# ตัวอย่างเช่น แปลง [2.5, -1.2] เป็น [0.97, 0.03]
score = tf.nn.softmax(predictions[0])

# 6. ประกาศผลสอบ!
# np.argmax คือการใช้ NumPy หาว่า "ตัวเลขในตำแหน่งไหนมีค่าสูงที่สุด?" (ตำแหน่งที่ 0 หรือ 1)
# แล้วเอาตำแหน่งนั้นไปเทียบกับป้ายชื่อ (class_names) ว่าตรงกับโฟลเดอร์ผลไม้อะไร
predicted_class = class_names[np.argmax(score)]

# np.max คือการใช้ NumPy ดึง "ตัวเลขที่สูงที่สุด" ออกมาตรงๆ (เช่น 0.97)
# แล้วเอามาคูณ 100 เพื่อทำเป็นเปอร์เซ็นต์ (เช่น 97.0)
confidence = 100 * np.max(score)

# พริ้นท์ผลลัพธ์ออกหน้าจอแบบสวยงาม (.2f คือขอทศนิยมแค่ 2 ตำแหน่ง)
print(f"\n=====================================")
print(f"🍏🍊 ผลสอบออกมาแล้ว!!")
print(f"AI มั่นใจ {confidence:.2f}% ว่ารูปนี้คือ: {predicted_class}")
print(f"=====================================\n")