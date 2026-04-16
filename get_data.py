from bing_image_downloader import downloader

# โค้ดชุดที่ 1: ดูดรูปแอปเปิ้ล 
# ทริค: ใช้คำค้นหาภาษาอังกฤษจะเจาะจงกว่า เช่น "red apple fruit isolated" (แอปเปิ้ลแดงพื้นหลังขาว)
downloader.download("red apple fruit isolated", limit=100, output_dir="dataset")

# โค้ดชุดที่ 2: ดูดรูปส้ม
# ทริค: อย่าลืมดักคำว่า fruit ไม่งั้นอาจจะได้รูปสีส้มแทน
downloader.download("orange fruit isolated", limit=100, output_dir="dataset")

print("ดาวน์โหลดรูปภาพเสร็จสิ้นแล้วลูกพี่!")