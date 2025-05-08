import os

print("başlıyoruz...")

# segmentasyon
print("1) segment.py çalışıyor")
os.system("python segment.py")

# giysi hizalama
print("2) warp_clothing.py çalışıyor")
os.system("python warp_clothing.py")

# giydirme
print("3) tryon_improved.py çalışıyor")
os.system("python tryon_improved.py")

print("bitti. sonucu outputs/tryon_results içinde bulabilirsin.")
