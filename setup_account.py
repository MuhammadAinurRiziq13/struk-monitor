import firebase_admin
from firebase_admin import credentials, auth, firestore
import os

def setup():
    # 1. Inisialisasi Firebase
    if not firebase_admin._apps:
        print("[-] Menghubungkan ke Firebase...")
        cred = credentials.Certificate('config/firebase_credentials.json')
        firebase_admin.initialize_app(cred)
    
    db = firestore.client()
    
    email = "admin@strokemonitor.com"
    password = "keluarga123"
    
    print(f"[-] Menyiapkan akun untuk: {email}")
    
    # 2. Buat atau ambil User dari Firebase Auth
    try:
        user = auth.get_user_by_email(email)
        print(f"[OK] Akun sudah ada. UID: {user.uid}")
    except auth.UserNotFoundError:
        print("[*] Membuat akun baru di Firebase Auth...")
        user = auth.create_user(
            email=email,
            password=password,
            display_name="Keluarga StrokeMonitor"
        )
        print(f"[OK] Akun berhasil dibuat! UID: {user.uid}")

    uid = user.uid

    # 3. Setup Firestore Document
    print(f"[-] Menyiapkan database Firestore untuk UID: {uid}...")
    
    # Root user document
    db.collection('users').document(uid).set({
        'nama': 'Keluarga StrokeMonitor',
        'email': email,
        'role': 'caregiver'
    }, merge=True)

    # Gesture Config
    doc_ref = db.collection('users').document(uid).collection('config').document('gestures')
    doc_ref.set({
        'mapping': {
            'Makan': ['A'],
            'Minum': ['B'],
            'Toilet': ['C'],
            'Tidur': ['D']
        }
    }, merge=True)
    print("[OK] Database Firestore sudah siap.")

    # 4. Update file .env
    print("[-] Memperbarui file .env...")
    env_file = ".env"
    lines = []
    
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            lines = f.readlines()
    
    # Update atau tambah USER_ID
    found = False
    new_lines = []
    for line in lines:
        if line.startswith("USER_ID="):
            new_lines.append(f"USER_ID={uid}\n")
            found = True
        else:
            new_lines.append(line)
            
    if not found:
        new_lines.append(f"USER_ID={uid}\n")
        
    with open(env_file, "w") as f:
        f.writelines(new_lines)
        
    print(f"[BERHASIL] File .env telah diperbarui dengan UID: {uid}")
    print("\n" + "="*50)
    print("SELESAI! Sekarang Anda bisa:")
    print("1. Jalankan server: uvicorn app.main:app --reload")
    print("2. Buka aplikasi di HP dan klik 'Masuk'")
    print("="*50)

if __name__ == "__main__":
    setup()
