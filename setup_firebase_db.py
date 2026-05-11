import firebase_admin
from firebase_admin import credentials, firestore

try:
    print("Menghubungkan ke Firebase...")
    cred = credentials.Certificate('config/firebase_credentials.json')
    firebase_admin.initialize_app(cred)
    db = firestore.client()

    print("Membuat dokumen user patient_001...")
    db.collection('users').document('patient_001').set({
        'nama': 'Pasien Uji Coba',
        'keterangan': 'Dibuat otomatis oleh sistem'
    })

    print("Membuat konfigurasi gesture mapping...")
    doc_ref = db.collection('users').document('patient_001').collection('config').document('gestures')
    doc_ref.set({
        'mapping': {
            'Makan': ['A'],
            'Minum': ['B'],
            'Toilet': ['C'],
            'Tidur': ['D']
        }
    })

    print("[BERHASIL] Database Firestore sudah disetting otomatis.")
except Exception as e:
    print(f"[GAGAL] Error: {e}")
