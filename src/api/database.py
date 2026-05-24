import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base

# =====================================================================
# 1. KONFIGURASI VARIABEL LINGKUNGAN (ENVIRONMENT)
# =====================================================================
# Memuat variabel rahasia dari file .env (seperti password dan URL database)
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Safeguard: Memastikan aplikasi tidak berjalan jika URL database kosong
if not DATABASE_URL:
    raise ValueError("❌ DATABASE_URL tidak ditemukan! Pastikan file .env sudah terisi dengan benar.")


# =====================================================================
# 2. INISIALISASI ENGINE ASYNCHRONOUS
# =====================================================================
# create_async_engine memungkinkan koneksi non-blocking ke PostgreSQL.
# Catatan: Ubah echo=True menjadi echo=False jika aplikasi sudah rilis (tahap produksi) 
# agar log terminal tidak terlalu penuh oleh kueri SQL.
engine = create_async_engine(DATABASE_URL, echo=True)


# =====================================================================
# 3. PEMBUATAN PABRIK SESI (SESSION FACTORY) & BASE MODEL
# =====================================================================
# Menonaktifkan expire_on_commit agar objek tetap bisa diakses setelah sesi ditutup (sangat penting di arsitektur Async)
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Kelas dasar yang akan diwarisi oleh semua model tabel di models.py
Base = declarative_base()


# =====================================================================
# 4. DEPENDENCY INJECTION UNTUK FASTAPI
# =====================================================================
async def get_db():
    """
    Fungsi generator (dependency) untuk menyediakan sesi database ke setiap request API.
    Penggunaan blok 'async with' memastikan sesi (koneksi) akan selalu ditutup secara 
    otomatis dan aman setelah request selesai diproses, meskipun terjadi error.
    """
    async with AsyncSessionLocal() as session:
        yield session