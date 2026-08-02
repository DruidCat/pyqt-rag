import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import shutil
import faiss
import pickle
from transformers import AutoTokenizer, AutoModel
import torch
import time
import threading
from datetime import datetime
import zipfile

# Пути к папкам
base_dir = "/mnt/Yandex.Disk/Мои Документы/БД/A.I. СССР"
data_dir = os.path.join(base_dir, "base")
add_dir = os.path.join(base_dir, "add")
arch_dir = os.path.join(base_dir, "arch")
index_dir = os.path.join(base_dir, "faiss_index")

# Время старта скрипта
start_time = time.time()

def create_backup_archive():
    """Создание архива папки base и перемещение в arch"""
    if not os.path.exists(data_dir):
        print("⚠️  Папка base не найдена, архивация пропущена")
        return False
    
    # Создание папки arch если её нет
    os.makedirs(arch_dir, exist_ok=True)
    
    # Генерация имени архива
    current_date = datetime.now().strftime("%Y-%m-%d")
    archive_name = f"{current_date}.zip"
    archive_path = os.path.join(arch_dir, archive_name)
    
    # Проверка на существование архива с таким же именем
    if os.path.exists(archive_path):
        counter = 1
        while os.path.exists(archive_path):
            archive_name = f"{current_date}_{counter}.zip"
            archive_path = os.path.join(arch_dir, archive_name)
            counter += 1
    
    print(f"\n📦 Создание резервной копии базы данных...")
    print(f"   Архив: {archive_name}")
    
    try:
        # Создание zip архива
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Подсчёт файлов для прогресса
            total_files = sum([len(files) for _, _, files in os.walk(data_dir)])
            processed = 0
            
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, data_dir)
                    zipf.write(file_path, arcname)
                    processed += 1
                    
                    # Показываем прогресс каждые 10 файлов
                    if processed % 10 == 0 or processed == total_files:
                        percent = (processed / total_files) * 100
                        sys.stdout.write(f'\r   📄 Архивировано файлов: {processed}/{total_files} ({percent:.1f}%)')
                        sys.stdout.flush()
            
            print()  # Новая строка после прогресса
        
        # Получение размера архива
        archive_size = os.path.getsize(archive_path)
        size_mb = archive_size / (1024 * 1024)
        
        print(f"   ✓ Архив создан: {archive_name} ({size_mb:.1f} MB)")
        print(f"   📁 Местоположение: {arch_dir}")
        return True
    
    except Exception as e:
        print(f"\n   ✗ Ошибка при создании архива: {e}")
        return False

def format_time(seconds):
    """Форматирование времени в читаемый вид"""
    if seconds < 60:
        return f"{seconds:.1f} сек"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} мин"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} час"

# ==================== ПЕРВЫЙ МОДУЛЬ: ПРОВЕРКА НА ПЕРЕСОЗДАНИЕ RAG ====================

if os.path.exists(index_dir) and os.path.isdir(index_dir):
    print("="*70)
    print("⚠️  Обнаружена существующая векторная база данных RAG")
    print("="*70)
    response = input("\n🔄 Пересоздать векторную базу данных RAG? (Д/Н): ")
    
    if response in ['Д', 'д', 'Y', 'y']:
        print("\n🗑️  Удаление старой базы данных...")
        shutil.rmtree(index_dir)
        print("✓ Старая база удалена\n")
    else:
        print("\n❌ Операция отменена. Выход из программы.")
        sys.exit(0)

print("="*70)
print("🚀 СОЗДАНИЕ ВЕКТОРНОЙ БАЗЫ ДАННЫХ RAG")
print("="*70)

# ==================== ВТОРОЙ МОДУЛЬ: ПРОВЕРКА НОВЫХ ФАЙЛОВ В ADD ====================

# Проверка новых файлов в папке add
def check_and_move_new_files():
    """Проверка и перенос новых файлов из папки add в base"""
    if not os.path.exists(add_dir):
        return
    
    # Поиск txt файлов в папке add
    new_files = []
    for root, _, files in os.walk(add_dir):
        for file in files:
            if file.endswith(".txt"):
                new_files.append(os.path.join(root, file))
    
    if not new_files:
        return
    
    # Показываем найденные файлы
    print("="*70)
    print("📥 ОБНАРУЖЕНЫ НОВЫЕ ФАЙЛЫ В ПАПКЕ ДЛЯ ДОБАВЛЕНИЯ")
    print("="*70)
    print(f"\nПапка: {add_dir}\n")
    
    for idx, file_path in enumerate(new_files, 1):
        relative_path = os.path.relpath(file_path, add_dir)
        file_size = os.path.getsize(file_path)
        size_kb = file_size / 1024
        print(f"  [{idx}] {relative_path} ({size_kb:.1f} KB)")
    
    print(f"\n📊 Всего файлов: {len(new_files)}")
    print("="*70)
    
    response = input("\n🔄 Перенести файл(ы) в папку base? (Д/Н): ")
    
    if response in ['Д', 'д', 'Y', 'y']:
        # Создание резервной копии базы перед добавлением новых файлов
        if not create_backup_archive():
            print("\n❌ Ошибка создания архива. Операция отменена.")
            sys.exit(1)
        
        # Создание подпапки с текущей датой
        current_date = datetime.now().strftime("%Y-%m-%d")
        target_dir = os.path.join(data_dir, current_date)
        
        os.makedirs(target_dir, exist_ok=True)
        
        print(f"\n📁 Создана папка: {current_date}/")
        print("🔄 Перенос файлов...\n")
        
        moved_count = 0
        for file_path in new_files:
            try:
                file_name = os.path.basename(file_path)
                target_path = os.path.join(target_dir, file_name)
                
                # Проверка на существование файла с таким же именем
                if os.path.exists(target_path):
                    base_name, ext = os.path.splitext(file_name)
                    counter = 1
                    while os.path.exists(target_path):
                        file_name = f"{base_name}_{counter}{ext}"
                        target_path = os.path.join(target_dir, file_name)
                        counter += 1
                
                shutil.move(file_path, target_path)
                print(f"  ✓ {os.path.basename(file_path)} → {current_date}/{file_name}")
                moved_count += 1
            except Exception as e:
                print(f"  ✗ Ошибка при переносе {os.path.basename(file_path)}: {e}")
        
        print(f"\n✓ Перенесено файлов: {moved_count}/{len(new_files)}\n")
        
        # Удаление пустых папок в add
        try:
            for root, dirs, files in os.walk(add_dir, topdown=False):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    if not os.listdir(dir_path):
                        os.rmdir(dir_path)
        except:
            pass
    else:
        print("\n❌ Файлы не перенесены. Продолжение работы с текущей базой.\n")

# Вызываем проверку новых файлов ТОЛЬКО после подтверждения пересоздания RAG
check_and_move_new_files()

print("\n📥 Загрузка модели...")
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
print("✓ Модель загружена")

def mean_pooling(model_output, attention_mask):
    """Mean Pooling - учитываем attention mask для корректного усреднения"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class SpinnerThread(threading.Thread):
    """Поток для плавной анимации спиннера"""
    def __init__(self):
        super().__init__()
        self.spinner = ['/', '-', '\\', '|']
        self.idx = 0
        self.running = True
        self.message = ""
        self.daemon = True
    
    def run(self):
        while self.running:
            if self.message:
                sys.stdout.write(f'\r  {self.spinner[self.idx]} {self.message}')
                sys.stdout.flush()
                self.idx = (self.idx + 1) % 4
            time.sleep(0.1)  # Обновление каждые 100мс для плавной анимации
    
    def update_message(self, msg):
        self.message = msg
    
    def stop(self):
        self.running = False
        sys.stdout.write('\r')
        sys.stdout.flush()

def encode_texts(texts, batch_size=32):
    """Кодирование текстов в эмбеддинги"""
    all_embeddings = []
    total = len(texts)
    
    # Запуск спиннера в отдельном потоке
    spinner = SpinnerThread()
    spinner.start()
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, 
                          max_length=512, return_tensors='pt')
        
        with torch.no_grad():
            model_output = model(**encoded)
            batch_embeddings = mean_pooling(model_output, encoded['attention_mask'])
            # Нормализация
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            all_embeddings.append(batch_embeddings.cpu())
        
        # Обновление сообщения для спиннера
        processed = i + len(batch)
        percent = (processed / total) * 100
        spinner.update_message(f'Обработано {processed}/{total} фрагментов ({percent:.1f}%)')
    
    # Остановка спиннера
    spinner.stop()
    spinner.join()
    print(f'  ✓ Обработано {total}/{total} фрагментов (100.0%)')
    
    return torch.vstack(all_embeddings)

# Список всех текстовых файлов
print("\n📂 Поиск текстовых файлов...")
txt_files = []
for root, _, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".txt"):
            txt_files.append(os.path.join(root, file))

print(f"✓ Найдено {len(txt_files)} текстовых файлов\n")
print("="*70)
print("📖 ОБРАБОТКА ФАЙЛОВ")
print("="*70)

# Чтение и разбиение текста
documents = []
metadatas = []

for idx, file_path in enumerate(txt_files, 1):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            # Разбиение по абзацам
            chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
            
            chunk_count = 0
            for chunk in chunks:
                if len(chunk) > 50:
                    documents.append(chunk)
                    metadatas.append({
                        "source": file_path,
                        "filename": os.path.basename(file_path)
                    })
                    chunk_count += 1
        
        # Показываем относительный путь от data_dir
        relative_path = os.path.relpath(file_path, data_dir)
        print(f"[{idx:2d}/{len(txt_files)}] ✓ {relative_path:50s} ({chunk_count:4d} фрагментов)")
    except Exception as e:
        print(f"[{idx:2d}/{len(txt_files)}] ✗ Ошибка {os.path.basename(file_path)}: {e}")

print("="*70)
print(f"📊 Всего фрагментов: {len(documents)}")
print("="*70)

# Генерация эмбеддингов
print("\n⚙️  Генерация эмбеддингов...")
embeddings_tensor = encode_texts(documents)

print(f"\n✓ Размерность эмбеддингов: {embeddings_tensor.shape}")

# Создание FAISS индекса
print("\n🔨 Создание FAISS индекса...")
dimension = embeddings_tensor.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_tensor.detach().numpy().astype('float32'))
print("✓ Индекс создан")

# Сохранение
print("\n💾 Сохранение базы данных...")
os.makedirs(index_dir, exist_ok=True)
faiss.write_index(index, f"{index_dir}/index.faiss")

with open(f"{index_dir}/index.faiss", "rb") as f_check:
    pass

with open(f"{index_dir}/documents.pkl", "wb") as f:
    pickle.dump(documents, f)

with open(f"{index_dir}/metadatas.pkl", "wb") as f:
    pickle.dump(metadatas, f)

print("✓ База данных сохранена")

# Подсчёт времени работы
end_time = time.time()
elapsed_time = end_time - start_time

print("\n" + "="*70)
print("🎉 УСПЕШНО!")
print("="*70)
print(f"  📁 Местоположение: {index_dir}")
print(f"  📚 Файлов обработано: {len(txt_files)}")
print(f"  📄 Фрагментов создано: {len(documents)}")
print(f"  🔢 Размерность векторов: {dimension}")
print(f"  ⏱️  Время работы: {format_time(elapsed_time)}")
print("="*70)
