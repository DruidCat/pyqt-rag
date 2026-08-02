import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import faiss
import pickle
from transformers import AutoTokenizer, AutoModel
import torch
import sys

# Путь к базе данных
base_dir = "/mnt/Yandex.Disk/Мои Документы/БД/A.I. СССР"
index_dir = os.path.join(base_dir, "faiss_index")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_text(text, tokenizer, model):
    encoded = tokenizer([text], padding=True, truncation=True, 
                       max_length=512, return_tensors='pt')
    
    with torch.no_grad():
        model_output = model(**encoded)
        embedding = mean_pooling(model_output, encoded['attention_mask'])
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    
    return embedding.cpu().numpy()

def clear_screen():
    """Очистка экрана"""
    os.system('clear' if os.name == 'posix' else 'cls')

def print_header():
    """Печать заголовка"""
    print("="*70)
    print("🔍 СЕМАНТИЧЕСКИЙ ПОИСК ПО БАЗЕ ЗНАНИЙ RAG")
    print("="*70)

# Проверка существования базы данных
if not os.path.exists(index_dir):
    print("❌ Ошибка: База данных RAG не найдена!")
    print(f"   Ожидаемый путь: {index_dir}")
    print("\n💡 Сначала создайте базу данных, запустив: python rag.py")
    sys.exit(1)

# Загрузка модели
print("📥 Загрузка модели...")
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Загрузка индекса
print("📚 Загрузка базы данных...")
try:
    index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
    
    with open(os.path.join(index_dir, "documents.pkl"), "rb") as f:
        documents = pickle.load(f)
    
    with open(os.path.join(index_dir, "metadatas.pkl"), "rb") as f:
        metadatas = pickle.load(f)
except Exception as e:
    print(f"❌ Ошибка при загрузке базы данных: {e}")
    sys.exit(1)

# Очищаем экран и показываем стартовую информацию
clear_screen()
print_header()
print(f"\n✓ Модель загружена")
print(f"✓ База данных загружена")
print(f"  📊 Фрагментов в базе: {len(documents)}")
print(f"  📁 Уникальных файлов: {len(set(m['filename'] for m in metadatas))}")
print(f"  📂 Местоположение: {index_dir}")
print("\n" + "="*70)
print("Команды:")
print("  'exit', 'quit', 'q' - выход")
print("  'top5' - показывать 5 результатов (по умолчанию)")
print("  'top10' - показывать 10 результатов")
print("  'clear', 'cls' - очистить экран")
print("="*70)

# Поиск
num_results = 5

while True:
    query = input("\n🔍 Введите запрос: ")
    
    # Команда выхода
    if query.lower() in ['exit', 'quit', 'q']:
        clear_screen()
        print("\n👋 До свидания!\n")
        break
    
    # Команда изменения количества результатов
    if query.lower() == 'top10':
        num_results = 10
        print("✓ Теперь показываю 10 результатов")
        continue
    
    if query.lower() == 'top5':
        num_results = 5
        print("✓ Теперь показываю 5 результатов")
        continue
    
    # Команда очистки экрана
    if query.lower() in ['clear', 'cls']:
        clear_screen()
        print_header()
        print(f"\n📊 Фрагментов в базе: {len(documents)}")
        print(f"📁 Уникальных файлов: {len(set(m['filename'] for m in metadatas))}")
        print("="*70)
        continue
    
    # Пустой запрос
    if not query.strip():
        continue
    
    # Очистка экрана перед показом результатов
    clear_screen()
    print_header()
    print(f"\n🔎 Запрос: \"{query}\"")
    print("="*70)
    
    # Поиск
    try:
        query_embedding = encode_text(query, tokenizer, model)
        distances, indices = index.search(query_embedding.astype('float32'), k=num_results)
        
        print(f"\n📚 НАЙДЕНО {num_results} НАИБОЛЕЕ РЕЛЕВАНТНЫХ ФРАГМЕНТОВ:")
        print("="*70)
        
        for i, idx in enumerate(indices[0]):
            score = 1 / (1 + distances[0][i])
            
            print(f"\n┌─ [{i+1}] Релевантность: {score:.1%}")
            print(f"│ 📄 Файл: {metadatas[idx]['filename']}")
            print(f"│ 📂 {metadatas[idx]['source']}")
            print(f"│")
            print(f"│ Текст:")
            
            # Разбиваем длинный текст на строки
            text = documents[idx]
            if len(text) > 500:
                text = text[:500] + "..."
            
            for line in text.split('\n'):
                if line.strip():
                    # Ограничиваем длину строки для красивого вывода
                    if len(line) > 66:
                        # Разбиваем длинные строки
                        words = line.split()
                        current_line = ""
                        for word in words:
                            if len(current_line) + len(word) + 1 <= 66:
                                current_line += (word + " ")
                            else:
                                print(f"│   {current_line.strip()}")
                                current_line = word + " "
                        if current_line:
                            print(f"│   {current_line.strip()}")
                    else:
                        print(f"│   {line}")
            
            print("└" + "─"*68)
        
        print("\n" + "="*70)
        print("💡 Введите новый запрос или команду")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Ошибка при поиске: {e}")
        print("="*70)
