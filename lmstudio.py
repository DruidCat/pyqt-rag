#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import faiss
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from openai import OpenAI
import time

# ==================== НАСТРОЙКИ ====================

# Пути к файлам индекса
INDEX_DIR = "/mnt/Yandex.Disk/Мои Документы/БД/A.I. СССР/faiss_index"
INDEX_FILE = os.path.join(INDEX_DIR, "index.faiss")
DOCUMENTS_FILE = os.path.join(INDEX_DIR, "documents.pkl")
METADATAS_FILE = os.path.join(INDEX_DIR, "metadatas.pkl")

# Настройки LM Studio
LM_STUDIO_URL = "http://localhost:1234/v1"
LM_STUDIO_MODEL = "local-model"  # Любое имя, LM Studio игнорирует

# Параметры поиска
TOP_K = 5  # Количество найденных фрагментов
SIMILARITY_THRESHOLD = 0.5  # Порог схожести (0-1, чем выше - строже)

# ==================== ЗАГРУЗКА ИНДЕКСА ====================

def load_faiss_index():
    """Загрузка FAISS индекса и документов"""
    print("📂 Загрузка векторной базы данных...")
    
    if not os.path.exists(INDEX_FILE):
        print(f"❌ Файл индекса не найден: {INDEX_FILE}")
        sys.exit(1)
    
    index = faiss.read_index(INDEX_FILE)
    
    with open(DOCUMENTS_FILE, "rb") as f:
        documents = pickle.load(f)
    
    with open(METADATAS_FILE, "rb") as f:
        metadatas = pickle.load(f)
    
    print(f"✓ Индекс загружен: {index.ntotal} векторов")
    print(f"✓ Документы: {len(documents)} фрагментов")
    
    return index, documents, metadatas

# ==================== МОДЕЛЬ ДЛЯ ЭМБЕДДИНГОВ ====================

class EmbeddingModel:
    """Класс для генерации эмбеддингов (та же модель, что при создании)"""
    
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        print("📥 Загрузка модели эмбеддингов...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        print("✓ Модель готова")
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode(self, texts, batch_size=32):
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True, 
                                   max_length=512, return_tensors='pt')
            
            with torch.no_grad():
                model_output = self.model(**encoded)
                batch_embeddings = self.mean_pooling(model_output, encoded['attention_mask'])
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                all_embeddings.append(batch_embeddings.cpu())
        
        return torch.vstack(all_embeddings).numpy().astype('float32')

# ==================== ПОИСК В FAISS ====================

def search_faiss(index, query_embedding, documents, metadatas, top_k=5):
    """Поиск наиболее похожих фрагментов"""
    
    # Поиск
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(documents):
            distance = float(distances[0][i])
            # Конвертация расстояния L2 в схожесть (примерно)
            similarity = 1 / (1 + distance)
            
            if similarity >= SIMILARITY_THRESHOLD:
                results.append({
                    'text': documents[idx],
                    'metadata': metadatas[idx],
                    'similarity': similarity,
                    'distance': distance
                })
    
    # Сортировка по схожести
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    return results

# ==================== LM STUDIO API ====================

def query_lm_studio(context, question, max_tokens=1024):
    """Отправка запроса в LM Studio"""
    
    client = OpenAI(
        base_url=LM_STUDIO_URL,
        api_key="not-needed"  # LM Studio не требует ключ
    )
    
    # Формирование промпта с контекстом
    system_prompt = """Ты — помощник, отвечающий на вопросы на основе предоставленного контекста.
Если ответа нет в контексте, честно скажи об этом.
Отвечай на русском языке."""

    user_prompt = f"""Контекст из документов:
{context}

Вопрос: {question}

Ответ:"""

    try:
        response = client.chat.completions.create(
            model=LM_STUDIO_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=max_tokens,
            stream=False
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"❌ Ошибка LM Studio: {e}"

# ==================== ОСНОВНОЙ КЛАСС RAG ====================

class RAGQuery:
    """Класс для выполнения RAG-запросов"""
    
    def __init__(self):
        self.index, self.documents, self.metadatas = load_faiss_index()
        self.embedder = EmbeddingModel()
    
    def query(self, question, top_k=TOP_K, show_sources=True):
        """Выполнение запроса"""
        
        print("\n" + "="*70)
        print("🔍 ПОИСК ОТВЕТА")
        print("="*70)
        print(f"Вопрос: {question}\n")
        
        # Генерация эмбеддинга вопроса
        print("⚙️  Генерация эмбеддинга вопроса...")
        query_embedding = self.embedder.encode([question])
        
        # Поиск в индексе
        print("🔎 Поиск в векторной базе...")
        results = search_faiss(self.index, query_embedding, self.documents, self.metadatas, top_k)
        
        if not results:
            print("❌ Не найдено релевантных фрагментов")
            return None
        
        # Формирование контекста
        context_parts = []
        print(f"\n📚 Найдено {len(results)} релевантных фрагментов:\n")
        
        for i, result in enumerate(results, 1):
            if show_sources:
                filename = os.path.basename(result['metadata'].get('filename', 'unknown'))
                print(f"  [{i}] {filename} (схожесть: {result['similarity']:.3f})")
            context_parts.append(f"[{i}] {result['text']}")
        
        context = "\n\n".join(context_parts)
        
        # Запрос к LM Studio
        print("\n🤖 Генерация ответа через LM Studio...")
        start_time = time.time()
        
        answer = query_lm_studio(context, question)
        
        elapsed = time.time() - start_time
        
        # Вывод ответа
        print("\n" + "="*70)
        print("💬 ОТВЕТ")
        print("="*70)
        print(answer)
        print("="*70)
        print(f"⏱️  Время генерации: {elapsed:.1f} сек")
        
        return {
            'answer': answer,
            'sources': results,
            'time': elapsed
        }

# ==================== ИНТЕРАКТИВНЫЙ РЕЖИМ ====================

def interactive_mode():
    """Интерактивный режим запросов"""
    
    print("\n" + "="*70)
    print("🤖 RAG-ПОМОЩНИК С LM STUDIO")
    print("="*70)
    print("Введите вопрос или 'выход' для завершения\n")
    
    rag = RAGQuery()
    
    while True:
        try:
            question = input("\n❓ Вопрос: ").strip()
            
            if question.lower() in ['выход', 'exit', 'quit', 'q']:
                print("\n👋 До свидания!")
                break
            
            if not question:
                continue
            
            rag.query(question)
        
        except KeyboardInterrupt:
            print("\n\n👋 До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")

# ==================== ЗАПУСК ====================

if __name__ == "__main__":
    
    # Проверка подключения к LM Studio
    print("🔌 Проверка подключения к LM Studio...")
    try:
        import requests
        response = requests.get(f"{LM_STUDIO_URL}/models", timeout=5)
        if response.status_code == 200:
            print("✓ LM Studio подключён\n")
        else:
            print("⚠️  LM Studio ответил, но не стандартно\n")
    except:
        print("❌ LM Studio не доступен!")
        print("   Запустите LM Studio и включите Local Server")
        print("   Порт: 1234 (по умолчанию)\n")
        sys.exit(1)
    
    # Запуск
    interactive_mode()
