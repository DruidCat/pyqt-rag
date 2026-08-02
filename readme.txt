Инструкция.

# Ссылка на github:
https://github.com/DruidCat/pyqt-rag

# Создайте новое виртуальное окружение
mkdir ~/git/rag
cd ~/git/rag
python -m venv venv

# Активируйте его
source venv/bin/activate

# Установите чистые зависимости
pip install --upgrade pip
pip install -r pipinstall.txt

(venv) druidcat@druidcat:~/rag$ python make.py
После завершения можете протестировать поиск:
(venv) druidcat@druidcat:~/rag$ python search.py
Подключение к LM Studio с RAG моделью. Нужно, чтоб была загружена языковая модель, включен локальный сервер, порт 1234:
(venv) druidcat@druidcat:~/rag$ python lmstudio.py
