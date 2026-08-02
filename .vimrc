"Сохранить все открытые изменённые файл и Запустить проект с помощью клавиши Alt+F7
"nnoremap <a-F7> :wa<cr>:!./ru.Mentor .<cr>
"inoremap <a-F7> <ESC>:wa<cr>:!./ru.Mentor .<cr>

" Запуск скрипта через Alt+F7
"nnoremap <M-F7> :w<CR>:!cd ~/git/rag && source venv/bin/activate && python make.py<CR>

" Альтернативный вариант с отображением вывода в нижней панели
"nnoremap <silent> <M-F7> :w<CR>:execute '!cd ~/git/rag && source venv/bin/activate && python make.py'<CR>


nnoremap <a-F7> :wa<cr>:!cd ~/git/rag && source venv/bin/activate && python make.py<cr>
inoremap <a-F7> <ESC>:wa<cr>:!cd ~/git/rag && source venv/bin/activate && python make.py<cr>


