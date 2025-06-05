# Запуск
Для локального развёртывания достаточно склонировать репозиторий и запустить проект.
### Клонирование репозитория
```
git clone https://github.com/BeidinaDaria/Store.git
cd Store
```
### Запуск проекта
```
python main.py
```
Стандартно проект запустится по адресу http://127.0.0.1:8080/

### Конфигурация
Файл [конфигурации](https://flask.palletsprojects.com/en/2.2.x/config/) Flask находится по пути `app/config.cfg`.

### База данных
В директории `app/database/` находится база данных `fistore.db`. Для некоторых операций с ней создан скрипт `db_manage.py`. Пример его использования для ручной очистки и заполнения базы данных:
```
python
>>> import app.database.db_manage as manager
>>> manager.clear_db()
>>> manager.fill_db()
```
