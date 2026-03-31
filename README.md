# Credit Scoring Project

## Структура проекта

```
src/
  app/
    antifraud/
    core/
    modelling/
    scoring/
    utils/
  config/
```

### Клонирование репозитория

git clone https://shift.gitlab.yandexcloud.net/shift/sharinsky/credit_scoring

### Создание виртуального окружения

conda create -n credit_scoring python=3.9 -y  
conda activate credit_scoring  

### Установка зависимостей

pip install poetry  
poetry init  
poetry add jupyter  

### В переменнызх окружения нужно задать значения для

DB_USER  
DB_PASSWORD  
