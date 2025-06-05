from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from .securityUnit import IntrusionDetectionSystem
from .middleware import PacketAnalyzerMiddleware

#   INITIALIZING

# init application
app = Flask(__name__)

# load config
app.config.from_pyfile("config.cfg")

# init database
db = SQLAlchemy(app)

# Инициализация модуля безопасности
security = IntrusionDetectionSystem()  
app.wsgi_app = PacketAnalyzerMiddleware(app.wsgi_app, security)  # Подключение middleware

# init controllers
from . import controllers