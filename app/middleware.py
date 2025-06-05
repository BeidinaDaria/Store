from flask import request
from werkzeug.wsgi import ClosingIterator
from datetime import datetime

class PacketAnalyzerMiddleware:
    def __init__(self, app, security_module):
        self.app = app
        self.i=None
        self.t=datetime.now()
        self.security_module = security_module

    def __call__(self, environ, start_response):
        protocol_type={'http/1.1':0, 'http/2':1, 'https/1.1':2,'https/2':3, 'http':1, 'https':3, '':-1}
        c_type={'application/javascript':0, 'application/json':1, 'image/jpeg':2, 'image/png':3, 'text/css':4, 'text/html':5, 'text/plain': 6, 'multipart/form-data':7, 'application/x-www-form-urlencoded':8, '':-1}
        c_charset={'utf-8':0, 'windows-1251':1, 'iso-8859-1':2, 'utf-16':0, 'utf-32':0, '':-1}
        methods={'get':0, 'post':1, 'put':2, 'delete':3, 'patch':4, 'head':5}
        paths=['product', 'catalog', 'favourites', 'basket', 'profile', 'login', 'signup', 'logout', 'add', 'remove']
        encodings={'gzip':0, 'gzip, deflate':1, 'deflate':2, 'br':3, 'identity':4, 'compress':5, '':-1}
        # Извлечение данных из environ (сырой запрос)
        client_ip = environ.get('REMOTE_ADDR')
        server_protocol = str(environ.get('SERVER_PROTOCOL'))
        content_length = float(environ.get('CONTENT_LENGTH', 0))
        if environ.get('CONTENT_TYPE'):
            content_type, content_charset = str(environ.get('CONTENT_TYPE')).split('; charset=')
            content_type=c_type[content_type.lower()] if (content_type.lower() in c_type.keys()) else 8
            content_charset=c_charset[content_charset.lower()] if (content_charset.lower() in c_charset.keys()) else 4
        else:
            content_type, content_charset = (-1,-1)
        method = environ.get('REQUEST_METHOD')
        path = '/'+str(environ.get('PATH_INFO')).split('/')[-1]
        if path=='/':
            p=0
        else:
            p=11
            for i in range(0,len(paths)):
                if paths[i] in path:
                    p=i+1
                    break
        encoding = environ.get('HTTP_ACCEPT_ENCODING')
        d=datetime.now()
        prev=d
        d=(d-self.t).total_seconds()
        self.t=prev
        if 'SELECT' in str(environ.get('PATH_INFO')) or 'DELETE' in str(environ.get('PATH_INFO')) or 'UPDATE' in str(environ.get('PATH_INFO')) or 'OR' in str(environ.get('PATH_INFO')):
            start_response('403 Forbidden', [('Content-Type', 'text/plain')])
            return [b'Blocked by security policy']
        if (self.i==client_ip and ('jpeg' in path or 'css' in path or 'svg' in path or 'ttf' in path or 'png' in path)):
            return self.app(environ, start_response)
        # Формирование метрик
        metrics = {
            'protocol': protocol_type[server_protocol.lower()] if (server_protocol.lower() in protocol_type.keys()) else 4,
            'port':int(str(environ.get('REMOTE_PORT', 0)).split(':')[-1]),
            'content_length':content_length,
            'content_type':content_type,
            'content_charset':content_charset,
            'method':methods[method.lower()] if (method.lower() in methods.keys()) else 6,
            'path':p,
            'encoding':encodings[encoding.lower()] if (encoding.lower() in encodings.keys()) else 6,
            'duration':d
        }
        print(metrics)
        # Проверка запроса (если False — безопасен)
        if self.security_module.check(client_ip, metrics):
            start_response('403 Forbidden', [('Content-Type', 'text/plain')])
            return [b'Blocked by security policy']
        self.i=client_ip
        # Передача управления Flask
        return self.app(environ, start_response)