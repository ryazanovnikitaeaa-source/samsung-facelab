"""Local HTTP API and static user interface."""
import csv
import json
from io import StringIO
import os
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .matching import DEFAULT_THRESHOLD
from .service import FaceService
from .storage import Store
from .vision import FaceEngine, MAX_FILE_BYTES, ROOT
from .reporting import render_report
from . import __version__


def create_app(service=None):
    app=FastAPI(title='FaceLab',version=__version__,docs_url=None,redoc_url=None)
    service=service or FaceService(Store(Path(os.environ.get('FACELAB_DATA_DIR',ROOT/'data'))/'facelab.sqlite3'),FaceEngine())
    app.state.service=service
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=['127.0.0.1','localhost','testserver'])

    @app.middleware('http')
    async def local_only(request: Request, call_next):
        if request.method not in {'GET','HEAD','OPTIONS'}:
            origin=request.headers.get('origin')
            if origin:
                parsed=urlsplit(origin)
                if parsed.scheme!='http' or parsed.netloc!=request.headers.get('host'):
                    return JSONResponse({'detail':'Запрос разрешён только со страницы локального приложения.'},status_code=403)
            if request.headers.get('sec-fetch-site')=='cross-site':
                return JSONResponse({'detail':'Межсайтовый запрос отклонён.'},status_code=403)
            try:
                content_length=int(request.headers.get('content-length','0'))
            except ValueError:
                return JSONResponse({'detail':'Некорректная длина запроса.'},status_code=400)
            if content_length>50*1024*1024:
                return JSONResponse({'detail':'Вся загрузка должна быть меньше 50 МБ.'},status_code=413)
        response=await call_next(request)
        response.headers['Cache-Control']='no-store'
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='no-referrer'
        response.headers.setdefault('Content-Security-Policy',"default-src 'self'; img-src 'self' data: blob:; script-src 'self'; style-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'self'")
        return response

    @app.exception_handler(ValueError)
    async def bad_input(request,exc):
        return JSONResponse({'detail':str(exc)},status_code=400)

    @app.exception_handler(KeyError)
    async def missing(request,exc):
        return JSONResponse({'detail':'Человек не найден.'},status_code=404)

    @app.exception_handler(RuntimeError)
    async def not_ready(request,exc):
        return JSONResponse({'detail':str(exc)},status_code=503)

    @app.exception_handler(RequestValidationError)
    async def invalid_form(request,exc):
        return JSONResponse({'detail':'Проверьте заполнение полей и выбранные файлы.'},status_code=422)

    async def read_files(files):
        if not 1<=len(files)<=20:
            raise ValueError('Выберите от 1 до 20 файлов.')
        result=[]
        for file in files:
            try:
                content=await file.read(MAX_FILE_BYTES+1)
                if len(content)>MAX_FILE_BYTES:
                    raise ValueError('Каждый файл должен быть меньше 10 МБ.')
                filename=(file.filename or 'image').replace('\\','/').split('/')[-1][:120]
                result.append((filename,content))
            finally:
                await file.close()
        return result

    @app.get('/')
    def home():
        return FileResponse(ROOT/'static'/'index.html')

    @app.get('/api/status')
    def status():
        people=service.store.people()
        return {'ready':getattr(service.engine,'ready',True),'people_count':len(people),
                'samples_count':sum(p['samples'] for p in people),'default_threshold':DEFAULT_THRESHOLD,
                'models':{'detection':'YuNet','recognition':'SFace'},'version':__version__}

    @app.get('/api/people')
    def people():
        return {'people':service.store.people()}

    @app.post('/api/people')
    async def enroll(name: str=Form(...),files: list[UploadFile]=File(...),face_choices: str|None=Form(None)):
        from starlette.concurrency import run_in_threadpool
        data=await read_files(files)
        choices=None
        if face_choices is not None:
            try:
                choices=json.loads(face_choices)
            except (ValueError, RecursionError) as exc:
                raise ValueError('Некорректный выбор лиц. Повторите предварительную проверку снимков.') from exc
            if not isinstance(choices,list):
                raise ValueError('Некорректный выбор лиц. Повторите предварительную проверку снимков.')
        person=await run_in_threadpool(service.enroll,name,data,choices)
        return {'person':person}

    @app.get('/api/people/{person_id}/thumbnail')
    def thumbnail(person_id: str):
        data=service.store.thumbnail(person_id)
        if data is None:
            raise KeyError(person_id)
        return Response(data,media_type='image/jpeg')

    @app.delete('/api/people/{person_id}')
    def delete(person_id: str):
        service.delete(person_id)
        return {'ok':True}

    @app.post('/api/recognize')
    async def recognize(file: UploadFile=File(...),threshold: float=Form(DEFAULT_THRESHOLD)):
        from starlette.concurrency import run_in_threadpool
        data=(await read_files([file]))[0][1]
        return await run_in_threadpool(service.recognize,data,threshold)

    @app.post('/api/quality')
    async def inspect_quality(files: list[UploadFile]=File(...)):
        from starlette.concurrency import run_in_threadpool
        data=await read_files(files)
        return await run_in_threadpool(service.inspect,data)

    @app.post('/api/evaluate')
    async def evaluate(files: list[UploadFile]=File(...),expected_id: str=Form(''),threshold: float=Form(DEFAULT_THRESHOLD),purpose: str=Form('test')):
        from starlette.concurrency import run_in_threadpool
        data=await read_files(files)
        return await run_in_threadpool(service.evaluate,data,expected_id or None,threshold,purpose)

    @app.get('/api/report')
    def report():
        return service.report()

    @app.get('/api/report.html')
    def printable_report(author: str=''):
        return Response(render_report(service.report(),author=author),media_type='text/html',
                        headers={'Content-Security-Policy':"default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; frame-ancestors 'none'"})

    @app.get('/api/report.csv')
    def report_csv():
        output=StringIO(newline='')
        columns=['filename','expected_name','predicted_name','correct','similarity','threshold','elapsed_ms','purpose']
        writer=csv.writer(output)
        writer.writerow(columns)
        for case in service.report()['cases']:
            values=[]
            for column in columns:
                value=case.get(column,'')
                if isinstance(value,str) and value.lstrip().startswith(('=','+','-','@','\t','\r')):
                    value="'"+value
                values.append(value)
            writer.writerow(values)
        return Response('\ufeff'+output.getvalue(),media_type='text/csv',
                        headers={'Content-Disposition':'attachment; filename="facelab-evaluation.csv"'})

    @app.delete('/api/report')
    def clear_report():
        with service.lock:
            service.store.clear_evaluations()
        return service.report()

    app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')
    return app
