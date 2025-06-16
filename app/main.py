import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import camelot
import pandas as pd
from fastapi import (BackgroundTasks, FastAPI, File, Form, HTTPException,
                     UploadFile)
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.requests import Request

from alg.genetic import analyze

os.makedirs("uploads", exist_ok=True)
os.makedirs("parsed", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("static/files", exist_ok=True)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static", html=True), name="static")
templates = Jinja2Templates(directory="templates")


class AnalyzerParams(BaseModel):
    """
    Класс параметров анализатора.
    """
    checkbox1: bool = False  # Пн
    checkbox2: bool = False  # Вт
    checkbox3: bool = False  # Ср
    checkbox4: bool = False  # Чт
    checkbox5: bool = False  # Пт
    checkbox6: bool = False  # Сб
    checkbox7: bool = False  # Вс
    parts_count: int = 13  # Количество частей для разбиения
    threshold: int = 10  # Порог для слияния частей (%)


class FileUploadStatus(BaseModel):
    """
    Класс для отслеживания состояния загрузки и обработки файла.
    """
    id: str
    file_type: str
    original_filename: str = ""
    upload_path: Optional[str] = None
    parsed_path: Optional[str] = None
    analysis_path: Optional[str] = None
    # 1: Выбор типа файла, 2: Загрузка, 3: Обработка, 4: Параметры, 5: Анализ
    current_step: int = 1
    status: str = "в ожидании"
    error_message: Optional[str] = None
    analyzer_params: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None


# Хранение состояний в памяти
file_uploads = {}


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """
    Возвращает HTML-страницу с интерфейсом загрузки файлов.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/create-process")
async def create_process(request: dict):
    """
    Создает новый процесс загрузки файла.
    """
    process_id = str(uuid.uuid4())
    file_uploads[process_id] = FileUploadStatus(
        id=process_id,
        file_type=request.get("file_type"),
        current_step=1,
        status="в ожидании"
    )
    return {"id": process_id, "status": "создан"}


@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...), process_id: str = Form(...)):
    """
    Загружает файл на сервер и сохраняет его.
    """
    if process_id not in file_uploads:
        raise HTTPException(status_code=404, detail="Process not found")

    process = file_uploads[process_id]
    process.original_filename = file.filename

    if process.file_type == "csv" and not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Expected CSV file")
    if process.file_type == "pdf" and not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Expected PDF file")

    file_path = f"uploads/{process_id}_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    process.upload_path = file_path
    process.current_step = 2
    process.status = "загружен"

    return {"status": "загружен", "filename": file.filename}


@app.get("/status/{process_id}")
async def get_status(process_id: str):
    """
    Получает статус процесса загрузки и обработки файла.
    """
    if process_id not in file_uploads:
        raise HTTPException(status_code=404, detail="Process not found")

    process = file_uploads[process_id]
    response_dict = process.dict()

    return response_dict


@app.post("/process/{process_id}")
async def process_file(process_id: str, background_tasks: BackgroundTasks):
    """
    Запускает фоновую обработку загруженного файла.
    """
    if process_id not in file_uploads:
        raise HTTPException(status_code=404, detail="Process not found")

    process = file_uploads[process_id]
    if process.current_step < 2 or not process.upload_path:
        raise HTTPException(status_code=400, detail="File not uploaded yet")

    process.status = "Обрабатывается..."
    background_tasks.add_task(background_process_file, process_id)

    return {"status": "обработка", "message": "Обработка инициализирована"}


@app.post("/set-params/{process_id}")
async def set_params(process_id: str, params: AnalyzerParams):
    """
    Устанавливает параметры анализатора для загруженного файла.
    """
    if process_id not in file_uploads:
        raise HTTPException(status_code=404, detail="Process not found")

    process = file_uploads[process_id]
    if process.current_step < 3 or not process.parsed_path:
        raise HTTPException(status_code=400, detail="File not processed yet")

    process.analyzer_params = params.model_dump()
    process.current_step = 4
    process.status = "Параметры установлены"

    return {"status": "Параметры установлены"}


@app.post("/analyze/{process_id}")
async def analyze_file(process_id: str, background_tasks: BackgroundTasks):
    """
    Запускает фоновый анализ обработанного файла.
    """
    if process_id not in file_uploads:
        raise HTTPException(status_code=404, detail="Process not found")

    process = file_uploads[process_id]
    if process.current_step < 4 or not process.analyzer_params:
        raise HTTPException(
            status_code=400, detail="Analyzer parameters not set or file not processed yet")

    process.status = "Анализируется..."
    background_tasks.add_task(background_analyze_file, process_id)

    return {"status": "analyzing", "message": "Анализ инициализирован"}


@app.get("/download/{process_id}")
async def download_file(process_id: str):
    """
    Обрабатывает запрос на скачивание файла.
    """
    if process_id not in file_uploads:
        raise HTTPException(status_code=404, detail="Process not found")

    process = file_uploads[process_id]
    if process.current_step < 3 or not process.parsed_path:
        raise HTTPException(status_code=400, detail="File not processed yet")

    file_name = os.path.basename(process.parsed_path)
    file_path = f"static/files/{file_name}"

    if not os.path.exists(file_path):
        shutil.copy(process.parsed_path, file_path)

    return FileResponse(
        path=file_path,
        filename=file_name,
        media_type="text/csv"
    )


def background_process_file(process_id: str):
    """Фоновая обработка файла"""
    process = file_uploads[process_id]

    try:
        if process.file_type == "pdf":
            try:
                output_path = f"parsed/{process_id}_{Path(process.original_filename).stem}.csv"
                tables = camelot.read_pdf(
                    process.upload_path, pages='all', flavor='network')

                if tables:
                    df_list = [table.df.iloc[8:] for table in tables]
                    df = pd.concat(df_list, ignore_index=True)

                    df = df.iloc[:, 1:]
                    split_cols = df.iloc[:, 2].str.split("\t", expand=True)
                    df = pd.concat(
                        [df.iloc[:, 0:2], split_cols, df.iloc[:, 3:]], axis=1)

                    df.columns = ['dkNum', 'directionNum', 'date', 'accumulationStartTime',
                                  'accumulationInterval', 'characteristicNumber', 'intensity']
                    df.to_csv(output_path, index=False)

                    static_path = f"static/files/{process_id}_{Path(process.original_filename).stem}.csv"
                    shutil.copy(output_path, static_path)

                    process.parsed_path = output_path
                    process.status = "обработка завершена"
                else:
                    process.status = "ошибка"
                    process.error_message = "В этом PDF нет таблиц"
            except Exception as e:
                process.status = "ошибка"
                process.error_message = f"Ошибка при парсинге PDF: {str(e)}"
        else:
            # Для CSV файлов просто копируем
            parsed_path = f"parsed/{process_id}_{Path(process.original_filename).stem}.csv"
            static_path = f"static/files/{process_id}_{Path(process.original_filename).stem}.csv"

            shutil.copy(process.upload_path, parsed_path)
            shutil.copy(process.upload_path, static_path)

            process.parsed_path = parsed_path
            process.status = "обработка завершена"

        process.current_step = 3
    except Exception as e:
        process.status = "ошибка"
        process.error_message = str(e)


def background_analyze_file(process_id: str):
    """Фоновой анализ файла"""
    process = file_uploads[process_id]

    try:
        selected_days = []
        days_mapping = {
            "checkbox1": 0,
            "checkbox2": 1,
            "checkbox3": 2,
            "checkbox4": 3,
            "checkbox5": 4,
            "checkbox6": 5,
            "checkbox7": 6
        }

        for key, day in days_mapping.items():
            if process.analyzer_params.get(key, False):
                selected_days.append(day)

        parts_count = process.analyzer_params.get("parts_count")
        threshold = process.analyzer_params.get("threshold")

        html_table, image_base64 = analyze(
            process.parsed_path, selected_days, parts_count, threshold)

        analysis_results = {
            "image_data": f"data:image/png;base64,{image_base64}",
            "summary": html_table,
        }

        analysis_path = f"results/{process_id}_analysis.json"
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)

        process.analysis_path = analysis_path
        process.results = analysis_results
        process.current_step = 5
        process.status = "анализ завершен"
    except Exception as e:
        process.status = "ошибка"
        process.error_message = str(e)

