FROM python:3.11

WORKDIR /workdir

COPY requirements.txt /workdir/
COPY app/ /workdir/app/
COPY alg/ /workdir/alg/
COPY static/ /workdir/static/
COPY templates/ /workdir/templates/

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]