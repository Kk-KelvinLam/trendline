FROM python:3.12-slim

WORKDIR /app

COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

COPY app ./app
COPY src ./src
RUN mkdir -p data/artifacts
COPY data/artifacts/cards.json \
     data/artifacts/cards_shared.json \
     data/artifacts/cards_sector.json \
     data/artifacts/cards_stock.json \
     data/artifacts/metrics.json \
     data/artifacts/scoreboard.json \
     data/artifacts/per_ticker.json \
     data/artifacts/ledger.json \
     data/artifacts/

ENV PYTHONPATH=/app/src
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_ENABLE_CORS=false

EXPOSE 8080

CMD streamlit run app/streamlit_app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true
