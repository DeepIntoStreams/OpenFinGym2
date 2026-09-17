# Market and trading stack behind the broker. Build from the repository root:
#   docker build -t openfingym-broker -f images/broker.Dockerfile .
FROM python:3.12-slim

RUN pip install --no-cache-dir \
    numpy==2.0.2 \
    pandas==2.3.3 \
    scipy==1.15.3 \
    pydantic \
    requests==2.32.3 \
    websockets==15.0.1 \
    fastapi==0.141.1 \
    uvicorn==0.52.4 \
    alpaca-py==0.44.0 \
    polymarket-client==0.10.0

WORKDIR /broker
COPY src/open_fin_gym/realtime/ open_fin_gym/realtime/
COPY src/open_fin_gym/broker/ open_fin_gym/broker/
RUN touch open_fin_gym/__init__.py
ENV PYTHONPATH=/broker

CMD ["uvicorn", "--factory", "open_fin_gym.broker.server:create_app", "--host", "0.0.0.0", "--port", "8000"]
