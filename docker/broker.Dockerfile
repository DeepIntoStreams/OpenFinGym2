# Broker image: runs the market and trading stack behind the HTTP shell.
# Build from the repository root:
#   docker build -t openfingym-broker -f docker/broker.Dockerfile .
FROM python:3.12-slim

RUN pip install --no-cache-dir \
    numpy==2.0.2 \
    pandas==2.3.3 \
    scipy==1.15.3 \
    scikit-learn==1.7.2 \
    torch==2.11.0 \
    pydantic \
    requests==2.32.3 \
    websockets \
    fastapi \
    uvicorn

WORKDIR /broker
COPY src/open_fin_gym/realtime/ open_fin_gym/realtime/
COPY src/open_fin_gym/broker/ open_fin_gym/broker/
RUN touch open_fin_gym/__init__.py
ENV PYTHONPATH=/broker

CMD ["uvicorn", "--factory", "open_fin_gym.broker.server:create_app", "--host", "0.0.0.0", "--port", "8000"]
