# Build from the kalshi workspace root:
#   docker build -f kalorie2/lambda_poller/Dockerfile -t kalorie-poller .
FROM public.ecr.aws/lambda/python:3.12

# Package layout so saved model runtime can resolve kalorie2/src:
#   /opt/kalorie2/src/kalorie2/...
#   /opt/kalorie2/data/...
#   /opt/models/kalorie-v6/...
#   /var/task/handler.py
COPY kalorie2/src/kalorie2 /opt/kalorie2/src/kalorie2
COPY kalorie2/data/company_metadata.json /opt/kalorie2/data/company_metadata.json
COPY kalorie2/data/phrase_semantic_embeddings.json /opt/kalorie2/data/phrase_semantic_embeddings.json
COPY kalorie2/lambda_poller/handler.py ${LAMBDA_TASK_ROOT}/handler.py
COPY kalorie2/lambda_poller/requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt

RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Trimmed kalorie-v6: artifacts + runtime + seed/web packets (no historical CSV).
COPY models/kalorie-v6/README.md /opt/models/kalorie-v6/README.md
COPY models/kalorie-v6/artifacts /opt/models/kalorie-v6/artifacts
COPY models/kalorie-v6/runtime /opt/models/kalorie-v6/runtime
COPY models/kalorie-v6/training/data /opt/models/kalorie-v6/training/data
COPY models/kalorie-v6/training/web-evidence /opt/models/kalorie-v6/training/web-evidence

ENV PYTHONPATH=/opt/kalorie2/src
ENV MODELS_ROOT=/opt/models
ENV MODEL_NAME=kalorie-v6
ENV LIVE_WEB_EVIDENCE=true
ENV WEB_SEARCH_MODEL=gpt-5.4-mini

CMD ["handler.handler"]
