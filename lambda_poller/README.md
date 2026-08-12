# Lambda poller image

Docker entrypoint for the 6h S3 snapshot job.

- Handler: [`handler.py`](handler.py) → `kalorie2.s3_snapshot.handler_from_event`
- Image build context: parent workspace root (must include `kalorie2/` and `models/kalorie-v6/`)
- Infra / deploy docs: [`../infra/kalorie-poller/README.md`](../infra/kalorie-poller/README.md)

```powershell
# from workspace root (parent of kalorie2)
docker build -f kalorie2/lambda_poller/Dockerfile -t kalorie-poller .
```
