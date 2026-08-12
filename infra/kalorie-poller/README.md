# Kalorie S3 Snapshot Poller

Docker Lambda on an EventBridge Scheduler cron at **5:00 AM** and **8:00 PM Eastern** (`America/New_York`). Each run:

1. Pulls open Kalshi earnings-mention markets (public API)
2. Fetches live OpenAI web evidence (`gpt-5.4-mini`)
3. Scores with packaged `kalorie-v6`
4. Writes `s3://{bucket}/{yyyyMMddHH}.json` (UTC hour)

No local cache. Downstream apps read the latest S3 object.

## Prerequisites

- AWS account + credentials (`aws configure` or SSO)
- Node.js 20+
- Docker running (for Lambda image build/push)
- Local workspace layout with both:
  - `kalorie2/`
  - `models/kalorie-v6/`

## Secrets to paste

| Secret | Required | Where |
|--------|----------|--------|
| `OPENAI_API_KEY` | **Yes** | Secrets Manager JSON `{"OPENAI_API_KEY":"sk-..."}` |
| AWS deploy profile | Yes | local `aws` / `cdk deploy` |
| `KALSHI_API_KEY_ID` / private key | **No** | not used for this getter+score job |

## Deploy

Use an AWS profile/SSO that can deploy to your target account/region:

```powershell
$env:AWS_PROFILE = "your-deploy-profile"   # or: aws sso login --profile ...
cd kalorie2/infra/kalorie-poller
npm install
npx cdk bootstrap   # once per account/region
npx cdk deploy
```

Docker build context is the parent workspace that contains both `kalorie2/` and `models/kalorie-v6/` (detected automatically).

After deploy, copy the `OpenAiSecretArn` output and set the real key (use Python so the JSON stays valid on Windows):

```powershell
python -c "import json,os,subprocess; subprocess.check_call(['aws','secretsmanager','put-secret-value','--secret-id','<OpenAiSecretArn>','--secret-string',json.dumps({'OPENAI_API_KEY':os.environ['OPENAI_API_KEY']})])"
```

Invoke once immediately:

```powershell
aws lambda invoke `
  --function-name "<PollerFunctionName>" `
  --payload "{}" `
  out.json
Get-Content out.json
```

Confirm the object:

```powershell
aws s3 ls "s3://<SnapshotBucketName>/"
aws s3 cp "s3://<SnapshotBucketName>/<yyyymmddHH>.json" -
```

## Consumer contract

Object key: `YYYYMMDDHH.json` (UTC), e.g. `2026081200.json`.

```json
{
  "snapshot_id": "2026081200",
  "generated_at": "2026-08-12T00:05:12Z",
  "model_name": "kalorie-v6",
  "market_count": 123,
  "prediction_count": 123,
  "markets": [{ "market_ticker": "...", "yes_bid": 0.3, "yes_ask": 0.32, "yes_mid": 0.31 }],
  "predictions": [{
    "market_ticker": "...",
    "model_probability": 0.28,
    "market_probability": 0.35,
    "yes_bid": 0.33,
    "yes_ask": 0.37,
    "delta": -0.07,
    "abs_delta": 0.07
  }]
}
```

Also written as `latest.json` for the desk app. Trade side / stake are **not** in the snapshot; compute them in `kalorie-desk`.

Public base URL output: `SnapshotPublicBaseUrl`.

## Local package tests

```powershell
cd kalorie2
$env:PYTHONPATH="src"
python -m pytest tests/test_s3_snapshot.py -q
```

## Notes

- Lambda timeout is 15 minutes for live web evidence across many events.
- Historical training CSV is not baked into the image; only model weights, runtime, seed JSON, and web-evidence packets are.
- Do not commit `.env`, `kalshi.rsa`, or real API keys into the image or this repo.
