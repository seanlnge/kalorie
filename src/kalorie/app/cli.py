import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from time import monotonic
from typing import Annotated

import httpx
import typer
from sklearn.metrics import log_loss

from kalorie.benchmarking.packs import (
    BenchmarkEvent,
    BenchmarkEvidenceDocument,
    BenchmarkMarket,
    BenchmarkPack,
    BenchmarkPackManifest,
    BenchmarkRunMetadata,
    BenchmarkSnapshot,
    validate_benchmark_pack,
)
from kalorie.benchmarking.runner import run_model1_pack_benchmark
from kalorie.clients.defeatbeta import (
    DefeatBetaApiError,
    DefeatBetaArticle,
    DefeatBetaNewsClient,
)
from kalorie.clients.financial_modeling_prep import (
    FinancialModelingPrepClient,
    FmpApiAuthError,
    FmpApiParseError,
    FmpApiRateLimitError,
    FmpApiSubscriptionError,
    FmpTranscriptReference,
)
from kalorie.clients.kalshi import KalshiEarningsMarketsClient, KalshiPublicClient
from kalorie.clients.newsdata import (
    NewsDataAuthError,
    NewsDataClient,
    NewsDataRateLimitError,
    NewsDataSubscriptionError,
)
from kalorie.clients.sec_api import (
    SecApiClient,
    SecApiError,
    SecApiRateLimitError,
    select_best_company_mapping,
)
from kalorie.clients.tiingo import (
    TiingoApiError,
    TiingoArticle,
    TiingoAuthError,
    TiingoNewsClient,
    TiingoRateLimitError,
)
from kalorie.data_cleaning import normalize_and_dedupe_phrases
from kalorie.data_grepping import (
    EventScenarioCatalog,
    OpenAIEventScenarioGenerator,
    OpenAITemplatePhraseGenerator,
    TemplatePhraseCatalog,
    load_material_snippets,
)
from kalorie.domain.config import Settings
from kalorie.domain.models import (
    DocumentChunk,
    FeatureVector,
    MarketSnapshot,
    MentionLabel,
    MentionMarketContract,
    SourceDocument,
    TargetPhrase,
)
from kalorie.io.documents import content_hash, ingest_local_press_release, ingest_local_transcript
from kalorie.io.public_documents import PublicDocumentManifest, collect_public_document
from kalorie.io.transcript_corpus import scan_transcript_corpus
from kalorie.kalshi_pull import pull_historical_mention_contracts
from kalorie.market.hedging import build_hedge_plan
from kalorie.market.markets import MentionMarketParseError, parse_mention_market_title
from kalorie.market.paper import compare_prediction_to_market
from kalorie.ml.datasets import HistoricalTrainingExample
from kalorie.ml.embeddings import CachedEmbeddingProvider, OpenAIEmbeddingProvider
from kalorie.ml.evaluation import evaluate_binary_predictions
from kalorie.ml.features import extract_feature_vectors
from kalorie.ml.labeling import label_document_chunks
from kalorie.ml.model1 import (
    CompanyRetrainedModelArtifact,
    MentionModelArtifact,
    optimize_model1_for_brier,
    predict_company_model1,
    predict_model1,
    train_company_model1,
    train_model1,
)
from kalorie.ml.modeling import RuleBasedBaseline
from kalorie.ml.real_training_data import (
    DEFAULT_SYNTHETIC_TARGET_PHRASES,
    build_examples_from_transcript_records,
    build_synthetic_phrase_examples_from_transcript_records,
    source_document_from_text_file,
)
from kalorie.ml.synthetic_phrases import generate_synthetic_phrase_candidates
from kalorie.ml.training import train_and_evaluate
from kalorie.workflows.event_dossiers import (
    generate_event_dossiers,
    load_event_dossier_catalogs,
)
from kalorie.workflows.evidence_collection import (
    collect_historical_news_manifests as collect_historical_news_manifests_for_examples,
)
from kalorie.workflows.evidence_collection import (
    collect_sec_ex99_manifests,
    merge_document_manifests,
    plan_sec_requests,
)
from kalorie.workflows.historical_synthetic import (
    build_historical_synthetic_rows,
    build_transcript_inventory,
)
from kalorie.workflows.kalshi_event_pack import (
    KalshiEventPackCandidate,
    build_event_pack,
)
from kalorie.workflows.models import PhraseCatalog
from kalorie.workflows.phrase_generation import (
    build_phrase_catalog,
    generate_openai_phrase_response,
)
from kalorie.workflows.real_event_rows import build_real_event_pack_training_rows
from kalorie.workflows.verification import verify_event_pack_artifacts

app = typer.Typer()

INITIAL_MARKET_PHRASES = [
    "traffic",
    "same restaurant sales",
    "digital revenue",
    "geopolitical uncertainty",
    "value proposition",
    "margin",
]


@app.callback()
def main() -> None:
    """Local Kalorie research workflows."""


@app.command()
def run_local_cava(
    pdf: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    market_title: Annotated[str, typer.Option()],
    yes_bid: Annotated[str, typer.Option()],
    yes_ask: Annotated[str, typer.Option()],
    out: Annotated[Path, typer.Option()],
) -> None:
    try:
        parsed_market = parse_mention_market_title(market_title)
        bid = _parse_probability(yes_bid, "yes-bid")
        ask = _parse_probability(yes_ask, "yes-ask")
    except (MentionMarketParseError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    document, chunks = ingest_local_press_release(
        pdf,
        company_symbol="CAVA",
        fiscal_year=2026,
        fiscal_quarter=1,
        published_at=datetime(2026, 5, 19, tzinfo=UTC),
    )
    _write_market_workflow_artifacts(
        document=document,
        chunks=chunks,
        market_title=market_title,
        company_symbol=parsed_market.company_symbol,
        market_target=parsed_market.target_phrase,
        yes_bid=bid,
        yes_ask=ask,
        out=out,
    )
    typer.echo(f"Wrote local CAVA run artifacts to {out}")


@app.command()
def run_local_transcript(
    transcript: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    market_title: Annotated[str, typer.Option()],
    yes_bid: Annotated[str, typer.Option()],
    yes_ask: Annotated[str, typer.Option()],
    out: Annotated[Path, typer.Option()],
) -> None:
    try:
        parsed_market = parse_mention_market_title(market_title)
        bid = _parse_probability(yes_bid, "yes-bid")
        ask = _parse_probability(yes_ask, "yes-ask")
    except (MentionMarketParseError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    document, chunks = ingest_local_transcript(
        transcript,
        company_symbol=parsed_market.company_symbol,
        fiscal_year=2026,
        fiscal_quarter=1,
        published_at=datetime(2026, 5, 19, tzinfo=UTC),
    )
    _write_market_workflow_artifacts(
        document=document,
        chunks=chunks,
        market_title=market_title,
        company_symbol=parsed_market.company_symbol,
        market_target=parsed_market.target_phrase,
        yes_bid=bid,
        yes_ask=ask,
        out=out,
    )
    typer.echo(f"Wrote local transcript run artifacts to {out}")



@app.command()
def evaluate_run(
    run: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    out: Annotated[Path, typer.Option()],
) -> None:
    labels = [
        MentionLabel.model_validate(row)
        for row in json.loads((run / "labels.json").read_text(encoding="utf-8"))
    ]
    features = [
        FeatureVector.model_validate(row)
        for row in json.loads((run / "features.json").read_text(encoding="utf-8"))
    ]
    predictions = [RuleBasedBaseline().predict_proba(feature) for feature in features]
    report = evaluate_binary_predictions(
        predictions,
        labels,
        evaluation_kind="smoke",
        trained_model=False,
    )

    _write_json(out, report.model_dump(mode="json"))
    typer.echo(f"Brier score: {report.brier_score:.6f}")
    typer.echo(f"ECE (10-bin): {report.expected_calibration_error:.6f}")
    typer.echo("Training: smoke-only metric; not statistically meaningful training performance.")


@app.command("build-hedge-plan")
def build_hedge_plan_command(
    predictions: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option()],
    budget: Annotated[float, typer.Option()] = 100.0,
    model_probability_key: Annotated[str, typer.Option()] = "model_company_probability",
    min_edge: Annotated[float, typer.Option()] = 0.0,
    risk_aversion: Annotated[float, typer.Option()] = 0.50,
    max_fraction_per_market: Annotated[float, typer.Option()] = 0.35,
    max_positions: Annotated[int | None, typer.Option()] = None,
    force_full_deployment: Annotated[bool, typer.Option()] = False,
) -> None:
    payload = json.loads(predictions.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("rows", [])
        context = {
            "event_ticker": payload.get("event_ticker"),
            "generated_at": payload.get("generated_at"),
            "source_document": payload.get("source_document"),
            "predictions_path": str(predictions),
        }
    elif isinstance(payload, list):
        rows = payload
        context = {"predictions_path": str(predictions)}
    else:
        raise typer.BadParameter("predictions must be either a JSON object with rows or a JSON array")

    if not isinstance(rows, list) or not rows:
        raise typer.BadParameter("predictions payload does not contain any rows")
    try:
        hedge = build_hedge_plan(
            rows,
            budget=budget,
            model_probability_key=model_probability_key,
            min_edge=min_edge,
            risk_aversion=risk_aversion,
            max_fraction_per_market=max_fraction_per_market,
            force_full_deployment=force_full_deployment,
            max_positions=max_positions,
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    result = {
        "context": context,
        **hedge,
    }
    _write_json(out, result)
    typer.echo(
        f"Hedge plan written to {out} "
        f"(positions={result['position_count']}, deployed=${result['deployed_dollars']:.2f}, "
        f"expected_profit=${result['expected_profit_dollars']:.2f}, "
        f"stdev=${result['stdev_dollars']:.2f})"
    )


@app.command()
def discover_kalshi_words(
    event_ticker: Annotated[str, typer.Option()],
    out: Annotated[Path, typer.Option()],
    base_url: Annotated[str, typer.Option()] = "https://api.elections.kalshi.com/trade-api/v2",
) -> None:
    with httpx.Client(timeout=30) as http_client:
        client = KalshiPublicClient(http_client=http_client, base_url=base_url)
        contracts = client.get_event_mention_markets(event_ticker)
    _write_json(out, [contract.model_dump(mode="json") for contract in contracts])
    typer.echo(f"Wrote {len(contracts)} Kalshi mention markets to {out}")


@app.command("discover-company-earnings-markets")
def discover_company_earnings_markets(
    company_symbol: Annotated[str, typer.Option()],
    out: Annotated[Path, typer.Option()],
    status: Annotated[str, typer.Option()] = "open",
    limit: Annotated[int, typer.Option()] = 400,
    max_pages: Annotated[int, typer.Option()] = 10,
    event_ticker: Annotated[str | None, typer.Option()] = None,
    base_url: Annotated[str, typer.Option()] = "https://api.elections.kalshi.com/trade-api/v2",
) -> None:
    with httpx.Client(timeout=30) as http_client:
        client = KalshiEarningsMarketsClient(http_client=http_client, base_url=base_url)
        contracts = client.list_company_mention_markets(
            company_symbol=company_symbol,
            status=status,
            limit=limit,
            max_pages=max_pages,
            event_ticker=event_ticker,
        )
        latest_event_ticker = client.get_latest_company_event_ticker(
            company_symbol=company_symbol,
            status=status,
            limit=limit,
            max_pages=max_pages,
        )
        event_tickers = client.list_company_event_tickers(
            company_symbol=company_symbol,
            status=status,
            limit=limit,
            max_pages=max_pages,
        )
    payload = {
        "company_symbol": company_symbol.upper(),
        "status": status,
        "event_ticker": event_ticker,
        "latest_event_ticker": latest_event_ticker,
        "event_tickers": event_tickers,
        "market_count": len(contracts),
        "markets": [contract.model_dump(mode="json") for contract in contracts],
    }
    _write_json(out, payload)
    typer.echo(
        f"Wrote {len(contracts)} company earnings mention markets for "
        f"{company_symbol.upper()} to {out}"
    )


@app.command()
def collect_kalshi_historical_markets(
    raw_out: Annotated[Path, typer.Option()],
    mentions_out: Annotated[Path, typer.Option()],
    search: Annotated[str | None, typer.Option()] = None,
    status: Annotated[str, typer.Option()] = "closed",
    limit: Annotated[int, typer.Option()] = 100,
    cursor: Annotated[str | None, typer.Option()] = None,
    base_url: Annotated[str, typer.Option()] = "https://api.elections.kalshi.com/trade-api/v2",
) -> None:
    with httpx.Client(timeout=30) as http_client:
        client = KalshiPublicClient(http_client=http_client, base_url=base_url)
        payload, contracts = pull_historical_mention_contracts(
            client,
            status=status,
            search=search,
            limit=limit,
            cursor=cursor,
        )
    _write_json(raw_out, payload)
    _write_json(mentions_out, [contract.model_dump(mode="json") for contract in contracts])
    typer.echo(
        f"Wrote {len(payload.get('markets', []))} raw markets and "
        f"{len(contracts)} mention markets"
    )


@app.command()
def collect_public_material(
    url: Annotated[str, typer.Option()],
    company_symbol: Annotated[str, typer.Option()],
    fiscal_year: Annotated[int, typer.Option()],
    fiscal_quarter: Annotated[int, typer.Option()],
    source_type: Annotated[str, typer.Option()],
    published_at: Annotated[datetime, typer.Option()],
    raw_dir: Annotated[Path, typer.Option()],
    manifest_out: Annotated[Path, typer.Option()],
) -> None:
    with httpx.Client(timeout=30, follow_redirects=True) as http_client:
        manifest = collect_public_document(
            url=url,
            company_symbol=company_symbol,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            source_type=source_type,
            published_at=published_at,
            fetched_at=datetime.now(tz=UTC),
            raw_dir=raw_dir,
            http_client=http_client,
        )
    _write_json(manifest_out, manifest.model_dump(mode="json"))
    typer.echo(f"Wrote public document manifest to {manifest_out}")


@app.command()
def collect_newsdata_company_articles(
    company_symbol: Annotated[str, typer.Option()],
    company_name: Annotated[str, typer.Option()],
    from_date: Annotated[str, typer.Option()],
    to_date: Annotated[str, typer.Option()],
    raw_dir: Annotated[Path, typer.Option()],
    manifest_out: Annotated[Path, typer.Option()],
    language: Annotated[str, typer.Option()] = "en",
    max_articles: Annotated[int, typer.Option()] = 50,
    size: Annotated[int, typer.Option()] = 50,
    opinion_share: Annotated[float, typer.Option()] = 0.6,
    fiscal_year: Annotated[int | None, typer.Option()] = None,
    fiscal_quarter: Annotated[int | None, typer.Option()] = None,
    base_url: Annotated[str, typer.Option()] = "https://newsdata.io/api/1",
) -> None:
    try:
        parsed_from_date = datetime.fromisoformat(from_date).date()
        parsed_to_date = datetime.fromisoformat(to_date).date()
    except ValueError as exc:
        raise typer.BadParameter("from-date and to-date must use YYYY-MM-DD format") from exc
    if parsed_from_date > parsed_to_date:
        raise typer.BadParameter("from-date must be on or before to-date")
    if max_articles < 1:
        raise typer.BadParameter("max-articles must be at least 1")
    if size < 1:
        raise typer.BadParameter("size must be at least 1")
    if not 0 <= opinion_share <= 1:
        raise typer.BadParameter("opinion-share must be between 0 and 1")
    if (fiscal_year is None) != (fiscal_quarter is None):
        raise typer.BadParameter("fiscal-year and fiscal-quarter must be provided together")
    if fiscal_quarter is not None and fiscal_quarter not in {1, 2, 3, 4}:
        raise typer.BadParameter("fiscal-quarter must be between 1 and 4")

    settings = _load_settings()
    if settings.newsdata_api_key is None:
        raise typer.BadParameter("NEWSDATA_API_KEY is required to collect NewsData articles")

    symbol = company_symbol.upper()
    base_company_query = f'"{company_name}" OR {symbol}'
    opinion_query = (
        f"({base_company_query}) AND (earnings OR \"quarterly results\" OR guidance) "
        "AND (opinion OR analysis OR editorial)"
    )
    relevant_query = (
        f"({base_company_query}) AND "
        "(earnings OR \"quarterly results\" OR guidance OR revenue OR \"data center\" OR ai)"
    )
    opinion_latest_query = f"{company_name} {symbol} earnings opinion analysis"
    relevant_latest_query = f"{company_name} {symbol} earnings guidance revenue ai"
    opinion_target = max(1, round(max_articles * opinion_share))
    relevant_target = max(1, max_articles - opinion_target)
    with httpx.Client(timeout=45, follow_redirects=True) as http_client:
        client = NewsDataClient(
            api_key=settings.newsdata_api_key.get_secret_value(),
            http_client=http_client,
            base_url=base_url,
        )
        try:
            opinion_articles = client.search_archive(
                query=opinion_query,
                from_date=parsed_from_date.isoformat(),
                to_date=parsed_to_date.isoformat(),
                language=language,
                size=size,
                max_articles=opinion_target,
            )
            relevant_articles = client.search_archive(
                query=relevant_query,
                from_date=parsed_from_date.isoformat(),
                to_date=parsed_to_date.isoformat(),
                language=language,
                size=size,
                max_articles=relevant_target,
            )
        except NewsDataSubscriptionError:
            typer.echo("NewsData archive endpoint unavailable on current plan; using latest endpoint")
            latest_size = min(size, 10)
            opinion_articles = client.search_latest(
                query=opinion_latest_query,
                language=language,
                size=latest_size,
                max_articles=opinion_target,
            )
            relevant_articles = client.search_latest(
                query=relevant_latest_query,
                language=language,
                size=latest_size,
                max_articles=relevant_target,
            )
        except (NewsDataAuthError, NewsDataRateLimitError) as exc:
            raise typer.BadParameter(str(exc)) from exc

    combined = {article.article_id: article for article in opinion_articles}
    for article in relevant_articles:
        combined.setdefault(article.article_id, article)
    articles = sorted(combined.values(), key=lambda article: article.published_at, reverse=True)[
        :max_articles
    ]

    fetched_at = datetime.now(tz=UTC)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifests: list[PublicDocumentManifest] = []
    for article in articles:
        article_fiscal_year, article_fiscal_quarter = (
            (fiscal_year, fiscal_quarter)
            if fiscal_year is not None and fiscal_quarter is not None
            else _calendar_fiscal_period(article.published_at)
        )
        text = _news_article_text(article=article, company_symbol=symbol)
        digest = content_hash(text)
        reliability_score = _news_source_reliability(article.source_priority)
        flavor = "opinion" if _is_opinion_article(article) else "relevant"
        raw_path = (
            raw_dir
            / f"{company_symbol.upper()}-{article_fiscal_year}-Q{article_fiscal_quarter}-"
            f"{digest[:12]}-newsdata.txt"
        )
        raw_path.write_text(text, encoding="utf-8")
        manifests.append(
            PublicDocumentManifest(
                source_url=article.link or f"newsdata://article/{article.article_id}",
                company_symbol=company_symbol,
                fiscal_year=article_fiscal_year,
                fiscal_quarter=article_fiscal_quarter,
                source_type=(
                    f"news_article_{flavor}_reliability_{int(round(reliability_score * 100)):03d}"
                ),
                published_at=article.published_at,
                fetched_at=fetched_at,
                raw_path=str(raw_path),
                extracted_text_path=str(raw_path),
                content_hash=digest,
                extraction_method="newsdata_api",
            )
        )

    _write_json(manifest_out, [manifest.model_dump(mode="json") for manifest in manifests])
    typer.echo(
        f"Wrote {len(manifests)} NewsData article manifests for {company_symbol.upper()} to {manifest_out}"
    )


@app.command()
def collect_tiingo_company_articles(
    company_symbol: Annotated[str, typer.Option()],
    company_name: Annotated[str, typer.Option()],
    from_date: Annotated[str, typer.Option()],
    to_date: Annotated[str, typer.Option()],
    raw_dir: Annotated[Path, typer.Option()],
    manifest_out: Annotated[Path, typer.Option()],
    max_articles: Annotated[int, typer.Option()] = 400,
    use_yfinance_fallback: Annotated[bool, typer.Option()] = True,
    fiscal_year: Annotated[int | None, typer.Option()] = None,
    fiscal_quarter: Annotated[int | None, typer.Option()] = None,
    base_url: Annotated[str, typer.Option()] = "https://api.tiingo.com",
) -> None:
    try:
        parsed_from_date = datetime.fromisoformat(from_date).date()
        parsed_to_date = datetime.fromisoformat(to_date).date()
    except ValueError as exc:
        raise typer.BadParameter("from-date and to-date must use YYYY-MM-DD format") from exc
    if parsed_from_date > parsed_to_date:
        raise typer.BadParameter("from-date must be on or before to-date")
    if max_articles < 1:
        raise typer.BadParameter("max-articles must be at least 1")
    if (fiscal_year is None) != (fiscal_quarter is None):
        raise typer.BadParameter("fiscal-year and fiscal-quarter must be provided together")
    if fiscal_quarter is not None and fiscal_quarter not in {1, 2, 3, 4}:
        raise typer.BadParameter("fiscal-quarter must be between 1 and 4")

    settings = _load_settings()
    symbol = company_symbol.upper()
    articles: list[TiingoArticle] = []
    if settings.tiingo_api_key is not None:
        with httpx.Client(timeout=45, follow_redirects=True) as http_client:
            client = TiingoNewsClient(
                api_key=settings.tiingo_api_key.get_secret_value(),
                http_client=http_client,
                base_url=base_url,
            )
            try:
                articles = client.search_news(
                    ticker=symbol,
                    start_date=parsed_from_date.isoformat(),
                    end_date=parsed_to_date.isoformat(),
                    limit=max_articles,
                )
            except (TiingoAuthError, TiingoRateLimitError, TiingoApiError) as exc:
                if not use_yfinance_fallback:
                    raise typer.BadParameter(str(exc)) from exc
                typer.echo(
                    f"Warning: Tiingo request failed for {symbol}; falling back to yfinance ({exc})",
                    err=True,
                )
                articles = []
    elif not use_yfinance_fallback:
        raise typer.BadParameter("TIINGO_API_KEY is required to collect Tiingo articles")

    if not articles and use_yfinance_fallback:
        typer.echo(
            f"No Tiingo articles returned for {symbol}; trying yfinance fallback for {company_name}"
        )
        articles = _collect_yfinance_news(
            company_symbol=symbol,
            company_name=company_name,
            from_date=parsed_from_date.isoformat(),
            to_date=parsed_to_date.isoformat(),
            max_articles=max_articles,
        )

    fetched_at = datetime.now(tz=UTC)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifests: list[PublicDocumentManifest] = []
    for article in articles[:max_articles]:
        article_fiscal_year, article_fiscal_quarter = (
            (fiscal_year, fiscal_quarter)
            if fiscal_year is not None and fiscal_quarter is not None
            else _calendar_fiscal_period(article.published_at)
        )
        text = _news_article_text(article=article, company_symbol=symbol)
        digest = content_hash(text)
        reliability_score = _tiingo_source_reliability(article.source_name)
        flavor = "opinion" if _is_opinion_article(article) else "relevant"
        raw_path = (
            raw_dir
            / f"{symbol}-{article_fiscal_year}-Q{article_fiscal_quarter}-{digest[:12]}-tiingo.txt"
        )
        raw_path.write_text(text, encoding="utf-8")
        manifests.append(
            PublicDocumentManifest(
                source_url=article.link or f"tiingo://article/{article.article_id}",
                company_symbol=symbol,
                fiscal_year=article_fiscal_year,
                fiscal_quarter=article_fiscal_quarter,
                source_type=(
                    f"news_article_{flavor}_reliability_{int(round(reliability_score * 100)):03d}"
                ),
                published_at=article.published_at,
                fetched_at=fetched_at,
                raw_path=str(raw_path),
                extracted_text_path=str(raw_path),
                content_hash=digest,
                extraction_method="tiingo_api",
            )
        )

    _write_json(manifest_out, [manifest.model_dump(mode="json") for manifest in manifests])
    typer.echo(f"Wrote {len(manifests)} Tiingo article manifests for {symbol} to {manifest_out}")


@app.command()
def collect_defeatbeta_pre_earnings_week_articles(
    company_symbol: Annotated[str, typer.Option()],
    company_name: Annotated[str, typer.Option()],
    transcript_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    raw_dir: Annotated[Path, typer.Option()],
    manifest_out: Annotated[Path, typer.Option()],
    days_before_call: Annotated[int, typer.Option()] = 7,
    max_articles_per_event: Annotated[int, typer.Option()] = 80,
    max_total_articles: Annotated[int, typer.Option()] = 1200,
    use_yfinance_fallback: Annotated[bool, typer.Option()] = True,
    dataset_url: Annotated[str | None, typer.Option()] = None,
) -> None:
    if days_before_call < 1:
        raise typer.BadParameter("days-before-call must be at least 1")
    if max_articles_per_event < 1:
        raise typer.BadParameter("max-articles-per-event must be at least 1")
    if max_total_articles < 1:
        raise typer.BadParameter("max-total-articles must be at least 1")

    symbol = company_symbol.upper()
    records = [
        record for record in scan_transcript_corpus(transcript_root) if record.company_symbol == symbol
    ]
    if not records:
        raise typer.BadParameter(f"No transcript records found for {symbol}")

    client = DefeatBetaNewsClient(dataset_url=dataset_url)

    fetched_at = datetime.now(tz=UTC)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifests: list[PublicDocumentManifest] = []
    dedupe: set[str] = set()
    windows: list[tuple[object, datetime, str, str]] = []
    for record in sorted(records, key=lambda row: (row.fiscal_year, row.fiscal_quarter)):
        call_time = _estimated_call_time_for_period(
            fiscal_year=record.fiscal_year,
            fiscal_quarter=record.fiscal_quarter,
        )
        start_date = (call_time - timedelta(days=days_before_call)).date().isoformat()
        end_date = call_time.date().isoformat()
        windows.append((record, call_time, start_date, end_date))

    all_articles: list[DefeatBetaArticle] = []
    try:
        all_articles = client.search_stock_news(
            symbol=symbol,
            start_date=min(start for _, _, start, _ in windows),
            end_date=max(end for _, _, _, end in windows),
            max_rows=max(max_total_articles * 2, max_articles_per_event * len(windows)),
        )
    except DefeatBetaApiError as exc:
        if not use_yfinance_fallback:
            raise typer.BadParameter(str(exc)) from exc
        typer.echo(
            f"Warning: DefeatBeta request failed for {symbol} full-range pull; "
            f"falling back to yfinance windows ({exc})",
            err=True,
        )
        all_articles = []

    for record, call_time, start_date, end_date in windows:
        articles = [
            article
            for article in all_articles
            if start_date <= article.published_at.date().isoformat() <= end_date
        ][:max_articles_per_event]
        if not articles and use_yfinance_fallback:
            fallback_rows = _collect_yfinance_news(
                company_symbol=symbol,
                company_name=company_name,
                from_date=start_date,
                to_date=end_date,
                max_articles=max_articles_per_event,
            )
            articles = [
                DefeatBetaArticle(
                    article_id=row.article_id,
                    title=row.title,
                    link=row.link,
                    source_name=row.source_name,
                    description=row.description,
                    content=row.content,
                    datatype=row.datatype,
                    published_at=row.published_at,
                    tickers=row.tickers,
                    tags=row.tags,
                )
                for row in fallback_rows
            ]

        for article in articles:
            if len(manifests) >= max_total_articles:
                break
            if article.published_at > call_time:
                continue
            article_key = (
                f"{symbol}-{record.fiscal_year}-Q{record.fiscal_quarter}-{article.article_id}"
            )
            if article_key in dedupe:
                continue
            dedupe.add(article_key)
            text = _news_article_text(article=article, company_symbol=symbol)
            digest = content_hash(text)
            reliability_score = _defeatbeta_source_reliability(article.source_name)
            flavor = "opinion" if _is_opinion_article(article) else "relevant"
            raw_path = (
                raw_dir
                / f"{symbol}-{record.fiscal_year}-Q{record.fiscal_quarter}-{digest[:12]}-defeatbeta.txt"
            )
            raw_path.write_text(text, encoding="utf-8")
            manifests.append(
                PublicDocumentManifest(
                    source_url=article.link or f"defeatbeta://article/{article.article_id}",
                    company_symbol=symbol,
                    fiscal_year=record.fiscal_year,
                    fiscal_quarter=record.fiscal_quarter,
                    source_type=(
                        f"news_article_{flavor}_reliability_"
                        f"{int(round(reliability_score * 100)):03d}"
                    ),
                    published_at=article.published_at,
                    fetched_at=fetched_at,
                    raw_path=str(raw_path),
                    extracted_text_path=str(raw_path),
                    content_hash=digest,
                    extraction_method="defeatbeta_api",
                )
            )
        if len(manifests) >= max_total_articles:
            break

    _write_json(manifest_out, [manifest.model_dump(mode="json") for manifest in manifests])
    typer.echo(
        f"Wrote {len(manifests)} DefeatBeta pre-earnings-week manifests for {symbol} to {manifest_out}"
    )


@app.command()
def collect_fmp_transcripts(
    symbols: Annotated[str, typer.Option()],
    transcript_root: Annotated[Path, typer.Option()],
    start_year: Annotated[int, typer.Option()] = 2016,
    end_year: Annotated[int, typer.Option()] = datetime.now(tz=UTC).year,
    max_transcripts_per_symbol: Annotated[int | None, typer.Option()] = None,
    symbol_company_names: Annotated[str | None, typer.Option()] = None,
    overwrite: Annotated[bool, typer.Option()] = False,
    base_url: Annotated[str, typer.Option()] = "https://financialmodelingprep.com",
) -> None:
    if start_year > end_year:
        raise typer.BadParameter("start-year must be less than or equal to end-year")
    if max_transcripts_per_symbol is not None and max_transcripts_per_symbol < 1:
        raise typer.BadParameter("max-transcripts-per-symbol must be at least 1")
    settings = _load_settings()
    if settings.financial_modeling_prep_api_key is None:
        raise typer.BadParameter(
            "FINANCIAL_MODELING_PREP_API_KEY is required to collect FMP transcripts"
        )
    symbol_list = _parse_symbol_list(symbols)
    company_name_overrides = _parse_symbol_company_map(symbol_company_names)
    written = 0
    skipped_existing = 0
    skipped_missing = 0

    with httpx.Client(timeout=45, follow_redirects=True) as http_client:
        client = FinancialModelingPrepClient(
            api_key=settings.financial_modeling_prep_api_key.get_secret_value(),
            http_client=http_client,
            base_url=base_url,
        )
        for symbol in symbol_list:
            try:
                references = client.get_transcript_dates(symbol)
            except (FmpApiRateLimitError, FmpApiSubscriptionError) as exc:
                raise typer.BadParameter(str(exc)) from exc
            except FmpApiAuthError as exc:
                raise typer.BadParameter(str(exc)) from exc
            references = [
                reference
                for reference in references
                if start_year <= reference.fiscal_year <= end_year
            ]
            unique_refs = {}
            for reference in references:
                unique_refs[(reference.fiscal_year, reference.fiscal_quarter)] = reference
            ordered_refs = sorted(unique_refs.values(), key=_transcript_ref_sort_key, reverse=True)
            if max_transcripts_per_symbol is not None:
                ordered_refs = ordered_refs[:max_transcripts_per_symbol]
            company_name = company_name_overrides.get(symbol, symbol)
            company_dir = transcript_root / company_name
            company_dir.mkdir(parents=True, exist_ok=True)

            symbol_written = 0
            for reference in ordered_refs:
                transcript_path = company_dir / (
                    f"{reference.fiscal_year}_Q{reference.fiscal_quarter}_{symbol.lower()}_processed.txt"
                )
                if transcript_path.exists() and not overwrite:
                    skipped_existing += 1
                    continue
                try:
                    transcript_text, _published_at = client.get_transcript_text(
                        symbol=symbol,
                        fiscal_year=reference.fiscal_year,
                        fiscal_quarter=reference.fiscal_quarter,
                    )
                except (FmpApiAuthError, FmpApiRateLimitError, FmpApiSubscriptionError) as exc:
                    raise typer.BadParameter(str(exc)) from exc
                except (FmpApiParseError, httpx.HTTPError):
                    skipped_missing += 1
                    continue
                transcript_path.write_text(transcript_text.strip() + "\n", encoding="utf-8")
                written += 1
                symbol_written += 1
            typer.echo(
                f"{symbol}: wrote {symbol_written} transcripts "
                f"(available refs in range: {len(ordered_refs)})"
            )
    typer.echo(
        f"FMP transcript collection complete: wrote={written}, "
        f"skipped_existing={skipped_existing}, skipped_missing={skipped_missing}"
    )


@app.command()
def collect_sec_press_releases_for_transcripts(
    transcript_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    company_symbol: Annotated[str, typer.Option()],
    query: Annotated[str, typer.Option()],
    raw_dir: Annotated[Path, typer.Option()],
    manifest_out: Annotated[Path, typer.Option()],
    limit: Annotated[int, typer.Option()] = 25,
    base_url: Annotated[str, typer.Option()] = "https://api.sec-api.io",
) -> None:
    settings = _load_settings()
    if settings.sec_api_key is None:
        raise typer.BadParameter("SEC_API_KEY is required to collect SEC exhibits")

    records = [
        record
        for record in scan_transcript_corpus(transcript_root)
        if record.company_symbol == company_symbol.upper()
    ]
    records = sorted(
        records,
        key=lambda record: (record.fiscal_year, record.fiscal_quarter),
        reverse=True,
    )
    if not records:
        raise typer.BadParameter(f"No transcript records found for {company_symbol.upper()}")

    with httpx.Client(
        timeout=45,
        follow_redirects=True,
        headers={"User-Agent": "kalorie-research contact@example.com"},
    ) as http_client:
        sec_client = SecApiClient(
            api_key=settings.sec_api_key.get_secret_value(),
            http_client=http_client,
            base_url=base_url,
        )
        filings = sec_client.query_ex99_1_filings(query=query, size=limit)
        manifests = []
        for record, filing in _pair_records_to_filings(records, filings):
            for exhibit in filing.exhibits:
                manifests.append(
                    collect_public_document(
                        url=exhibit.document_url,
                        company_symbol=record.company_symbol,
                        fiscal_year=record.fiscal_year,
                        fiscal_quarter=record.fiscal_quarter,
                        source_type=_sec_exhibit_source_type(exhibit.document_type),
                        published_at=filing.filed_at,
                        fetched_at=datetime.now(tz=UTC),
                        raw_dir=raw_dir,
                        http_client=http_client,
                    )
                )

    _write_json(manifest_out, [manifest.model_dump(mode="json") for manifest in manifests])
    typer.echo(f"Wrote {len(manifests)} SEC EX-99 supplemental manifests")


@app.command()
def collect_sec_press_releases_for_corpus(
    transcript_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    company_map: Annotated[Path, typer.Option()],
    raw_dir: Annotated[Path, typer.Option()],
    manifest_out: Annotated[Path, typer.Option()],
    max_companies: Annotated[int, typer.Option()] = 10,
    filings_per_company: Annotated[int, typer.Option()] = 20,
    base_url: Annotated[str, typer.Option()] = "https://api.sec-api.io",
) -> None:
    settings = _load_settings()
    if settings.sec_api_key is None:
        raise typer.BadParameter("SEC_API_KEY is required to collect SEC exhibits")

    grouped_records = _group_transcript_records(transcript_root)
    cached_ciks = _read_company_map(company_map)
    manifests = []

    with httpx.Client(
        timeout=45,
        follow_redirects=True,
        headers={"User-Agent": "kalorie-research contact@example.com"},
    ) as http_client:
        sec_client = SecApiClient(
            api_key=settings.sec_api_key.get_secret_value(),
            http_client=http_client,
            base_url=base_url,
        )
        for company_name, records in grouped_records[:max_companies]:
            symbol = records[0].company_symbol
            cik = cached_ciks.get(symbol)
            if cik is None:
                try:
                    mapping = select_best_company_mapping(
                        sec_client.resolve_mapping(resolve_by="name", value=company_name)
                    )
                except SecApiRateLimitError as exc:
                    raise typer.BadParameter(str(exc)) from exc
                except (SecApiError, httpx.HTTPError):
                    continue
                symbol = mapping.ticker
                cik = mapping.cik
                cached_ciks[symbol] = cik

            query = (
                f'formType:"8-K" AND cik:{cik} AND '
                '(documentFormatFiles.type:"EX-99*" OR documentFormatFiles.type:"EX-99.1")'
            )
            try:
                filings = sec_client.query_ex99_1_filings(
                    query=query,
                    size=filings_per_company,
                )
            except SecApiRateLimitError as exc:
                raise typer.BadParameter(str(exc)) from exc
            except (SecApiError, httpx.HTTPError):
                continue

            for record, filing in _pair_records_to_filings(records, filings):
                for exhibit in filing.exhibits:
                    try:
                        manifests.append(
                            collect_public_document(
                                url=exhibit.document_url,
                                company_symbol=symbol,
                                fiscal_year=record.fiscal_year,
                                fiscal_quarter=record.fiscal_quarter,
                                source_type=_sec_exhibit_source_type(exhibit.document_type),
                                published_at=filing.filed_at,
                                fetched_at=datetime.now(tz=UTC),
                                raw_dir=raw_dir,
                                http_client=http_client,
                            )
                        )
                    except httpx.HTTPError:
                        continue

    if not manifests and manifest_out.exists():
        raise typer.BadParameter(
            "No SEC supplemental manifests were collected; leaving existing manifest unchanged"
        )

    _write_json(company_map, cached_ciks)
    _write_json(manifest_out, [manifest.model_dump(mode="json") for manifest in manifests])
    typer.echo(
        f"Wrote {len(manifests)} SEC EX-99 supplemental manifests "
        f"across {len({manifest.company_symbol for manifest in manifests})} companies"
    )


@app.command()
def build_historical_dataset(
    examples: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option()],
) -> None:
    training_examples = [
        HistoricalTrainingExample.model_validate(row)
        for row in json.loads(examples.read_text(encoding="utf-8"))
    ]
    _write_json(out, [example.model_dump(mode="json") for example in training_examples])
    typer.echo(f"Wrote {len(training_examples)} normalized historical examples to {out}")


@app.command()
def build_real_training_dataset(
    transcript_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    manifests: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    contracts: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option()],
    min_examples: Annotated[int, typer.Option()] = 250,
    template_catalog: Annotated[Path | None, typer.Option()] = None,
    scenario_catalogs: Annotated[Path | None, typer.Option()] = None,
    record_concurrency: Annotated[int, typer.Option(min=1, max=256)] = 1,
) -> None:
    records = scan_transcript_corpus(transcript_root)
    typer.echo(f"Loaded {len(records)} transcript records")
    manifest_rows = [
        PublicDocumentManifest.model_validate(row)
        for row in json.loads(manifests.read_text(encoding="utf-8"))
    ]
    documents_by_period = _documents_by_period_from_manifests(manifest_rows)
    mention_contracts = [
        MentionMarketContract.model_validate(row)
        for row in json.loads(contracts.read_text(encoding="utf-8"))
    ]
    if not mention_contracts:
        raise typer.BadParameter("contracts must contain at least one mention market")

    template_phrases_by_target, embedding_provider = _load_template_features_context(
        template_catalog
    )
    scenario_texts_by_event = _load_scenario_texts_by_event(scenario_catalogs)
    if scenario_texts_by_event and embedding_provider is None:
        embedding_provider = _load_embedding_provider(
            "OPENAI_API_KEY is required when scenario-catalogs is provided"
        )
    progress_hook = _dataset_progress_hook(total_records=len(records), label="real-dataset")
    progress_hook(0, 0)
    examples = build_examples_from_transcript_records(
        records=records,
        documents_by_period=documents_by_period,
        contracts=mention_contracts,
        min_examples=min_examples,
        template_phrases_by_target=template_phrases_by_target,
        scenario_texts_by_event=scenario_texts_by_event,
        embedding_provider=embedding_provider,
        progress_callback=progress_hook,
        record_concurrency=record_concurrency,
    )
    _write_json(out, [example.model_dump(mode="json") for example in examples])
    typer.echo(f"Wrote {len(examples)} real historical training examples to {out}")


@app.command()
def build_synthetic_phrase_dataset(
    transcript_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    manifests: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option()],
    target_phrases: Annotated[str, typer.Option()] = ",".join(DEFAULT_SYNTHETIC_TARGET_PHRASES),
    min_examples: Annotated[int | None, typer.Option()] = None,
    template_catalog: Annotated[Path | None, typer.Option()] = None,
    scenario_catalogs: Annotated[Path | None, typer.Option()] = None,
    market_contracts: Annotated[Path | None, typer.Option()] = None,
    include_market_phrases_globally: Annotated[bool, typer.Option()] = True,
    include_evidence_phrase_candidates: Annotated[bool, typer.Option()] = False,
    max_evidence_phrases_per_company: Annotated[int, typer.Option(min=1, max=5000)] = 500,
    record_concurrency: Annotated[int, typer.Option(min=1, max=256)] = 1,
) -> None:
    records = scan_transcript_corpus(transcript_root)
    typer.echo(f"Loaded {len(records)} transcript records")
    manifest_rows = [
        PublicDocumentManifest.model_validate(row)
        for row in json.loads(manifests.read_text(encoding="utf-8"))
    ]
    documents_by_period = _documents_by_period_from_manifests(manifest_rows)
    template_phrases_by_target, embedding_provider = _load_template_features_context(
        template_catalog
    )
    scenario_texts_by_event = _load_scenario_texts_by_event(scenario_catalogs)
    if scenario_texts_by_event and embedding_provider is None:
        embedding_provider = _load_embedding_provider(
            "OPENAI_API_KEY is required when scenario-catalogs is provided"
        )
    base_target_phrases = _parse_target_phrases(target_phrases)
    company_target_phrases = None
    if market_contracts is not None:
        contract_rows = [
            MentionMarketContract.model_validate(row)
            for row in json.loads(market_contracts.read_text(encoding="utf-8"))
        ]
        company_target_phrases = _company_target_phrases_from_contracts(contract_rows)
        if include_market_phrases_globally:
            market_target_phrases = _target_phrases_from_contracts(contract_rows)
            base_target_phrases = normalize_and_dedupe_phrases(
                [*base_target_phrases, *market_target_phrases]
            )
            typer.echo(
                f"Expanded global target phrase set with {len(market_target_phrases)} market phrases"
            )
        typer.echo(
            f"Loaded company-specific market targets for {len(company_target_phrases)} companies"
        )
    if include_evidence_phrase_candidates:
        evidence_target_phrases = _company_target_phrases_from_manifests(
            manifest_rows,
            seed_phrases=base_target_phrases,
            max_candidates=max_evidence_phrases_per_company,
        )
        company_target_phrases = _merge_company_target_phrases(
            company_target_phrases or {},
            evidence_target_phrases,
        )
        total_evidence_targets = sum(len(values) for values in evidence_target_phrases.values())
        typer.echo(
            f"Added {total_evidence_targets} evidence-derived phrase candidates "
            f"across {len(evidence_target_phrases)} companies"
        )
    progress_hook = _dataset_progress_hook(total_records=len(records), label="synthetic-dataset")
    progress_hook(0, 0)
    examples = build_synthetic_phrase_examples_from_transcript_records(
        records=records,
        documents_by_period=documents_by_period,
        target_phrases=base_target_phrases,
        min_examples=min_examples,
        template_phrases_by_target=template_phrases_by_target,
        scenario_texts_by_event=scenario_texts_by_event,
        embedding_provider=embedding_provider,
        progress_callback=progress_hook,
        record_concurrency=record_concurrency,
        company_target_phrases=company_target_phrases,
    )
    _write_json(out, [example.model_dump(mode="json") for example in examples])
    typer.echo(f"Wrote {len(examples)} synthetic phrase-presence examples to {out}")


@app.command("plan-historical-evidence-collection")
def plan_historical_evidence_collection(
    transcript_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    out_dir: Annotated[Path, typer.Option()],
    cached_ciks: Annotated[Path, typer.Option()] = Path("data/sec/company_to_cik.json"),
    sec_request_budget: Annotated[int, typer.Option(min=0)] = 80,
) -> None:
    inventory = build_transcript_inventory(transcript_root)
    cik_cache = _load_json_object(cached_ciks) if cached_ciks.exists() else {}
    plan = plan_sec_requests(
        inventory,
        cached_ciks={symbol.upper(): str(cik) for symbol, cik in cik_cache.items()},
        request_budget=sec_request_budget,
    )
    _write_json(out_dir / "transcript-inventory.json", inventory.model_dump(mode="json"))
    _write_json(out_dir / "sec-request-plan.json", plan.model_dump(mode="json"))
    typer.echo(
        f"Planned {plan.projected_requests} SEC API requests across "
        f"{len(plan.companies)} companies (budget={sec_request_budget})"
    )


@app.command("collect-historical-evidence-pack")
def collect_historical_evidence_pack(
    transcript_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    out_dir: Annotated[Path, typer.Option()],
    cached_ciks: Annotated[Path, typer.Option()] = Path("data/sec/company_to_cik.json"),
    sec_request_budget: Annotated[int, typer.Option(min=0)] = 80,
    news_manifests: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    max_document_workers: Annotated[int, typer.Option(min=1, max=16)] = 4,
) -> None:
    settings = _load_settings()
    if settings.sec_api_key is None:
        raise typer.BadParameter("SEC_API_KEY is required to collect SEC evidence")
    inventory = build_transcript_inventory(transcript_root)
    cik_cache = _load_json_object(cached_ciks) if cached_ciks.exists() else {}
    normalized_cik_cache = {symbol.upper(): str(cik) for symbol, cik in cik_cache.items()}

    def checkpoint(result, cache) -> None:
        _write_json(cached_ciks, cache)
        _write_json(out_dir / "evidence-summary.json", result.summary.model_dump(mode="json"))
        _write_json(
            out_dir / "evidence-manifests.json",
            [row.model_dump(mode="json") for row in result.manifests],
        )

    with httpx.Client(timeout=30.0) as http_client:
        result = collect_sec_ex99_manifests(
            inventory,
            sec_client=SecApiClient(
                api_key=settings.sec_api_key.get_secret_value(),
                http_client=http_client,
            ),
            cached_ciks=normalized_cik_cache,
            request_budget=sec_request_budget,
            raw_dir=out_dir / "sec",
            http_client=http_client,
            checkpoint_writer=checkpoint,
            max_document_workers=max_document_workers,
        )
    manifests = result.manifests
    if news_manifests is not None:
        manifests = merge_document_manifests(
            [
                *manifests,
                *[
                    PublicDocumentManifest.model_validate(row)
                    for row in json.loads(news_manifests.read_text(encoding="utf-8"))
                ],
            ]
        )
    _write_json(cached_ciks, normalized_cik_cache)
    _write_json(out_dir / "evidence-summary.json", result.summary.model_dump(mode="json"))
    _write_json(
        out_dir / "evidence-manifests.json",
        [row.model_dump(mode="json") for row in manifests],
    )
    typer.echo(f"Wrote {len(manifests)} historical evidence manifests to {out_dir}")


@app.command("collect-historical-news-manifests")
def collect_historical_news_manifests_command(
    examples: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    raw_dir: Annotated[Path, typer.Option()],
    manifest_out: Annotated[Path, typer.Option()],
    summary_out: Annotated[Path | None, typer.Option()] = None,
    company_names: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    days_before_cutoff: Annotated[int, typer.Option(min=1)] = 60,
    max_articles_per_window: Annotated[int, typer.Option(min=1)] = 80,
    max_total_articles: Annotated[int, typer.Option(min=1)] = 5000,
    use_yfinance_fallback: Annotated[bool, typer.Option()] = True,
    dataset_url: Annotated[str | None, typer.Option()] = None,
) -> None:
    example_rows = [
        HistoricalTrainingExample.model_validate(row)
        for row in json.loads(examples.read_text(encoding="utf-8"))
    ]
    name_map = _read_company_map(company_names) if company_names is not None else {}
    result = collect_historical_news_manifests_for_examples(
        example_rows,
        defeatbeta_client=DefeatBetaNewsClient(dataset_url=dataset_url),
        raw_dir=raw_dir,
        company_names=name_map,
        yfinance_collector=_collect_yfinance_news if use_yfinance_fallback else None,
        days_before_cutoff=days_before_cutoff,
        max_articles_per_window=max_articles_per_window,
        max_total_articles=max_total_articles,
    )
    _write_json(
        manifest_out,
        [
            PublicDocumentManifest.model_validate(row).model_dump(mode="json")
            for row in result.manifests
        ],
    )
    if summary_out is not None:
        _write_json(summary_out, result.summary.model_dump(mode="json"))
    typer.echo(
        f"Wrote {len(result.manifests)} historical news manifests to {manifest_out}; "
        f"skipped={len(result.summary.skipped_records)}"
    )


@app.command("generate-transcript-kalshi-phrases")
def generate_transcript_kalshi_phrases(
    transcript_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    out: Annotated[Path, typer.Option()],
    max_transcripts: Annotated[int | None, typer.Option(min=1)] = None,
    max_per_label: Annotated[int, typer.Option(min=1, max=12)] = 12,
    use_openai: Annotated[bool, typer.Option()] = True,
    llm_model: Annotated[str, typer.Option()] = "gpt-4o-mini",
    max_workers: Annotated[int, typer.Option(min=1, max=32)] = 4,
) -> None:
    settings = _load_settings()
    openai_client = None
    if use_openai and settings.openai_api_key is not None:
        from openai import OpenAI

        openai_client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
    inventory = build_transcript_inventory(transcript_root)

    def openai_response_provider(row, transcript_text: str) -> str | None:
        if openai_client is None:
            return None
        return generate_openai_phrase_response(
            client=openai_client,
            model=llm_model,
            company_name=row.company_name,
            transcript_text=transcript_text,
            max_per_label=max_per_label,
        ).model_dump_json()

    selected_rows = inventory.rows[:max_transcripts]
    catalog = build_phrase_catalog(
        rows=selected_rows,
        openai_response_provider=openai_response_provider if openai_client is not None else None,
        max_per_label=max_per_label,
        max_workers=max_workers,
        checkpoint_writer=lambda catalog: _write_json(out, catalog.model_dump(mode="json")),
    )
    _write_json(out, catalog.model_dump(mode="json"))
    typer.echo(f"Wrote {len(catalog.entries)} validated Kalshi-style phrase rows to {out}")


@app.command("build-historical-synthetic-kalshi-rows")
def build_historical_synthetic_kalshi_rows(
    transcript_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    manifests: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    phrase_catalog: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option()],
    skipped_out: Annotated[Path | None, typer.Option()] = None,
    event_dossiers: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    record_concurrency: Annotated[int, typer.Option(min=1, max=64)] = 1,
) -> None:
    inventory = build_transcript_inventory(transcript_root)
    manifest_rows = [
        PublicDocumentManifest.model_validate(row)
        for row in json.loads(manifests.read_text(encoding="utf-8"))
    ]
    catalog = PhraseCatalog.model_validate(json.loads(phrase_catalog.read_text(encoding="utf-8")))
    dossier_catalogs = (
        load_event_dossier_catalogs(event_dossiers)
        if event_dossiers is not None
        else None
    )
    embedding_provider = (
        _load_embedding_provider(
            "OPENAI_API_KEY is required when event-dossiers is provided"
        )
        if dossier_catalogs
        else None
    )
    result = build_historical_synthetic_rows(
        inventory=inventory,
        manifests=manifest_rows,
        phrase_catalog=catalog,
        event_dossiers=dossier_catalogs,
        embedding_provider=embedding_provider,
        record_concurrency=record_concurrency,
    )
    _write_json(out, [example.model_dump(mode="json") for example in result.examples])
    if skipped_out is not None:
        _write_json(skipped_out, [row.model_dump(mode="json") for row in result.skipped_records])
    typer.echo(f"Wrote {len(result.examples)} historical synthetic Kalshi-style rows to {out}")


@app.command("generate-historical-event-dossiers")
def generate_historical_event_dossiers(
    transcript_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    manifests: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    phrase_catalog: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    cache_dir: Annotated[Path, typer.Option()],
    out: Annotated[Path, typer.Option()],
    max_documents: Annotated[int, typer.Option(min=1, max=100)] = 20,
    max_chars_per_document: Annotated[int, typer.Option(min=500, max=20000)] = 3000,
    max_items: Annotated[int, typer.Option(min=1, max=25)] = 20,
    llm_model: Annotated[str, typer.Option()] = "gpt-4o-mini",
    max_workers: Annotated[int, typer.Option(min=1, max=32)] = 4,
) -> None:
    settings = _load_settings()
    if settings.openai_api_key is None:
        raise typer.BadParameter("OPENAI_API_KEY is required to generate event dossiers")
    inventory = build_transcript_inventory(transcript_root)
    manifest_rows = [
        PublicDocumentManifest.model_validate(row)
        for row in json.loads(manifests.read_text(encoding="utf-8"))
    ]
    catalog = PhraseCatalog.model_validate(json.loads(phrase_catalog.read_text(encoding="utf-8")))
    generator = OpenAIEventScenarioGenerator(
        api_key=settings.openai_api_key.get_secret_value(),
        model=llm_model,
    )
    dossiers = generate_event_dossiers(
        inventory_rows=inventory.rows,
        manifests=manifest_rows,
        phrase_catalog=catalog,
        generator=generator,
        cache_dir=cache_dir,
        max_documents=max_documents,
        max_chars_per_document=max_chars_per_document,
        max_items=max_items,
        max_workers=max_workers,
        progress_callback=lambda done, total, event_id, reused: typer.echo(
            f"[{done}/{total}] {'reused' if reused else 'generated'} {event_id}"
        ),
    )
    _write_json(out, [catalog.model_dump(mode="json") for catalog in dossiers])
    typer.echo(f"Wrote {len(dossiers)} cached event dossiers to {out}")


@app.command("collect-kalshi-mention-event-pack")
def collect_kalshi_mention_event_pack(
    candidates: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    out_dir: Annotated[Path, typer.Option()],
) -> None:
    candidate_rows = [
        KalshiEventPackCandidate.model_validate(row)
        for row in json.loads(candidates.read_text(encoding="utf-8"))
    ]
    with httpx.Client(timeout=30.0) as http_client:
        summary = build_event_pack(
            candidates=candidate_rows,
            kalshi_client=KalshiPublicClient(http_client=http_client),
            output_dir=out_dir,
        )
    typer.echo(
        f"Collected {summary['inspected_count']} Kalshi event packs; "
        f"{summary['ready_count']} ready"
    )


@app.command("build-kalshi-event-pack-training-rows")
def build_kalshi_event_pack_training_rows(
    pack_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    out: Annotated[Path, typer.Option()],
    skipped_out: Annotated[Path | None, typer.Option()] = None,
    event_dossiers: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
) -> None:
    dossier_catalogs = (
        load_event_dossier_catalogs(event_dossiers)
        if event_dossiers is not None
        else None
    )
    embedding_provider = (
        _load_embedding_provider(
            "OPENAI_API_KEY is required when event-dossiers is provided"
        )
        if dossier_catalogs
        else None
    )
    result = build_real_event_pack_training_rows(
        pack_dir,
        event_dossiers=dossier_catalogs,
        embedding_provider=embedding_provider,
    )
    _write_json(out, [example.model_dump(mode="json") for example in result.examples])
    if skipped_out is not None:
        _write_json(skipped_out, [row.model_dump(mode="json") for row in result.skipped_records])
    typer.echo(f"Wrote {len(result.examples)} real Kalshi event-pack rows to {out}")


@app.command("verify-kalshi-workflow-artifacts")
def verify_kalshi_workflow_artifacts(
    event_pack_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    out: Annotated[Path | None, typer.Option()] = None,
) -> None:
    report = verify_event_pack_artifacts(event_pack_dir)
    if out is not None:
        _write_json(out, report.model_dump(mode="json"))
    if not report.ok:
        raise typer.BadParameter("; ".join(report.errors))
    typer.echo(f"Verified Kalshi event-pack artifacts in {event_pack_dir}")


@app.command("generate-template-phrases")
def generate_template_phrases(
    manifests: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option()],
    target_phrases: Annotated[str, typer.Option()] = ",".join(DEFAULT_SYNTHETIC_TARGET_PHRASES),
    max_documents: Annotated[int, typer.Option()] = 20,
    max_chars_per_document: Annotated[int, typer.Option()] = 3000,
    max_variants: Annotated[int, typer.Option()] = 12,
    llm_model: Annotated[str, typer.Option()] = "gpt-4o-mini",
    company_symbol: Annotated[str | None, typer.Option()] = None,
    max_concurrency: Annotated[int, typer.Option(min=1, max=128)] = 60,
) -> None:
    settings = _load_settings()
    if settings.openai_api_key is None:
        raise typer.BadParameter("OPENAI_API_KEY is required to generate template phrases")
    manifest_rows = [
        PublicDocumentManifest.model_validate(row)
        for row in json.loads(manifests.read_text(encoding="utf-8"))
    ]
    snippets = load_material_snippets(
        manifest_rows,
        max_documents=max_documents,
        max_chars_per_document=max_chars_per_document,
        company_symbol=company_symbol,
    )
    generator = OpenAITemplatePhraseGenerator(
        api_key=settings.openai_api_key.get_secret_value(),
        model=llm_model,
    )
    phrase_list = _parse_target_phrases(target_phrases)
    typer.echo(
        f"Generating template variants for {len(phrase_list)} targets "
        f"from {len(snippets)} material snippets with concurrency={max_concurrency}..."
    )
    phrase_variants = {}
    started = monotonic()
    phrase_meta: dict[str, tuple[int, float]] = {}
    for index, phrase in enumerate(phrase_list, start=1):
        phrase_meta[phrase] = (index, monotonic())
        typer.echo(f"[{index}/{len(phrase_list)}] queued '{phrase}'")
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        future_to_phrase = {
            executor.submit(
                generator.generate,
                target_phrase=phrase,
                material_snippets=snippets,
                max_variants=max_variants,
            ): phrase
            for phrase in phrase_list
        }
        for future in as_completed(future_to_phrase):
            phrase = future_to_phrase[future]
            index, phrase_started = phrase_meta[phrase]
            variants = future.result()
            phrase_variants[phrase] = variants
            typer.echo(
                f"[{index}/{len(phrase_list)}] done '{phrase}' -> {len(variants)} variants "
                f"in {monotonic() - phrase_started:.1f}s"
            )
    catalog = TemplatePhraseCatalog(llm_model=generator.model, phrase_variants=phrase_variants)
    _write_json(out, catalog.model_dump(mode="json"))
    typer.echo(
        f"Wrote template phrase catalog for {len(phrase_variants)} targets to {out} "
        f"(elapsed {monotonic() - started:.1f}s)"
    )


@app.command("generate-event-scenario-catalog")
def generate_event_scenario_catalog(
    manifests: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option()],
    event_id: Annotated[str, typer.Option()],
    company_symbol: Annotated[str, typer.Option()],
    company_name: Annotated[str, typer.Option()],
    target_phrases: Annotated[str, typer.Option()] = ",".join(DEFAULT_SYNTHETIC_TARGET_PHRASES),
    max_documents: Annotated[int, typer.Option()] = 20,
    max_chars_per_document: Annotated[int, typer.Option()] = 3000,
    max_items: Annotated[int, typer.Option(min=1, max=25)] = 8,
    llm_model: Annotated[str, typer.Option()] = "gpt-4o-mini",
) -> None:
    settings = _load_settings()
    if settings.openai_api_key is None:
        raise typer.BadParameter("OPENAI_API_KEY is required to generate event scenarios")
    manifest_rows = [
        PublicDocumentManifest.model_validate(row)
        for row in json.loads(manifests.read_text(encoding="utf-8"))
    ]
    snippets = load_material_snippets(
        manifest_rows,
        max_documents=max_documents,
        max_chars_per_document=max_chars_per_document,
        company_symbol=company_symbol,
    )
    if not snippets:
        raise typer.BadParameter("No material snippets found for scenario generation")
    generator = OpenAIEventScenarioGenerator(
        api_key=settings.openai_api_key.get_secret_value(),
        model=llm_model,
    )
    catalog = generator.generate(
        event_id=event_id,
        company_symbol=company_symbol,
        company_name=company_name,
        target_phrases=_parse_target_phrases(target_phrases),
        material_snippets=snippets,
        max_items=max_items,
    )
    _write_json(out, catalog.model_dump(mode="json"))
    typer.echo(f"Wrote event scenario catalog for {catalog.event_id} to {out}")


@app.command()
def embed_dataset(
    texts: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option()],
) -> None:
    settings = _load_settings()
    if settings.openai_api_key is None:
        raise typer.BadParameter("OPENAI_API_KEY is required to embed a dataset")
    payload = json.loads(texts.read_text(encoding="utf-8"))
    text_values = _extract_text_values(payload)
    provider = CachedEmbeddingProvider(
        provider=OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.openai_embedding_model,
            dimensions=settings.openai_embedding_dimensions,
            batch_size=settings.embedding_batch_size,
        ),
        cache_path=settings.embedding_cache_path,
    )
    embeddings = provider.embed(text_values)
    _write_json(
        out,
        [
            {"text": text, "embedding": embedding}
            for text, embedding in zip(text_values, embeddings, strict=True)
        ],
    )
    typer.echo(f"Wrote {len(embeddings)} embeddings to {out}")


@app.command()
def train_model(
    examples: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option()],
    test_fraction: Annotated[float, typer.Option()] = 0.25,
) -> None:
    training_examples = [
        HistoricalTrainingExample.model_validate(row)
        for row in json.loads(examples.read_text(encoding="utf-8"))
    ]
    report = train_and_evaluate(training_examples, test_fraction=test_fraction)
    _write_json(out, report.model_dump(mode="json"))
    typer.echo(f"Global Brier score: {report.global_brier_score:.6f}")
    typer.echo(f"Global log loss: {report.global_log_loss:.6f}")
    typer.echo(f"Company-adapted Brier score: {report.company_adapted_brier_score:.6f}")
    typer.echo(f"Company-adapted log loss: {report.company_adapted_log_loss:.6f}")
    typer.echo(f"Company adaptation improved Brier: {report.company_adaptation_improved_brier}")


@app.command()
def evaluate_model(
    examples: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option()],
    test_fraction: Annotated[float, typer.Option()] = 0.25,
) -> None:
    train_model(examples=examples, out=out, test_fraction=test_fraction)


@app.command("train-model1")
def train_model1_command(
    examples: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option()],
    min_company_rows: Annotated[int, typer.Option()] = 25,
    blend_weight: Annotated[float, typer.Option()] = 0.35,
    regularization_c: Annotated[float, typer.Option()] = 1.0,
    class_weight_balanced: Annotated[bool, typer.Option()] = False,
    include_target_indicator: Annotated[bool, typer.Option()] = False,
    enable_temperature_calibration: Annotated[bool, typer.Option()] = False,
    enable_isotonic_calibration: Annotated[bool, typer.Option()] = False,
    calibration_fraction: Annotated[float, typer.Option()] = 0.2,
    enforce_pretrain_gate: Annotated[bool, typer.Option()] = False,
    pretrain_gate_report_out: Annotated[Path | None, typer.Option()] = None,
) -> None:
    training_examples = [
        HistoricalTrainingExample.model_validate(row)
        for row in json.loads(examples.read_text(encoding="utf-8"))
    ]
    gate_report = _pretrain_gate_report(
        training_examples,
        min_rows=24,
        min_events=6,
        scope="global",
    )
    _emit_pretrain_gate_report(gate_report)
    if pretrain_gate_report_out is not None:
        _write_json(pretrain_gate_report_out, gate_report)
    if enforce_pretrain_gate and not gate_report["passed"]:
        raise typer.BadParameter(
            "pre-train gate failed: "
            + "; ".join(gate_report["errors"])
        )
    artifact = train_model1(
        training_examples,
        min_company_rows=min_company_rows,
        blend_weight=blend_weight,
        regularization_c=regularization_c,
        class_weight_balanced=class_weight_balanced,
        include_target_indicator=include_target_indicator,
        enable_temperature_calibration=enable_temperature_calibration,
        enable_isotonic_calibration=enable_isotonic_calibration,
        calibration_fraction=calibration_fraction,
    )
    _write_json(out, artifact.model_dump(mode="json"))
    typer.echo(
        f"Wrote model1 artifact with {len(artifact.feature_columns)} features and "
        f"{len(artifact.company_overrides)} company overrides"
    )


@app.command("train-model1-optimized")
def train_model1_optimized_command(
    examples: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option()],
    report_out: Annotated[Path, typer.Option()],
    test_fraction: Annotated[float, typer.Option()] = 0.25,
    regularization_grid: Annotated[str, typer.Option()] = "0.05,0.1,0.3,1,3,10",
    min_company_rows_grid: Annotated[str, typer.Option()] = "8,12,20,25,35",
    blend_weight_grid: Annotated[str, typer.Option()] = "0.15,0.25,0.35,0.5,0.7",
    class_weight_balanced_values: Annotated[str, typer.Option()] = "false,true",
    include_target_indicator_values: Annotated[str, typer.Option()] = "false,true",
    enable_temperature_calibration: Annotated[bool, typer.Option()] = False,
    enable_isotonic_calibration: Annotated[bool, typer.Option()] = True,
    calibration_fraction: Annotated[float, typer.Option()] = 0.2,
    enforce_pretrain_gate: Annotated[bool, typer.Option()] = False,
    pretrain_gate_report_out: Annotated[Path | None, typer.Option()] = None,
) -> None:
    training_examples = [
        HistoricalTrainingExample.model_validate(row)
        for row in json.loads(examples.read_text(encoding="utf-8"))
    ]
    gate_report = _pretrain_gate_report(
        training_examples,
        min_rows=24,
        min_events=6,
        scope="global_optimized",
    )
    _emit_pretrain_gate_report(gate_report)
    if pretrain_gate_report_out is not None:
        _write_json(pretrain_gate_report_out, gate_report)
    if enforce_pretrain_gate and not gate_report["passed"]:
        raise typer.BadParameter(
            "pre-train gate failed: "
            + "; ".join(gate_report["errors"])
        )
    artifact, report = optimize_model1_for_brier(
        training_examples,
        test_fraction=test_fraction,
        regularization_values=_parse_float_values(regularization_grid),
        min_company_rows_values=[
            int(value) for value in _parse_float_values(min_company_rows_grid)
        ],
        blend_weight_values=_parse_float_values(blend_weight_grid),
        class_weight_balanced_values=_parse_boolean_values(class_weight_balanced_values),
        include_target_indicator_values=_parse_boolean_values(include_target_indicator_values),
        enable_temperature_calibration=enable_temperature_calibration,
        enable_isotonic_calibration=enable_isotonic_calibration,
        calibration_fraction=calibration_fraction,
    )
    _write_json(out, artifact.model_dump(mode="json"))
    _write_json(report_out, report.model_dump(mode="json"))
    best = report.best_trial
    typer.echo(f"Optimized model1 holdout Brier score: {best.holdout_brier_score:.6f}")
    typer.echo(f"Optimized model1 holdout log loss: {best.holdout_log_loss:.6f}")
    typer.echo(
        "Best params -> "
        f"C={best.regularization_c}, min_company_rows={best.min_company_rows}, "
        f"blend_weight={best.blend_weight}, class_weight_balanced={best.class_weight_balanced}, "
        f"target_indicator={best.include_target_indicator}, "
        f"temperature={best.temperature_calibration}, "
        f"isotonic={best.isotonic_calibration}"
    )


@app.command("run-base-ablation-harness")
def run_base_ablation_harness(
    transcript_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    sec_manifests: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    out_dir: Annotated[Path, typer.Option()],
    target_phrases: Annotated[str, typer.Option()] = ",".join(DEFAULT_SYNTHETIC_TARGET_PHRASES),
    market_contracts: Annotated[Path | None, typer.Option()] = None,
    news_manifests: Annotated[Path | None, typer.Option()] = None,
    template_catalog: Annotated[Path | None, typer.Option()] = None,
    include_market_phrases_globally: Annotated[bool, typer.Option()] = True,
    record_concurrency: Annotated[int, typer.Option(min=1, max=256)] = 1,
    test_fraction: Annotated[float, typer.Option()] = 0.25,
    regularization_grid: Annotated[str, typer.Option()] = "0.05,0.1,0.3,1,3,10",
    min_company_rows_grid: Annotated[str, typer.Option()] = "8,12,20,25,35",
    blend_weight_grid: Annotated[str, typer.Option()] = "0.15,0.25,0.35,0.5,0.7",
    class_weight_balanced_values: Annotated[str, typer.Option()] = "false,true",
    include_target_indicator_values: Annotated[str, typer.Option()] = "false,true",
    enable_isotonic_calibration: Annotated[bool, typer.Option()] = True,
    calibration_fraction: Annotated[float, typer.Option()] = 0.2,
) -> None:
    records = scan_transcript_corpus(transcript_root)
    typer.echo(f"Loaded {len(records)} transcript records")
    sec_manifest_rows = [
        PublicDocumentManifest.model_validate(row)
        for row in json.loads(sec_manifests.read_text(encoding="utf-8"))
    ]
    news_manifest_rows = (
        [
            PublicDocumentManifest.model_validate(row)
            for row in json.loads(news_manifests.read_text(encoding="utf-8"))
        ]
        if news_manifests is not None
        else []
    )
    contract_rows: list[MentionMarketContract] = []
    if market_contracts is not None:
        contract_rows = [
            MentionMarketContract.model_validate(row)
            for row in json.loads(market_contracts.read_text(encoding="utf-8"))
        ]
    base_target_phrases = _parse_target_phrases(target_phrases)
    company_target_phrases = _company_target_phrases_from_contracts(contract_rows)
    expanded_target_phrases = base_target_phrases
    if include_market_phrases_globally and contract_rows:
        market_target_phrases = _target_phrases_from_contracts(contract_rows)
        expanded_target_phrases = normalize_and_dedupe_phrases(
            [*base_target_phrases, *market_target_phrases]
        )
        typer.echo(
            f"Expanded global target phrase set with {len(market_target_phrases)} market phrases"
        )

    template_phrases_by_target, embedding_provider = _load_template_features_context(
        template_catalog
    )
    regularization_values = _parse_float_values(regularization_grid)
    min_company_rows_values = [int(value) for value in _parse_float_values(min_company_rows_grid)]
    blend_weight_values = _parse_float_values(blend_weight_grid)
    class_weight_values = _parse_boolean_values(class_weight_balanced_values)
    target_indicator_values = _parse_boolean_values(include_target_indicator_values)

    variants: list[dict] = [
        {
            "name": "sec_only",
            "manifest_rows": sec_manifest_rows,
            "target_phrases": base_target_phrases,
            "company_target_phrases": None,
        }
    ]
    if contract_rows:
        variants.append(
            {
                "name": "sec_plus_market_phrases",
                "manifest_rows": sec_manifest_rows,
                "target_phrases": expanded_target_phrases,
                "company_target_phrases": company_target_phrases,
            }
        )
    if news_manifest_rows:
        sec_plus_news_rows = [*sec_manifest_rows, *news_manifest_rows]
        variants.append(
            {
                "name": "sec_plus_news",
                "manifest_rows": sec_plus_news_rows,
                "target_phrases": base_target_phrases,
                "company_target_phrases": None,
            }
        )
        if contract_rows:
            variants.append(
                {
                    "name": "sec_plus_news_plus_market_phrases",
                    "manifest_rows": sec_plus_news_rows,
                    "target_phrases": expanded_target_phrases,
                    "company_target_phrases": company_target_phrases,
                }
            )

    summary_rows: list[dict] = []
    for variant in variants:
        variant_name = str(variant["name"])
        typer.echo(f"[ablation] building variant '{variant_name}'")
        documents_by_period = _documents_by_period_from_manifests(variant["manifest_rows"])
        progress_hook = _dataset_progress_hook(total_records=len(records), label=f"ablation-{variant_name}")
        progress_hook(0, 0)
        examples = build_synthetic_phrase_examples_from_transcript_records(
            records=records,
            documents_by_period=documents_by_period,
            target_phrases=variant["target_phrases"],
            min_examples=None,
            template_phrases_by_target=template_phrases_by_target,
            embedding_provider=embedding_provider,
            progress_callback=progress_hook,
            record_concurrency=record_concurrency,
            company_target_phrases=variant["company_target_phrases"],
        )
        dataset_path = out_dir / "datasets" / f"{variant_name}.json"
        _write_json(dataset_path, [example.model_dump(mode="json") for example in examples])
        summary = {
            "variant": variant_name,
            "dataset_path": str(dataset_path),
            "sample_count": len(examples),
            "company_count": len({example.company_symbol for example in examples}),
            "target_phrase_count": len({example.target_phrase for example in examples}),
            "holdout_brier": None,
            "holdout_log_loss": None,
            "in_sample_brier": None,
            "status": "skipped",
        }
        if len(examples) < 12:
            summary["skip_reason"] = "at least 12 examples are required for optimization"
            summary_rows.append(summary)
            typer.echo(f"[ablation] skipped '{variant_name}' because only {len(examples)} rows were built")
            continue
        artifact, report = optimize_model1_for_brier(
            examples,
            test_fraction=test_fraction,
            regularization_values=regularization_values,
            min_company_rows_values=min_company_rows_values,
            blend_weight_values=blend_weight_values,
            class_weight_balanced_values=class_weight_values,
            include_target_indicator_values=target_indicator_values,
            enable_isotonic_calibration=enable_isotonic_calibration,
            calibration_fraction=calibration_fraction,
        )
        model_path = out_dir / "models" / f"model1-{variant_name}.json"
        report_path = out_dir / "eval" / f"model1-{variant_name}-optimization-report.json"
        _write_json(model_path, artifact.model_dump(mode="json"))
        _write_json(report_path, report.model_dump(mode="json"))
        labels: list[int] = []
        probabilities: list[float] = []
        for row in examples:
            prediction = predict_model1(
                artifact,
                company_symbol=row.company_symbol,
                feature_vector=FeatureVector(target_phrase=row.target_phrase, features=row.features),
            )
            labels.append(row.label)
            probabilities.append(prediction.probability)
        summary.update(
            {
                "status": "completed",
                "model_path": str(model_path),
                "report_path": str(report_path),
                "holdout_brier": report.best_trial.holdout_brier_score,
                "holdout_log_loss": report.best_trial.holdout_log_loss,
                "in_sample_brier": _brier_score(probabilities, labels),
                "best_trial": report.best_trial.model_dump(mode="json"),
            }
        )
        summary_rows.append(summary)
        typer.echo(
            f"[ablation] '{variant_name}' holdout Brier={report.best_trial.holdout_brier_score:.6f} "
            f"rows={len(examples)}"
        )

    sorted_summary_rows = sorted(
        summary_rows,
        key=lambda row: float(row["holdout_brier"]) if row["holdout_brier"] is not None else float("inf"),
    )
    summary_path = out_dir / "eval" / "base-ablation-summary.json"
    _write_json(
        summary_path,
        {
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "variant_count": len(sorted_summary_rows),
            "variants": sorted_summary_rows,
        },
    )
    typer.echo(f"Wrote base ablation summary to {summary_path}")


@app.command("evaluate-model1")
def evaluate_model1_command(
    model: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    examples: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option()],
) -> None:
    artifact = MentionModelArtifact.model_validate(json.loads(model.read_text(encoding="utf-8")))
    example_rows = [
        HistoricalTrainingExample.model_validate(row)
        for row in json.loads(examples.read_text(encoding="utf-8"))
    ]
    labels: list[int] = []
    probabilities: list[float] = []
    grouped_labels: dict[str, list[int]] = {}
    grouped_probabilities: dict[str, list[float]] = {}
    for row in example_rows:
        prediction = predict_model1(
            artifact,
            company_symbol=row.company_symbol,
            feature_vector=FeatureVector(target_phrase=row.target_phrase, features=row.features),
        )
        labels.append(row.label)
        probabilities.append(prediction.probability)
        grouped_labels.setdefault(row.company_symbol, []).append(row.label)
        grouped_probabilities.setdefault(row.company_symbol, []).append(prediction.probability)
    overall_brier = _brier_score(probabilities, labels)
    report = {
        "sample_count": len(example_rows),
        "model_version": artifact.model_version,
        "brier_score": overall_brier,
        "expected_calibration_error": _ece_score(probabilities, labels),
        "log_loss": _safe_log_loss(labels, probabilities),
        "company_brier_scores": {
            symbol: _brier_score(grouped_probabilities[symbol], grouped_labels[symbol])
            for symbol in sorted(grouped_labels)
        },
    }
    _write_json(out, report)
    typer.echo(f"Model1 Brier score: {overall_brier:.6f}")
    typer.echo(f"Model1 ECE (10-bin): {report['expected_calibration_error']:.6f}")
    typer.echo(f"Model1 log loss: {report['log_loss']:.6f}")


@app.command("build-benchmark-pack")
def build_benchmark_pack_command(
    event_config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    contracts: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    examples: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    evidence_manifests: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    snapshots: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option()],
    split: Annotated[str, typer.Option()] = "validation",
    allow_missing_snapshots: Annotated[bool, typer.Option()] = False,
) -> None:
    events = _load_model_rows(event_config, BenchmarkEvent)
    markets = _load_model_rows(contracts, BenchmarkMarket)
    evidence = _load_model_rows(evidence_manifests, BenchmarkEvidenceDocument)
    example_rows = _load_model_rows(examples, HistoricalTrainingExample)
    snapshot_rows = _load_model_rows(snapshots, BenchmarkSnapshot)
    pack = BenchmarkPack(
        manifest=BenchmarkPackManifest(
            pack_id=out.name,
            split=split,  # type: ignore[arg-type]
            created_at=datetime.now(tz=UTC),
            description=f"Benchmark pack built from {event_config}",
            source_paths=[
                str(event_config),
                str(contracts),
                str(examples),
                str(evidence_manifests),
                str(snapshots),
            ],
        ),
        events=events,
        markets=markets,
        snapshots=snapshot_rows,
        evidence=evidence,
        examples=example_rows,
    )
    try:
        validate_benchmark_pack(pack)
    except ValueError as exc:
        if not allow_missing_snapshots or "missing snapshots" not in str(exc):
            raise typer.BadParameter(str(exc)) from exc

    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "manifest.json", pack.manifest.model_dump(mode="json"))
    _write_json(out / "events.json", [event.model_dump(mode="json") for event in pack.events])
    _write_json(out / "markets.json", [market.model_dump(mode="json") for market in pack.markets])
    _write_json(
        out / "snapshots.json",
        [snapshot.model_dump(mode="json") for snapshot in pack.snapshots],
    )
    _write_json(
        out / "evidence.json",
        [document.model_dump(mode="json") for document in pack.evidence],
    )
    _write_json(out / "examples.json", [row.model_dump(mode="json") for row in pack.examples])
    _write_json(out / "pack.json", pack.model_dump(mode="json"))
    typer.echo(
        f"Wrote benchmark pack {pack.manifest.pack_id} with "
        f"{len(pack.examples)} examples to {out}"
    )


@app.command("run-benchmark-pack")
def run_benchmark_pack_command(
    pack: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    model: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    model_family: Annotated[str, typer.Option()],
    out: Annotated[Path, typer.Option()],
    table_out: Annotated[Path | None, typer.Option()] = None,
    log_out: Annotated[Path | None, typer.Option()] = None,
    diagnostics_out: Annotated[Path | None, typer.Option()] = None,
) -> None:
    benchmark_pack = BenchmarkPack.model_validate(json.loads(pack.read_text(encoding="utf-8")))
    artifact = MentionModelArtifact.model_validate(json.loads(model.read_text(encoding="utf-8")))
    metadata = BenchmarkRunMetadata(
        model_family=model_family,  # type: ignore[arg-type]
        model_path=str(model),
        pack_path=str(pack),
        calibration=_model1_calibration_summary(artifact),
    )
    report = run_model1_pack_benchmark(benchmark_pack, artifact, metadata)
    report["diagnostics"] = _kalshi_benchmark_diagnostics_report(report["rows"])
    _write_json(out, report)
    table_path = table_out or _benchmark_table_path(out)
    log_path = log_out or _benchmark_log_path(out)
    diagnostics_path = diagnostics_out or _benchmark_diagnostics_path(out)
    table_path.write_text(_kalshi_benchmark_table_markdown(report), encoding="utf-8")
    log_path.write_text(_kalshi_benchmark_detail_log(report["rows"]), encoding="utf-8")
    diagnostics_path.write_text(
        _kalshi_benchmark_diagnostics_markdown(report["diagnostics"]),
        encoding="utf-8",
    )
    typer.echo(
        "Model vs benchmark pack: "
        f"model_family={model_family}, "
        f"model Brier={report['model']['brier_score']:.6f}, "
        f"Kalshi mid Brier={report['kalshi_yes_mid']['brier_score']:.6f}"
    )


@app.command("evaluate-model1-kalshi-benchmark")
def evaluate_model1_kalshi_benchmark_command(
    model: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    examples: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    preclose_snapshots: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option()],
    table_out: Annotated[Path | None, typer.Option()] = None,
    log_out: Annotated[Path | None, typer.Option()] = None,
    diagnostics_out: Annotated[Path | None, typer.Option()] = None,
) -> None:
    artifact = MentionModelArtifact.model_validate(json.loads(model.read_text(encoding="utf-8")))
    example_rows = [
        HistoricalTrainingExample.model_validate(row)
        for row in json.loads(examples.read_text(encoding="utf-8"))
    ]
    snapshots = json.loads(preclose_snapshots.read_text(encoding="utf-8"))
    snapshot_by_market = {str(row["market_id"]): row for row in snapshots}
    benchmark_rows = []
    missing_snapshot_count = 0
    missing_quote_count = 0
    for row in example_rows:
        snapshot = snapshot_by_market.get(row.market_id)
        if snapshot is None:
            missing_snapshot_count += 1
            continue
        yes_bid = _optional_float(snapshot.get("preclose_yes_bid"))
        yes_ask = _optional_float(snapshot.get("preclose_yes_ask"))
        if yes_bid is None or yes_ask is None:
            missing_quote_count += 1
            continue
        prediction = predict_model1(
            artifact,
            company_symbol=row.company_symbol,
            feature_vector=FeatureVector(target_phrase=row.target_phrase, features=row.features),
        )
        benchmark_rows.append(
            {
                "event_id": str(snapshot.get("event_ticker") or row.company_symbol),
                "market_id": row.market_id,
                "target_phrase": row.target_phrase,
                "label": row.label,
                "model_probability": prediction.probability,
                "kalshi_yes_bid": yes_bid,
                "kalshi_yes_ask": yes_ask,
                "kalshi_yes_mid": round((yes_bid + yes_ask) / 2, 6),
                "snapshot_target_time": snapshot.get("snapshot_target_time"),
                "candle_end_ts": snapshot.get("candle_end_ts"),
            }
        )
    skip_summary = _kalshi_benchmark_skip_summary(
        total_examples=len(example_rows),
        evaluated_rows=benchmark_rows,
        missing_snapshot_count=missing_snapshot_count,
        missing_quote_count=missing_quote_count,
    )
    if not benchmark_rows:
        raise typer.BadParameter("no benchmark rows had matching bid/ask snapshots")
    diagnostics = _kalshi_benchmark_diagnostics_report(benchmark_rows)
    report = _kalshi_benchmark_metric_report(benchmark_rows)
    report["skip_summary"] = skip_summary
    report["diagnostics"] = diagnostics
    report["rows"] = benchmark_rows
    _write_json(out, report)
    table_path = table_out or _benchmark_table_path(out)
    log_path = log_out or _benchmark_log_path(out)
    diagnostics_path = diagnostics_out or _benchmark_diagnostics_path(out)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(_kalshi_benchmark_table_markdown(report), encoding="utf-8")
    log_path.write_text(_kalshi_benchmark_detail_log(benchmark_rows), encoding="utf-8")
    diagnostics_path.write_text(
        _kalshi_benchmark_diagnostics_markdown(diagnostics),
        encoding="utf-8",
    )
    typer.echo(
        "Model vs Kalshi benchmark: "
        f"model Brier={report['model']['brier_score']:.6f}, "
        f"Kalshi mid Brier={report['kalshi_yes_mid']['brier_score']:.6f}"
    )
    typer.echo(f"Wrote benchmark table to {table_path}")
    typer.echo(f"Wrote benchmark detail log to {log_path}")
    typer.echo(f"Wrote benchmark diagnostics to {diagnostics_path}")
    if skip_summary["skipped_rows"]:
        typer.echo(
            "Skipped benchmark rows: "
            f"{skip_summary['skipped_rows']} "
            f"(missing snapshots={missing_snapshot_count}, missing quotes={missing_quote_count})"
        )


@app.command("predict-model1")
def predict_model1_command(
    model: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    features: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    company_symbol: Annotated[str, typer.Option()],
    out: Annotated[Path, typer.Option()],
) -> None:
    artifact = MentionModelArtifact.model_validate(json.loads(model.read_text(encoding="utf-8")))
    feature_rows = [
        FeatureVector.model_validate(row)
        for row in json.loads(features.read_text(encoding="utf-8"))
    ]
    predictions = [
        predict_model1(artifact, company_symbol=company_symbol, feature_vector=feature_row)
        for feature_row in feature_rows
    ]
    _write_json(out, [prediction.model_dump(mode="json") for prediction in predictions])
    typer.echo(f"Wrote {len(predictions)} model1 predictions to {out}")


@app.command("train-model1-company")
def train_model1_company_command(
    examples: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    company_symbol: Annotated[str, typer.Option()],
    out: Annotated[Path, typer.Option()],
    min_company_rows: Annotated[int, typer.Option()] = 20,
    regularization_c: Annotated[float, typer.Option()] = 1.0,
    class_weight_balanced: Annotated[bool, typer.Option()] = False,
    include_target_indicator: Annotated[bool, typer.Option()] = False,
    recency_ema_half_life_quarters: Annotated[float | None, typer.Option()] = None,
    base_model: Annotated[Path | None, typer.Option()] = None,
    enforce_pretrain_gate: Annotated[bool, typer.Option()] = False,
    pretrain_gate_report_out: Annotated[Path | None, typer.Option()] = None,
) -> None:
    training_examples = [
        HistoricalTrainingExample.model_validate(row)
        for row in json.loads(examples.read_text(encoding="utf-8"))
    ]
    company_examples = [
        row for row in training_examples if row.company_symbol == company_symbol.upper()
    ]
    gate_report = _pretrain_gate_report(
        company_examples,
        min_rows=min_company_rows,
        min_events=4,
        scope=f"company:{company_symbol.upper()}",
    )
    _emit_pretrain_gate_report(gate_report)
    if pretrain_gate_report_out is not None:
        _write_json(pretrain_gate_report_out, gate_report)
    if enforce_pretrain_gate and not gate_report["passed"]:
        raise typer.BadParameter(
            "pre-train gate failed: "
            + "; ".join(gate_report["errors"])
        )
    source_global_model_version = None
    if base_model is not None:
        base_artifact = MentionModelArtifact.model_validate(
            json.loads(base_model.read_text(encoding="utf-8"))
        )
        source_global_model_version = base_artifact.model_version
    artifact = train_company_model1(
        training_examples,
        company_symbol=company_symbol,
        min_company_rows=min_company_rows,
        regularization_c=regularization_c,
        class_weight_balanced=class_weight_balanced,
        include_target_indicator=include_target_indicator,
        recency_ema_half_life_quarters=recency_ema_half_life_quarters,
        source_global_model_version=source_global_model_version,
    )
    _write_json(out, artifact.model_dump(mode="json"))
    typer.echo(
        f"Wrote company retrained model for {artifact.company_symbol} "
        f"with {artifact.training_rows} rows"
    )


@app.command("predict-model1-company")
def predict_model1_company_command(
    model: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    features: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option()],
) -> None:
    artifact = CompanyRetrainedModelArtifact.model_validate(
        json.loads(model.read_text(encoding="utf-8"))
    )
    feature_rows = [
        FeatureVector.model_validate(row)
        for row in json.loads(features.read_text(encoding="utf-8"))
    ]
    predictions = [
        predict_company_model1(artifact, feature_vector=feature_row) for feature_row in feature_rows
    ]
    _write_json(out, [prediction.model_dump(mode="json") for prediction in predictions])
    typer.echo(
        f"Wrote {len(predictions)} company-model predictions for {artifact.company_symbol} to {out}"
    )


def _write_market_workflow_artifacts(
    document: SourceDocument,
    chunks: list[DocumentChunk],
    market_title: str,
    company_symbol: str,
    market_target: TargetPhrase,
    yes_bid: Decimal,
    yes_ask: Decimal,
    out: Path,
) -> None:
    targets = _target_phrases(market_target)
    labels = label_document_chunks(chunks, targets)
    features = extract_feature_vectors(chunks, targets, labels)
    feature_by_target = {feature.target_phrase: feature for feature in features}
    prediction = RuleBasedBaseline().predict_proba(
        feature_by_target[market_target.normalized_phrase]
    )
    snapshot = MarketSnapshot(
        venue="kalshi-paper",
        market_id=f"{company_symbol}-{market_target.normalized_phrase}",
        title=market_title,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        observed_at=datetime.now(tz=UTC),
    )
    comparison = compare_prediction_to_market(prediction, snapshot)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "document.json", document.model_dump(mode="json"))
    _write_jsonl(out / "chunks.jsonl", [chunk.model_dump(mode="json") for chunk in chunks])
    _write_json(out / "labels.json", [label.model_dump(mode="json") for label in labels])
    _write_json(out / "features.json", [feature.model_dump(mode="json") for feature in features])
    _write_json(out / "prediction.json", prediction.model_dump(mode="json"))
    _write_json(out / "paper_comparison.json", comparison.model_dump(mode="json"))


def _target_phrases(market_target: TargetPhrase) -> list[TargetPhrase]:
    phrases = list(dict.fromkeys([market_target.normalized_phrase, *INITIAL_MARKET_PHRASES]))
    aliases = {
        "margin": ["restaurant-level profit margin"],
        "value proposition": ["compelling value proposition"],
    }
    return [
        TargetPhrase(
            phrase=phrase,
            normalized_phrase=phrase,
            aliases=aliases.get(phrase, []),
        )
        for phrase in phrases
    ]


def _extract_text_values(payload: object) -> list[str]:
    if isinstance(payload, list) and all(isinstance(item, str) for item in payload):
        return payload
    if isinstance(payload, list) and all(
        isinstance(item, dict) and "text" in item for item in payload
    ):
        return [str(item["text"]) for item in payload]
    raise typer.BadParameter("texts must be a JSON list of strings or objects with a text field")


def _parse_target_phrases(value: str) -> list[str]:
    phrases = normalize_and_dedupe_phrases(value.split(","))
    if not phrases:
        raise typer.BadParameter("target-phrases must contain at least one phrase")
    return phrases


def _parse_symbol_list(value: str) -> list[str]:
    symbols = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not symbols:
        raise typer.BadParameter("symbols must contain at least one ticker symbol")
    return list(dict.fromkeys(symbols))


def _parse_symbol_company_map(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    mapping: dict[str, str] = {}
    for item in value.split(";"):
        row = item.strip()
        if not row:
            continue
        if ":" not in row:
            raise typer.BadParameter(
                "symbol-company-names must use SYMBOL:Company Name;SYMBOL2:Company Name"
            )
        symbol, company_name = row.split(":", maxsplit=1)
        normalized_symbol = symbol.strip().upper()
        normalized_name = company_name.strip()
        if not normalized_symbol or not normalized_name:
            raise typer.BadParameter(
                "symbol-company-names must use SYMBOL:Company Name;SYMBOL2:Company Name"
            )
        mapping[normalized_symbol] = normalized_name
    return mapping


def _transcript_ref_sort_key(reference: FmpTranscriptReference) -> tuple[int, int]:
    return (reference.fiscal_year, reference.fiscal_quarter)


def _parse_float_values(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise typer.BadParameter("value list must contain at least one number")
    return values


def _parse_boolean_values(value: str) -> list[bool]:
    parsed: list[bool] = []
    for item in value.split(","):
        normalized = item.strip().lower()
        if not normalized:
            continue
        if normalized in {"true", "1", "yes"}:
            parsed.append(True)
        elif normalized in {"false", "0", "no"}:
            parsed.append(False)
        else:
            raise typer.BadParameter(f"invalid boolean value: {item}")
    if not parsed:
        raise typer.BadParameter("boolean value list must contain at least one value")
    return parsed


def _company_target_phrases_from_contracts(
    contracts: list[MentionMarketContract],
) -> dict[str, list[str]]:
    by_company: dict[str, list[str]] = {}
    for contract in contracts:
        company_symbol = _infer_company_symbol_from_contract(contract)
        if company_symbol is None:
            continue
        by_company.setdefault(company_symbol, []).append(contract.target_phrase.normalized_phrase)
    return {
        symbol: normalize_and_dedupe_phrases(values) for symbol, values in by_company.items()
    }


def _company_target_phrases_from_manifests(
    manifests: list[PublicDocumentManifest],
    *,
    seed_phrases: list[str],
    max_candidates: int,
) -> dict[str, list[str]]:
    texts_by_company: dict[str, list[str]] = {}
    for manifest in manifests:
        try:
            text = Path(manifest.raw_path).read_text(encoding="utf-8")
        except OSError:
            continue
        texts_by_company.setdefault(manifest.company_symbol, []).append(text)
    return {
        symbol: generate_synthetic_phrase_candidates(
            texts,
            seed_phrases=seed_phrases,
            max_candidates=max_candidates,
        )
        for symbol, texts in sorted(texts_by_company.items())
    }


def _merge_company_target_phrases(
    base: dict[str, list[str]],
    extra: dict[str, list[str]],
) -> dict[str, list[str]]:
    symbols = sorted({*base, *extra})
    return {
        symbol: normalize_and_dedupe_phrases(
            [*base.get(symbol, []), *extra.get(symbol, [])]
        )
        for symbol in symbols
    }


def _target_phrases_from_contracts(contracts: list[MentionMarketContract]) -> list[str]:
    return normalize_and_dedupe_phrases(
        [contract.target_phrase.normalized_phrase for contract in contracts]
    )


def _infer_company_symbol_from_contract(contract: MentionMarketContract) -> str | None:
    try:
        return parse_mention_market_title(contract.title).company_symbol
    except MentionMarketParseError:
        pass
    identifier = f"{contract.event_ticker}-{contract.market_id}".upper()
    marker = "KXEARNINGSMENTION"
    if marker in identifier:
        tail = identifier.split(marker, maxsplit=1)[1]
        symbol = []
        for char in tail:
            if not char.isalpha():
                break
            symbol.append(char)
        if symbol:
            return "".join(symbol)
    return None


def _dataset_progress_hook(*, total_records: int, label: str) -> Callable[[int, int], None]:
    next_emit = 0

    def emit(processed_records: int, generated_examples: int) -> None:
        nonlocal next_emit
        if processed_records < next_emit and processed_records != total_records:
            return
        typer.echo(
            f"[{label}] progress {processed_records}/{total_records} records, "
            f"{generated_examples} examples"
        )
        while next_emit <= processed_records:
            next_emit += 10

    return emit


def _load_template_features_context(
    template_catalog: Path | None,
) -> tuple[dict[str, list[str]] | None, CachedEmbeddingProvider | None]:
    if template_catalog is None:
        return None, None
    catalog_payload = json.loads(template_catalog.read_text(encoding="utf-8"))
    catalog = TemplatePhraseCatalog.model_validate(catalog_payload)
    provider = _load_embedding_provider(
        "OPENAI_API_KEY is required when template-catalog is provided"
    )
    template_texts = list(
        dict.fromkeys(
            template for values in catalog.phrase_variants.values() for template in values
        )
    )
    if template_texts:
        typer.echo(f"Prewarming template embeddings cache for {len(template_texts)} templates...")
        provider.embed(template_texts)
        typer.echo("Template cache prewarm complete")
    return catalog.phrase_variants, provider


def _load_scenario_texts_by_event(
    scenario_catalogs: Path | None,
) -> dict[str, list[str]] | None:
    if scenario_catalogs is None:
        return None
    payload = json.loads(scenario_catalogs.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else [payload]
    catalogs = [EventScenarioCatalog.model_validate(row) for row in rows]
    return {
        catalog.event_id: catalog.scenario_texts()
        for catalog in catalogs
        if catalog.scenario_texts()
    }


def _load_embedding_provider(error_message: str) -> CachedEmbeddingProvider:
    settings = _load_settings()
    if settings.openai_api_key is None:
        raise typer.BadParameter(error_message)
    return CachedEmbeddingProvider(
        provider=OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.openai_embedding_model,
            dimensions=settings.openai_embedding_dimensions,
            batch_size=settings.embedding_batch_size,
        ),
        cache_path=settings.embedding_cache_path,
    )


def _documents_by_period_from_manifests(
    manifest_rows: list[PublicDocumentManifest],
):
    documents_by_period = {}
    for manifest in manifest_rows:
        if not _manifest_matches_fiscal_period(manifest):
            continue
        key = (manifest.company_symbol, manifest.fiscal_year, manifest.fiscal_quarter)
        documents_by_period.setdefault(key, []).append(
            source_document_from_text_file(
                path=Path(manifest.raw_path),
                company_symbol=manifest.company_symbol,
                fiscal_year=manifest.fiscal_year,
                fiscal_quarter=manifest.fiscal_quarter,
                published_at=manifest.published_at,
                document_type=manifest.source_type,
            )
        )
    return documents_by_period


def _manifest_matches_fiscal_period(manifest: PublicDocumentManifest) -> bool:
    return abs(manifest.published_at.year - manifest.fiscal_year) <= 1


def _parse_probability(value: str, name: str) -> Decimal:
    try:
        probability = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be a decimal probability") from exc
    if probability < 0 or probability > 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return probability


def _load_settings() -> Settings:
    env_file = Path(".env")
    if env_file.exists():
        return Settings(_env_file=env_file)
    return Settings()


def _group_transcript_records(transcript_root: Path):
    grouped = {}
    for record in scan_transcript_corpus(transcript_root):
        grouped.setdefault(record.company_name, []).append(record)
    return [
        (
            company_name,
            sorted(
                records,
                key=lambda record: (record.fiscal_year, record.fiscal_quarter),
                reverse=True,
            ),
        )
        for company_name, records in sorted(
            grouped.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )
    ]


def _pair_records_to_filings(records, filings):
    used_filing_indexes = set()
    pairs = []
    for record in records:
        match = _best_filing_for_record(record, filings, used_filing_indexes)
        if match is None:
            continue
        filing_index, filing = match
        used_filing_indexes.add(filing_index)
        pairs.append((record, filing))
    return pairs


def _best_filing_for_record(record, filings, used_filing_indexes):
    candidates = []
    for index, filing in enumerate(filings):
        if index in used_filing_indexes:
            continue
        year_distance = abs(filing.filed_at.year - record.fiscal_year)
        if year_distance > 1:
            continue
        quarter_distance = abs(_calendar_quarter(filing.filed_at.month) - record.fiscal_quarter)
        candidates.append((year_distance, quarter_distance, index, filing))
    if not candidates:
        return None
    _, _, index, filing = min(candidates, key=lambda candidate: candidate[:3])
    return index, filing


def _calendar_quarter(month: int) -> int:
    return ((month - 1) // 3) + 1


def _calendar_fiscal_period(published_at: datetime) -> tuple[int, int]:
    return published_at.year, _calendar_quarter(published_at.month)


def _is_opinion_article(article) -> bool:
    datatype = str(getattr(article, "datatype", "") or "").lower()
    title = str(getattr(article, "title", "") or "").lower()
    if datatype in {"opinion", "analysis", "blog"}:
        return True
    return any(token in title for token in ["opinion", "analysis", "editorial", "what this means"])


def _news_source_reliability(source_priority: int | None) -> float:
    if source_priority is None:
        return 0.65
    if source_priority <= 1:
        return 0.95
    if source_priority == 2:
        return 0.9
    if source_priority == 3:
        return 0.85
    if source_priority == 4:
        return 0.8
    if source_priority == 5:
        return 0.75
    return max(0.55, 0.72 - ((source_priority - 5) * 0.02))


def _tiingo_source_reliability(source_name: str | None) -> float:
    if not source_name:
        return 0.7
    normalized = source_name.lower()
    high_trust_tokens = ("reuters", "bloomberg", "wsj", "financial times", "marketwatch")
    medium_trust_tokens = ("yahoo", "cnbc", "benzinga", "insider monkey")
    if any(token in normalized for token in high_trust_tokens):
        return 0.9
    if any(token in normalized for token in medium_trust_tokens):
        return 0.78
    return 0.7


def _defeatbeta_source_reliability(source_name: str | None) -> float:
    if not source_name:
        return 0.72
    normalized = source_name.lower()
    high_trust_tokens = ("reuters", "bloomberg", "wsj", "financial times", "marketwatch")
    medium_trust_tokens = ("yahoo", "cnbc", "benzinga", "insider monkey")
    if any(token in normalized for token in high_trust_tokens):
        return 0.9
    if any(token in normalized for token in medium_trust_tokens):
        return 0.78
    return 0.72


def _estimated_call_time_for_period(*, fiscal_year: int, fiscal_quarter: int) -> datetime:
    period_end_month = fiscal_quarter * 3
    quarter_end_day = 31 if period_end_month in {3, 12} else 30
    fiscal_period_end = datetime(
        year=fiscal_year,
        month=period_end_month,
        day=quarter_end_day,
        tzinfo=UTC,
    )
    # Earnings calls are typically held several weeks after quarter close.
    return fiscal_period_end + timedelta(days=50)


def _collect_yfinance_news(
    *,
    company_symbol: str,
    company_name: str,
    from_date: str,
    to_date: str,
    max_articles: int,
) -> list[TiingoArticle]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise typer.BadParameter(
            "yfinance is not installed; install it to enable fallback news ingestion"
        ) from exc

    start = datetime.fromisoformat(from_date).date()
    end = datetime.fromisoformat(to_date).date()
    search_queries = [company_symbol.upper(), f"{company_name} earnings"]
    by_article_id: dict[str, TiingoArticle] = {}
    for query in search_queries:
        try:
            rows = yf.Search(query, news_count=max_articles).news
        except Exception:
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            article_id = str(row.get("uuid") or row.get("id") or "").strip()
            title = str(row.get("title") or "").strip()
            link = str(row.get("link") or "").strip()
            publish_ts = row.get("providerPublishTime")
            if not article_id or not title or not link or not publish_ts:
                continue
            try:
                published_at = datetime.fromtimestamp(int(publish_ts), tz=UTC)
            except (TypeError, ValueError, OSError):
                continue
            published_date = published_at.date()
            if published_date < start or published_date > end:
                continue
            article = TiingoArticle(
                article_id=article_id,
                title=title,
                link=link,
                source_name=str(row.get("publisher") or "").strip() or "yfinance",
                description=str(row.get("summary") or "").strip() or None,
                datatype=str(row.get("type") or "story").strip().lower(),
                published_at=published_at,
                tickers=[str(value).upper() for value in row.get("relatedTickers", []) or []],
                tags=[],
            )
            by_article_id.setdefault(article_id, article)
            if len(by_article_id) >= max_articles:
                break
        if len(by_article_id) >= max_articles:
            break
    return sorted(by_article_id.values(), key=lambda row: row.published_at, reverse=True)[
        :max_articles
    ]


def _news_article_text(*, article, company_symbol: str) -> str:
    parts = [
        "# NewsData article",
        f"Company symbol: {company_symbol}",
        f"Article ID: {getattr(article, 'article_id', '')}",
        f"Published at: {getattr(article, 'published_at', '')}",
        f"Source: {getattr(article, 'source_name', '') or 'unknown'}",
        f"Source priority: {getattr(article, 'source_priority', '')}",
        f"Datatype: {getattr(article, 'datatype', '')}",
        f"Title: {getattr(article, 'title', '')}",
        f"URL: {getattr(article, 'link', '')}",
        "",
        "Description:",
        str(getattr(article, "description", "") or ""),
        "",
        "Content:",
        str(getattr(article, "content", "") or ""),
        "",
    ]
    return "\n".join(parts).strip() + "\n"


def _read_company_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {str(key).upper(): str(value) for key, value in json.loads(path.read_text()).items()}


def _sec_exhibit_source_type(document_type: str) -> str:
    normalized = document_type.lower().replace("-", "_").replace(".", "_")
    return f"sec_{normalized}_supplemental"


def _pretrain_gate_report(
    examples: list[HistoricalTrainingExample],
    *,
    min_rows: int,
    min_events: int,
    scope: str,
) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    sample_count = len(examples)
    event_keys = {
        (row.company_symbol, row.fiscal_year, row.fiscal_quarter)
        for row in examples
    }
    positive_labels = sum(row.label for row in examples)
    positive_rate = (positive_labels / sample_count) if sample_count else 0.0
    leakage_violations = 0
    for row in examples:
        latest_allowed = _latest_allowed_evidence_cutoff(
            fiscal_year=row.fiscal_year,
            fiscal_quarter=row.fiscal_quarter,
        )
        if row.evidence_cutoff > latest_allowed:
            leakage_violations += 1
    exact_feature_coverage = (
        sum("exact_signal_binary" in row.features for row in examples) / sample_count
        if sample_count
        else 0.0
    )
    semantic_feature_coverage = (
        sum("semantic_signal_max_tfidf" in row.features for row in examples) / sample_count
        if sample_count
        else 0.0
    )
    hard_negative_feature_coverage = (
        sum("hard_negative_neighbor_present" in row.features for row in examples) / sample_count
        if sample_count
        else 0.0
    )
    news_rows = (
        sum(row.features.get("evidence_news_doc_ratio", 0.0) > 0 for row in examples)
        if sample_count
        else 0
    )
    if sample_count < min_rows:
        errors.append(
            f"sample_count={sample_count} below required min_rows={min_rows}"
        )
    if len(event_keys) < min_events:
        errors.append(
            f"event_count={len(event_keys)} below required min_events={min_events}"
        )
    if sample_count > 0 and (positive_rate <= 0.0 or positive_rate >= 1.0):
        errors.append(
            f"positive_rate={positive_rate:.4f} indicates one-class training labels"
        )
    if leakage_violations > 0:
        errors.append(
            f"{leakage_violations} examples exceed latest allowed evidence cutoff windows"
        )
    if sample_count > 0 and exact_feature_coverage < 1.0:
        warnings.append(
            f"exact feature coverage={exact_feature_coverage:.3f}; some rows miss exact channel"
        )
    if sample_count > 0 and semantic_feature_coverage < 1.0:
        warnings.append(
            f"semantic feature coverage={semantic_feature_coverage:.3f}; some rows miss semantic channel"
        )
    if sample_count > 0 and hard_negative_feature_coverage < 1.0:
        warnings.append(
            "hard-negative feature channel missing from some rows"
        )
    if sample_count > 0 and news_rows == 0:
        warnings.append("news evidence ratio is zero for all rows")
    return {
        "scope": scope,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "metrics": {
            "sample_count": sample_count,
            "event_count": len(event_keys),
            "positive_rate": round(positive_rate, 6),
            "news_rows": news_rows,
            "leakage_violations": leakage_violations,
            "exact_feature_coverage": round(exact_feature_coverage, 6),
            "semantic_feature_coverage": round(semantic_feature_coverage, 6),
            "hard_negative_feature_coverage": round(hard_negative_feature_coverage, 6),
        },
    }


def _latest_allowed_evidence_cutoff(*, fiscal_year: int, fiscal_quarter: int) -> datetime:
    period_end_month = fiscal_quarter * 3
    quarter_end_day = 31 if period_end_month in {3, 12} else 30
    period_end = datetime(
        year=fiscal_year,
        month=period_end_month,
        day=quarter_end_day,
        tzinfo=UTC,
    )
    return period_end + timedelta(days=120)


def _emit_pretrain_gate_report(report: dict) -> None:
    status = "PASS" if report.get("passed") else "FAIL"
    typer.echo(
        f"Pre-train gate [{report.get('scope', 'unknown')}] {status}: "
        f"{json.dumps(report.get('metrics', {}), sort_keys=True)}"
    )
    for message in report.get("warnings", []):
        typer.echo(f"Pre-train gate warning: {message}")
    for message in report.get("errors", []):
        typer.echo(f"Pre-train gate error: {message}")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise typer.BadParameter(f"{path} must contain a JSON object")
    return payload


def _load_model_rows(path: Path, model_class):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise typer.BadParameter(f"{path} must contain a JSON array")
    return [model_class.model_validate(row) for row in payload]


def _model1_calibration_summary(artifact: MentionModelArtifact) -> str:
    if artifact.temperature_calibration is not None:
        return f"temperature:{artifact.temperature_calibration.temperature}"
    if artifact.isotonic_calibration is not None:
        return "isotonic"
    return "none"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _brier_score(probabilities: list[float], labels: list[int]) -> float:
    squared_errors = [
        (probability - label) ** 2
        for probability, label in zip(probabilities, labels, strict=True)
    ]
    return round(sum(squared_errors) / len(labels), 6)


def _ece_score(probabilities: list[float], labels: list[int], bins: int = 10) -> float:
    if not probabilities:
        return 0.0
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must be the same length")
    ece = 0.0
    total = len(probabilities)
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        if index == bins - 1:
            members = [
                (probability, label)
                for probability, label in zip(probabilities, labels, strict=True)
                if lower <= probability <= upper
            ]
        else:
            members = [
                (probability, label)
                for probability, label in zip(probabilities, labels, strict=True)
                if lower <= probability < upper
            ]
        if not members:
            continue
        mean_probability = sum(probability for probability, _ in members) / len(members)
        mean_label = sum(label for _, label in members) / len(members)
        ece += (len(members) / total) * abs(mean_label - mean_probability)
    return round(ece, 6)


def _kalshi_benchmark_metric_report(rows: list[dict], *, include_per_event: bool = True) -> dict:
    labels = [int(row["label"]) for row in rows]
    report = {
        "sample_count": len(rows),
        "model": _probability_metric_block(
            [float(row["model_probability"]) for row in rows],
            labels,
        ),
        "kalshi_yes_bid": _probability_metric_block(
            [float(row["kalshi_yes_bid"]) for row in rows],
            labels,
        ),
        "kalshi_yes_ask": _probability_metric_block(
            [float(row["kalshi_yes_ask"]) for row in rows],
            labels,
        ),
        "kalshi_yes_mid": _probability_metric_block(
            [float(row["kalshi_yes_mid"]) for row in rows],
            labels,
        ),
        "per_event": {},
    }
    report["deltas_vs_kalshi"] = {
        "model_minus_yes_bid_brier": round(
            report["model"]["brier_score"] - report["kalshi_yes_bid"]["brier_score"],
            6,
        ),
        "model_minus_yes_ask_brier": round(
            report["model"]["brier_score"] - report["kalshi_yes_ask"]["brier_score"],
            6,
        ),
        "model_minus_yes_mid_brier": round(
            report["model"]["brier_score"] - report["kalshi_yes_mid"]["brier_score"],
            6,
        ),
        "model_minus_yes_mid_ece": round(
            report["model"]["expected_calibration_error"]
            - report["kalshi_yes_mid"]["expected_calibration_error"],
            6,
        ),
    }
    if include_per_event:
        rows_by_event: dict[str, list[dict]] = {}
        for row in rows:
            rows_by_event.setdefault(str(row["event_id"]), []).append(row)
        report["per_event"] = {
            event_id: _kalshi_benchmark_metric_report(event_rows, include_per_event=False)
            for event_id, event_rows in sorted(rows_by_event.items())
        }
    return report


def _kalshi_benchmark_table_markdown(report: dict) -> str:
    lines = [
        "# Kalshi Benchmark Summary",
        "",
        "| Event | Rows | Model Brier | Model ECE | Kalshi Mid Brier | "
        "Kalshi Mid ECE | Model minus Kalshi mid Brier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    summary_rows = [("ALL", report)]
    summary_rows.extend((event_id, row) for event_id, row in report.get("per_event", {}).items())
    for event_id, row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    event_id,
                    str(row["sample_count"]),
                    _format_probability(row["model"]["brier_score"]),
                    _format_probability(row["model"]["expected_calibration_error"]),
                    _format_probability(row["kalshi_yes_mid"]["brier_score"]),
                    _format_probability(
                        row["kalshi_yes_mid"]["expected_calibration_error"]
                    ),
                    _format_probability(
                        row["deltas_vs_kalshi"]["model_minus_yes_mid_brier"]
                    ),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Negative `Model minus Kalshi mid Brier` means the model beat Kalshi mid.",
        ]
    )
    return "\n".join(lines) + "\n"


def _kalshi_benchmark_detail_log(rows: list[dict]) -> str:
    columns = [
        ("market_id", 46),
        ("target_phrase", 22),
        ("ask", 6),
        ("bid", 6),
        ("mid", 6),
        ("model", 6),
        ("outcome", 7),
        ("event_id", 34),
        ("snapshot_target_time", 25),
        ("candle_ts", 12),
    ]
    lines = [_fixed_width_log_row([name for name, _ in columns], columns)]
    for row in rows:
        lines.append(
            _fixed_width_log_row(
                [
                    str(row.get("market_id", "")),
                    str(row.get("target_phrase", "")),
                    _format_log_probability(float(row["kalshi_yes_ask"])),
                    _format_log_probability(float(row["kalshi_yes_bid"])),
                    _format_log_probability(float(row["kalshi_yes_mid"])),
                    _format_log_probability(float(row["model_probability"])),
                    str(int(row["label"])),
                    str(row.get("event_id", "")),
                    str(row.get("snapshot_target_time", "")),
                    str(row.get("candle_end_ts", "")),
                ],
                columns,
            )
        )
    return "\n".join(lines) + "\n"


def _kalshi_benchmark_diagnostics_report(rows: list[dict]) -> dict:
    worst_rows = sorted(
        (_diagnostic_row(row) for row in rows),
        key=lambda row: row["brier_contribution"],
        reverse=True,
    )
    probability_bins = [
        _diagnostic_probability_bin(rows, lower=index / 10, upper=(index + 1) / 10)
        for index in range(10)
    ]
    probabilities = [float(row["model_probability"]) for row in rows]
    labels = [int(row["label"]) for row in rows]
    return {
        "sample_count": len(rows),
        "false_positive_count": sum(
            1
            for row in rows
            if float(row["model_probability"]) >= 0.5 and int(row["label"]) == 0
        ),
        "false_negative_count": sum(
            1
            for row in rows
            if float(row["model_probability"]) < 0.5 and int(row["label"]) == 1
        ),
        "probability_saturation": {
            "at_or_above_0_90_count": sum(probability >= 0.9 for probability in probabilities),
            "at_or_below_0_10_count": sum(probability <= 0.1 for probability in probabilities),
            "distinct_model_probability_count": len(
                {round(probability, 6) for probability in probabilities}
            ),
        },
        "quote_artifacts": {
            "wide_spread_count": sum(_is_wide_spread_quote(row) for row in rows),
            "bid_zero_ask_one_count": sum(
                float(row["kalshi_yes_bid"]) <= 0.0 and float(row["kalshi_yes_ask"]) >= 1.0
                for row in rows
            ),
        },
        "model_base_rate": round(sum(probabilities) / len(probabilities), 6)
        if probabilities
        else 0.0,
        "label_positive_rate": round(sum(labels) / len(labels), 6) if labels else 0.0,
        "probability_bins": [bucket for bucket in probability_bins if bucket["sample_count"]],
        "phrase_categories": _phrase_category_blocks(rows),
        "worst_rows": worst_rows[:20],
    }


def _kalshi_benchmark_skip_summary(
    *,
    total_examples: int,
    evaluated_rows: list[dict],
    missing_snapshot_count: int,
    missing_quote_count: int,
) -> dict:
    evaluated_count = len(evaluated_rows)
    return {
        "total_examples": total_examples,
        "evaluated_rows": evaluated_count,
        "skipped_rows": total_examples - evaluated_count,
        "missing_snapshot_count": missing_snapshot_count,
        "missing_quote_count": missing_quote_count,
    }


def _diagnostic_probability_bin(rows: list[dict], *, lower: float, upper: float) -> dict:
    if upper >= 1.0:
        bucket_rows = [
            row for row in rows if lower <= float(row["model_probability"]) <= upper
        ]
    else:
        bucket_rows = [
            row for row in rows if lower <= float(row["model_probability"]) < upper
        ]
    labels = [int(row["label"]) for row in bucket_rows]
    probabilities = [float(row["model_probability"]) for row in bucket_rows]
    return {
        "lower": round(lower, 1),
        "upper": round(upper, 1),
        "sample_count": len(bucket_rows),
        "mean_probability": round(sum(probabilities) / len(probabilities), 6)
        if probabilities
        else 0.0,
        "positive_rate": round(sum(labels) / len(labels), 6) if labels else 0.0,
        "brier_score": _brier_score(probabilities, labels) if labels else 0.0,
    }


def _phrase_category_blocks(rows: list[dict]) -> dict:
    rows_by_category: dict[str, list[dict]] = {}
    for row in rows:
        rows_by_category.setdefault(
            _phrase_category(str(row.get("target_phrase", ""))),
            [],
        ).append(row)
    return {
        category: _phrase_category_block(category_rows)
        for category, category_rows in sorted(rows_by_category.items())
    }


def _phrase_category_block(rows: list[dict]) -> dict:
    labels = [int(row["label"]) for row in rows]
    probabilities = [float(row["model_probability"]) for row in rows]
    return {
        "sample_count": len(rows),
        "brier_score": _brier_score(probabilities, labels) if labels else 0.0,
        "false_positive_count": sum(
            1
            for row in rows
            if float(row["model_probability"]) >= 0.5 and int(row["label"]) == 0
        ),
        "false_negative_count": sum(
            1
            for row in rows
            if float(row["model_probability"]) < 0.5 and int(row["label"]) == 1
        ),
    }


def _diagnostic_row(row: dict) -> dict:
    probability = float(row["model_probability"])
    label = int(row["label"])
    return {
        "market_id": str(row.get("market_id", "")),
        "event_id": str(row.get("event_id", "")),
        "target_phrase": str(row.get("target_phrase", "")),
        "phrase_category": _phrase_category(str(row.get("target_phrase", ""))),
        "direction": _diagnostic_direction(probability, label),
        "brier_contribution": round((probability - label) ** 2, 6),
        "model_probability": round(probability, 6),
        "label": label,
        "kalshi_yes_mid": round(float(row["kalshi_yes_mid"]), 6),
    }


def _diagnostic_direction(probability: float, label: int) -> str:
    if probability >= 0.5 and label == 0:
        return "false_positive"
    if probability < 0.5 and label == 1:
        return "false_negative"
    return "correct_side"


def _is_wide_spread_quote(row: dict) -> bool:
    return float(row["kalshi_yes_ask"]) - float(row["kalshi_yes_bid"]) >= 0.5


def _phrase_category(phrase: str) -> str:
    normalized = phrase.lower()
    if "/" in normalized:
        return "alias"
    macro_terms = {
        "china",
        "inflation",
        "iran",
        "oil",
        "rate cut",
        "recession",
        "tariff",
    }
    competitor_terms = {
        "anthropic",
        "deepmind",
        "gemini",
        "nvidia",
        "openai",
    }
    codename_terms = {
        "alexa+",
        "circle to search",
        "fairwater",
        "ironwood",
        "liquid glass",
        "nano banana",
        "project kuiper",
        "rufus",
        "siri",
        "wiz",
    }
    if normalized in macro_terms:
        return "macro"
    if normalized in competitor_terms:
        return "competitor"
    if normalized in codename_terms:
        return "codename_or_product"
    if len(normalized.split()) >= 2:
        return "multiword"
    return "generic"


def _kalshi_benchmark_diagnostics_markdown(diagnostics: dict) -> str:
    lines = [
        "# Benchmark Diagnostics",
        "",
        f"- Rows: `{diagnostics['sample_count']}`",
        f"- False positives: `{diagnostics['false_positive_count']}`",
        f"- False negatives: `{diagnostics['false_negative_count']}`",
        f"- Model mean probability: `{diagnostics['model_base_rate']:.6f}`",
        f"- Label positive rate: `{diagnostics['label_positive_rate']:.6f}`",
        "",
        "## Probability Saturation",
        "",
        "| Metric | Count |",
        "|---|---:|",
        "| p >= 0.90 | "
        f"{diagnostics['probability_saturation']['at_or_above_0_90_count']} |",
        "| p <= 0.10 | "
        f"{diagnostics['probability_saturation']['at_or_below_0_10_count']} |",
        "| Distinct probabilities | "
        f"{diagnostics['probability_saturation']['distinct_model_probability_count']} |",
        "",
        "## Worst Rows",
        "",
        "| Market | Event | Phrase | Category | Direction | Brier | Model | Label | Mid |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in diagnostics["worst_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["market_id"]),
                    str(row["event_id"]),
                    str(row["target_phrase"]),
                    str(row["phrase_category"]),
                    str(row["direction"]),
                    _format_probability(float(row["brier_contribution"])),
                    _format_probability(float(row["model_probability"])),
                    str(row["label"]),
                    _format_probability(float(row["kalshi_yes_mid"])),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _fixed_width_log_row(values: list[str], columns: list[tuple[str, int]]) -> str:
    return "  ".join(
        _clip_log_cell(value, width).ljust(width)
        for value, (_, width) in zip(values, columns, strict=True)
    ).rstrip()


def _clip_log_cell(value: str, width: int) -> str:
    return value[:width]


def _format_probability(value: float) -> str:
    return f"{value:.6f}"


def _format_log_probability(value: float) -> str:
    return f"{value:.3f}"


def _benchmark_table_path(out: Path) -> Path:
    return out.with_name(f"{out.stem}.table.md")


def _benchmark_log_path(out: Path) -> Path:
    return out.with_suffix(".log")


def _benchmark_diagnostics_path(out: Path) -> Path:
    return out.with_name(f"{out.stem}.diagnostics.md")


def _probability_metric_block(probabilities: list[float], labels: list[int]) -> dict:
    return {
        "brier_score": _brier_score(probabilities, labels) if labels else 0.0,
        "expected_calibration_error": _ece_score(probabilities, labels),
        "log_loss": _safe_log_loss(labels, probabilities) if labels else 0.0,
    }


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _safe_log_loss(labels: list[int], probabilities: list[float]) -> float:
    clipped = [min(0.99, max(0.01, probability)) for probability in probabilities]
    return round(float(log_loss(labels, clipped, labels=[0, 1])), 6)


if __name__ == "__main__":
    app()
