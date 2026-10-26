"""One startup-loaded frozen system and compact durable analyst/demo store."""

import gzip
import json
from threading import RLock

import numpy as np
import pandas as pd
from sqlalchemy import URL, create_engine, func, insert, select, text
from sqlalchemy.orm import Session

from riskbyte_api.core.config import REPOSITORY_ROOT
from riskbyte_api.db.base import Base
from riskbyte_api.db.models import AuditEvent, Investigation, Prediction, Transaction, now
from riskbyte_api.db.session import create_database_engine
from riskbyte_ml.abrf.contracts import A_THRESHOLD, P_THRESHOLD, FusionInput
from riskbyte_ml.abrf.engine import fuse
from riskbyte_ml.anomaly.development import DevelopmentInputs
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.datasets.synthetic_riskbyte.config import load_synthetic_config
from riskbyte_ml.explainability.schemas import NOTICE, UnifiedExplanation
from riskbyte_ml.final_holdout.pipeline import verify_locked
from riskbyte_ml.inference.engine import FrozenSystem, future_features
from riskbyte_ml.statistics.verification import read_json


def response_from_explanation(e):
    f = e.fusion
    return {
        "transaction_id": e.transaction_id,
        "risk": {
            "score": f.final_risk_score,
            "level": f.risk_level,
            "investigation_required": f.investigation_flag,
        },
        "supervised": {
            "probability": e.supervised.calibrated_probability,
            "prediction": e.supervised.prediction,
            "model_version": e.supervised.model_version,
        },
        "anomaly": {
            "score": e.anomaly.score,
            "flag": e.anomaly.flag,
            "diagnostic_only": True,
            "abrf_weight": 0.0,
        },
        "behaviour": {
            "score": e.behaviour.behaviour_score,
            "confidence": e.behaviour.behaviour_confidence,
            "history_count": e.behaviour.history_count,
            "available": e.behaviour.behaviour_score is not None,
        },
        "abrf": {
            "version": f.version,
            "base_weights": f.base_weights.model_dump(),
            "effective_weights": f.effective_weights,
            "contributions": f.weighted_contributions,
            "reason_codes": f.reason_codes,
        },
        "explanation": {
            "status": "AVAILABLE",
            "top_increasing_features": [
                x.model_dump(mode="json") for x in e.supervised.top_positive_features
            ],
            "top_decreasing_features": [
                x.model_dump(mode="json") for x in e.supervised.top_negative_features
            ],
            "behavioural_reasons": [x.model_dump(mode="json") for x in e.behaviour.components[:5]],
        },
        "versions": {
            "feature": "synthetic-features-v1:entity-v1:strict-prior-v1:encoding-v1",
            "anomaly": "anomaly-detection-v1",
            "behaviour": "behavioural-deviation-v1",
            "abrf": "abrf-v1",
            "explainability": "explainability-v1.1-shap",
            "shap": "0.53.0",
        },
    }


class RuntimeService:
    def __init__(self, settings):
        self.settings = settings
        self.lock = RLock()
        config_path = settings.dataset_config_path
        config_path = config_path if config_path.is_absolute() else REPOSITORY_ROOT / config_path
        config = load_synthetic_config(config_path)
        self.root = (
            settings.abrf_artifacts_dir
            if settings.abrf_artifacts_dir.is_absolute()
            else REPOSITORY_ROOT / settings.abrf_artifacts_dir
        ).parent
        self.final_dir = self.root / "final_holdout"
        self.manifest = verify_locked(self.final_dir)
        self.system = FrozenSystem(config, self.root)
        development = DevelopmentInputs(config, self.root)
        self.historical_id_limit = development.generation["transaction_count"]
        self.history = development.frame.drop(columns="isFraud")
        if settings.database_url:
            self.engine = create_database_engine(settings)
        else:
            db = settings.development_database_path
            db = db if db.is_absolute() else REPOSITORY_ROOT / db
            db.parent.mkdir(parents=True, exist_ok=True)
            self.engine = create_engine(
                URL.create("sqlite+pysqlite", database=str(db)),
                connect_args={"check_same_thread": False},
                echo=False,
            )
        # Local development bootstrap; production PostgreSQL uses reviewed Alembic migration.
        if self.engine.dialect.name == "sqlite":
            Base.metadata.create_all(self.engine)
        case_map = {
            c["explanation"]["transaction_id"]: c["explanation"]
            for c in read_json(self.final_dir / "case_studies.json")["cases"]
            if c["status"] == "AVAILABLE"
        }
        contexts = []
        with Session(self.engine) as session:
            existing = set(session.scalars(select(Transaction.id)))
            chunk = []
            with gzip.open(
                self.final_dir / "demo_records.jsonl.gz", "rt", encoding="utf-8"
            ) as stream:
                for line in stream:
                    r = json.loads(line)
                    contexts.append(r["transaction"])
                    tid = str(r["transaction"]["TransactionID"])
                    if tid not in existing:
                        chunk.append((tid, r))
                    if len(chunk) >= 500:
                        self._import_chunk(session, chunk, case_map)
                        chunk = []
                if chunk:
                    self._import_chunk(session, chunk, case_map)
            session.commit()
            self.watermark = float(
                session.scalar(select(func.max(Transaction.elapsed_seconds))) or 0
            )
        self.final_reports = {
            name: read_json(self.final_dir / file)
            for name, file in {
                "final": "final_metrics.json",
                "scenarios": "scenario_metrics.json",
                "hard-negatives": "hard_negative_metrics.json",
            }.items()
        }

    def _import_chunk(self, session, chunk, case_map):
        transactions = []
        predictions = []
        investigations = []
        for tid, r in chunk:
            tx, feature, p = r["transaction"], r["features"], r["prediction"]
            explanation = case_map.get(tid)
            trace = fuse(
                FusionInput(
                    transaction_id=tid,
                    supervised_probability=float(p["P"]),
                    supervised_prediction=p["P"] >= P_THRESHOLD,
                    anomaly_score=float(p["A"]),
                    anomaly_prediction=p["A"] >= A_THRESHOLD,
                    behaviour_score=p["B"],
                    behaviour_confidence=p["confidence"],
                    behaviour_available=p["B"] is not None,
                ),
                self.system.fusion_config,
            )
            result = (
                response_from_explanation(UnifiedExplanation.model_validate(explanation))
                if explanation
                else {
                    "transaction_id": tid,
                    "risk": {
                        "score": p["risk"],
                        "level": p["risk_level"],
                        "investigation_required": p["investigation_flag"],
                    },
                    "supervised": {
                        "probability": p["P"],
                        "prediction": p["P"] >= self.system.classifier.threshold,
                        "model_version": self.system.classifier.model_version,
                    },
                    "anomaly": {
                        "score": p["A"],
                        "flag": p["A"] >= A_THRESHOLD,
                        "diagnostic_only": True,
                        "abrf_weight": 0.0,
                    },
                    "behaviour": {
                        "score": p["B"],
                        "confidence": p["confidence"],
                        "history_count": int(feature["behavioural_history_count"]),
                        "available": p["B"] is not None,
                    },
                    "abrf": {
                        "version": trace.version,
                        "base_weights": trace.base_weights.model_dump(),
                        "effective_weights": trace.effective_weights,
                        "contributions": trace.weighted_contributions,
                        "reason_codes": trace.reason_codes,
                    },
                    "explanation": {
                        "status": "NOT_REQUESTED",
                        "top_increasing_features": [],
                        "top_decreasing_features": [],
                        "behavioural_reasons": [],
                    },
                    "versions": {
                        "feature": self.system.provenance["feature_version"],
                        "anomaly": "anomaly-detection-v1",
                        "behaviour": "behavioural-deviation-v1",
                        "abrf": "abrf-v1",
                        "explainability": "explainability-v1.1-shap",
                        "shap": "0.53.0",
                    },
                }
            )
            transactions.append(
                {
                    "id": tid,
                    "elapsed_seconds": float(tx["TransactionDT"]),
                    "entity": tx["behavioural_entity_id"],
                    "context": tx,
                    "features": feature,
                }
            )
            predictions.append(
                {
                    "transaction_id": tid,
                    "risk_score": p["risk"],
                    "risk_level": p["risk_level"],
                    "investigation_required": p["investigation_flag"],
                    "result": result,
                    "explanation": explanation,
                }
            )
            investigations.append({"transaction_id": tid, "state": "UNREVIEWED"})
        session.execute(insert(Transaction), transactions)
        session.execute(insert(Prediction), predictions)
        session.execute(insert(Investigation), investigations)
        session.execute(
            insert(AuditEvent),
            [
                {
                    "transaction_id": tid,
                    "action": "prediction_imported",
                    "safe_metadata": {"origin": "locked_synthetic_final_holdout"},
                }
                for tid, _ in chunk
            ],
        )

    def predict(self, transactions):
        if len(transactions) > self.settings.max_batch_size:
            raise DatasetError("batch_limit_exceeded", "Batch exceeds configured limit", 413)
        frame = pd.DataFrame([t.model_dump() for t in transactions])
        with self.lock, Session(self.engine) as session:
            if self.engine.dialect.name == "sqlite":
                session.connection().exec_driver_sql("BEGIN IMMEDIATE")
            elif self.engine.dialect.name == "postgresql":
                session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": 70517})
            if (
                frame.TransactionID.duplicated().any()
                or any(session.get(Transaction, id_) for id_ in frame.TransactionID)
                or any(
                    id_.isdigit() and 1 <= int(id_) <= getattr(self, "historical_id_limit", 0)
                    for id_ in frame.TransactionID
                )
            ):
                raise DatasetError(
                    "transaction_exists", "Transaction IDs must be new and unique", 409
                )
            watermark = float(
                session.scalar(select(func.max(Transaction.elapsed_seconds))) or self.watermark
            )
            if frame.TransactionDT.min() < watermark:
                raise DatasetError(
                    "out_of_order_transaction", "Elapsed time precedes committed history", 409
                )
            history = self.history[
                self.history.behavioural_entity_id.isin(frame.behavioural_entity_id)
            ]
            observed = list(
                session.scalars(
                    select(Transaction.context).where(
                        Transaction.entity.in_(frame.behavioural_entity_id.unique().tolist())
                    )
                )
            )
            history = pd.concat([history, pd.DataFrame(observed)], ignore_index=True)
            features = future_features(history, frame)
            scores = self.system.score(features)
            results = []
            for i in range(len(frame)):
                e = self.system.explain(frame.iloc[i], features.iloc[[i]], scores, i)
                response = response_from_explanation(e)
                result_features = {
                    k: None if pd.isna(v) else v.item() if isinstance(v, np.generic) else v
                    for k, v in features.iloc[i].items()
                }
                tid = str(frame.TransactionID.iloc[i])
                session.add(
                    Transaction(
                        id=tid,
                        elapsed_seconds=float(frame.TransactionDT.iloc[i]),
                        entity=frame.behavioural_entity_id.iloc[i],
                        context=transactions[i].model_dump(),
                        features=result_features,
                    )
                )
                session.flush()
                session.add(
                    Prediction(
                        transaction_id=tid,
                        risk_score=e.fusion.final_risk_score,
                        risk_level=e.fusion.risk_level,
                        investigation_required=e.fusion.investigation_flag,
                        result=response,
                        explanation=e.model_dump(mode="json"),
                    )
                )
                session.add(Investigation(transaction_id=tid, state="UNREVIEWED"))
                session.add(
                    AuditEvent(
                        transaction_id=tid,
                        action="prediction_generated",
                        safe_metadata={"model_version": self.system.classifier.model_version},
                    )
                )
                results.append(response)
            session.commit()
            self.watermark = max(self.watermark, float(frame.TransactionDT.max()))
            return results

    def listing(
        self,
        page,
        page_size,
        level=None,
        investigation=None,
        sort="recent",
        search=None,
        review_state=None,
    ):
        with Session(self.engine) as s:
            query = (
                select(Transaction, Prediction, Investigation)
                .join(Prediction, Prediction.transaction_id == Transaction.id)
                .join(Investigation, Investigation.transaction_id == Transaction.id)
            )
            if level:
                query = query.where(Prediction.risk_level == level)
            if investigation is not None:
                query = query.where(Prediction.investigation_required == investigation)
            if review_state is not None:
                query = query.where(Investigation.state == review_state)
            if search:
                query = query.where(Transaction.id.contains(search, autoescape=True))
            total = s.scalar(select(func.count()).select_from(query.subquery()))
            ordering = (
                Prediction.risk_score.desc()
                if sort == "risk"
                else Transaction.elapsed_seconds.desc()
            )
            rows = s.execute(
                query.order_by(ordering, Transaction.id)
                .offset((page - 1) * page_size)
                .limit(page_size)
            ).all()
            return {
                "total": total,
                "page": page,
                "page_size": page_size,
                "items": [
                    {"transaction": t.context, "prediction": p.result, "investigation": i.state}
                    for t, p, i in rows
                ],
            }

    def detail(self, tid):
        with Session(self.engine) as s:
            t, p, i = s.get(Transaction, tid), s.get(Prediction, tid), s.get(Investigation, tid)
            if not t:
                raise DatasetError("transaction_not_found", "Transaction not found", 404)
            return {"transaction": t.context, "prediction": p.result, "investigation": i.state}

    def explanation(self, tid):
        with self.lock, Session(self.engine) as s:
            t, p = s.get(Transaction, tid), s.get(Prediction, tid)
            if not t:
                raise DatasetError("transaction_not_found", "Transaction not found", 404)
            if not p.explanation:
                features = pd.DataFrame([t.features], columns=self.system.preprocessor.columns)
                e = self.system.explain(pd.Series(t.context), features)
                if abs(e.fusion.final_risk_score - p.risk_score) > 1e-12:
                    raise DatasetError(
                        "prediction_inconsistent", "Stored explanation does not match prediction"
                    )
                p.explanation = e.model_dump(mode="json")
                p.result = {**p.result, "explanation": response_from_explanation(e)["explanation"]}
                s.commit()
            return p.explanation

    def investigate(self, tid, state):
        with self.lock, Session(self.engine) as s:
            i = s.get(Investigation, tid)
            if not i:
                raise DatasetError("transaction_not_found", "Transaction not found", 404)
            previous = i.state
            if previous != state:
                i.state = state
                i.updated_at = now()
                s.add(
                    AuditEvent(
                        transaction_id=tid,
                        action="investigation_opened"
                        if state == "UNDER_REVIEW"
                        else "investigation_status_changed",
                        safe_metadata={"previous": previous, "state": state},
                    )
                )
                s.commit()
            return {"transaction_id": tid, "state": state, "scores_unchanged": True}

    def dashboard(self):
        with Session(self.engine) as s:
            count = s.scalar(select(func.count(Transaction.id)))
            levels = dict(
                s.execute(
                    select(Prediction.risk_level, func.count()).group_by(Prediction.risk_level)
                ).all()
            )
            daily = s.execute(
                select(
                    (Transaction.elapsed_seconds / 86400)
                    .cast(__import__("sqlalchemy").Integer)
                    .label("day"),
                    func.count(),
                    func.avg(Prediction.risk_score),
                )
                .join(Prediction, Prediction.transaction_id == Transaction.id)
                .group_by("day")
                .order_by("day")
            ).all()
            return {
                "transaction_count": count,
                "risk_level_distribution": levels,
                "investigation_count": s.scalar(
                    select(func.count())
                    .select_from(Prediction)
                    .where(Prediction.investigation_required)
                ),
                "average_risk_score": s.scalar(select(func.avg(Prediction.risk_score))),
                "risk_trend": [
                    {"elapsed_day_index": d, "transactions": n, "average_risk_score": v}
                    for d, n, v in daily
                ],
                "recent_transaction_activity": self.listing(1, 10)["items"],
                "recent_high_risk_transactions": self.listing(1, 10, investigation=True)["items"],
                "scope": "Locked holdout demo plus subsequent submitted synthetic transactions",
            }

    def model_info(self):
        with Session(self.engine) as session:
            watermark = float(
                session.scalar(select(func.max(Transaction.elapsed_seconds))) or self.watermark
            )
        return {
            "dataset_type": "synthetic",
            "notice": NOTICE,
            "supervised_model": "Decision Tree",
            "supervised_model_id": self.system.classifier.model_id,
            "supervised_calibration": "sigmoid",
            "development_fingerprint": self.system.provenance.get("dataset_fingerprint"),
            "final_holdout_fingerprint": self.manifest.get("holdout_fingerprint"),
            "model_version": self.system.classifier.model_version,
            "feature_version": self.system.provenance["feature_version"],
            "anomaly_version": "anomaly-detection-v1",
            "behaviour_version": "behavioural-deviation-v1",
            "abrf_version": "abrf-v1",
            "explainability_version": "explainability-v1.1-shap",
            "shap_version": "0.53.0",
            "system_status": "READY",
            "minimum_elapsed_seconds": watermark,
            "final_holdout_status": self.manifest["status"],
        }
