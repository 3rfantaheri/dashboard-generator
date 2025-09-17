import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
try:
    from prometheus_client.parser import text_string_to_metric_families  # optional
except ImportError:  # pragma: no cover
    text_string_to_metric_families = None

class DashboardGenerator:
    """Generates a Grafana dashboard based on a Prometheus metrics text endpoint."""
    def __init__(
        self,
        metrics_endpoint: str,
        dashboard_name: str = "Microservice Metrics Dashboard",
        quantiles: Optional[List[float]] = None,
        max_metrics: int = 100,
        columns_per_row: int = 2,
        custom_group_patterns: Optional[List[Tuple[str, Callable, str, str, Callable]]] = None,
        panel_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        timeout: float = 5.0,
        retries: int = 3,
        retry_backoff: float = 0.5,
        verbose: bool = False,
        service_label_override: Optional[str] = None,
        categories_config: Optional[str] = None,
        validate_overrides: bool = True,
        legend_max_len: int = 60,
        raise_on_error: bool = False,
        enable_service_var: bool = True,
        enable_quantile_var: bool = True
    ):
        self.metrics_endpoint = metrics_endpoint.rstrip("/")
        self.dashboard_name = dashboard_name
        self.quantiles = quantiles if quantiles else [0.5, 0.9, 0.99]
        self.max_metrics = max_metrics
        self.columns_per_row = columns_per_row
        self.panel_overrides = panel_overrides or {}  # <-- store overrides
        self.timeout = timeout
        self.retries = retries
        self.retry_backoff = retry_backoff
        self.verbose = verbose
        self.service_label_override = service_label_override
        self.categories_config = categories_config
        self.validate_overrides = validate_overrides
        self.legend_max_len = legend_max_len
        self.raise_on_error = raise_on_error
        self.enable_service_var = enable_service_var
        self.enable_quantile_var = enable_quantile_var
        # self._classification_cache: Dict[str, Dict[str, Any]] = {}  # removed unused cache
        self._metric_base_pairs: Dict[str, Dict[str, bool]] = {}   # base -> {'sum':bool,'count':bool}
        self.has_histogram = False
        self.PANEL_HEIGHT = 8  # constant panel height used across layout
        self.logger = logging.getLogger("DashboardGenerator")
        if verbose and not self.logger.handlers:
            logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")

        self.SERVICE_LABEL_CANDIDATES = ["service", "application", "app", "microservice", "job"]

        self.HISTOGRAMS = ["_bucket", "_count", "_sum", "_quantile", "_histogram"]
        self.ERRORS = [
            "error", "errors", "fail", "fails", "failure", "failures", "panic", "exception", "exceptions",
            "rejected", "denied", "abort", "invalid", "timeout", "timeouts", "unavailable", "critical", "fatal"
        ]
        self.LATENCY = [
            "latency", "latency_ms", "latency_seconds", "duration", "elapsed", "wait", "delay", "time",
            "response_time", "service_time", "request_time", "processing_time", "execution_time"
        ]
        self.THROUGHPUT = [
            "bytes", "traffic", "bandwidth", "throughput", "size", "capacity", "ops", "operations",
            "records", "messages", "events", "produced", "consumed", "written", "read"
        ]
        self.RATIOS = [
            "ratio", "percent", "utilization", "usage", "load", "saturation", "efficiency", "availability", "success_rate", "hit_ratio"
        ]
        self.RESOURCES = [
            "go_", "python", "process_", "thread", "threads", "gc", "heap", "cpu", "memory", "rss", "fds", "runtime",
            "uptime", "restart", "restart_count", "open_files", "system_", "os_", "jvm_", "clr_", "dotnet_", "jre_", "gc_", "cpu_", "mem_", "disk_", "swap_"
        ]
        self.HTTP = [
            "http_", "http_server_", "http_client_", "rest_", "api_", "request", "requests", "response", "responses",
            "status", "status_code", "code", "method", "verb", "path", "route", "endpoint", "uri", "url",
            "network", "connect", "disconnect", "socket", "connections"
        ]
        self.INFRA = [
            "connection", "connections", "queue", "queues", "session", "sessions", "pool", "replica", "replicas", "shard",
            "partition", "node", "nodes", "cluster", "leader", "follower", "master", "replication", "instance", "broker"
        ]
        self.CACHE = [
            "cache", "hit", "hits", "miss", "misses", "evict", "lookup", "insert", "remove", "redis_", "memcached_"
        ]
        self.DB = [
            "db_", "database", "sql", "query", "queries", "storage", "disk", "table", "tables", "row", "rows", "column",
            "columns", "index", "commit", "rollback", "jdbc_", "datasource_", "connection_", "pool_", "hibernate_", "orm_"
        ]
        self.KUBE = [
            "kube_", "container", "containers", "pod", "pods", "deployment", "namespace", "job", "cronjob", "daemonset", "statefulset"
        ]
        self.BROKER = [
            "rabbitmq_", "amqp_", "kafka_", "broker_", "exchange_", "queue_", "consumer_", "producer_", "partition_", "topic_", "offset_", "lag_"
        ]
        self.DOTNET = [
            "dotnet_", "clr_", "aspnet_", "gc_", "threadpool_", "exceptions_", "assemblies_", "contentions_", "allocations_", "jitted_"
        ]
        self.JAVA = [
            "jvm_", "java_", "jre_", "gc_", "memory_", "threads_", "classes_", "buffer_", "pool_", "collections_", "loaded_", "unloaded_"
        ]
        self.REDIS = [
            "redis_", "rdb_", "aof_", "command_", "keyspace_", "expired_", "evicted_", "connected_clients", "blocked_clients"
        ]

        self.ROW_CATEGORIES = [
            ("HTTP & API", self.HTTP),
            ("Errors & Failures", self.ERRORS),
            ("Timing & Latency", self.LATENCY),
            ("Throughput & Bandwidth", self.THROUGHPUT),
            ("Ratios & Utilization", self.RATIOS),
            ("Runtime & Resources", self.RESOURCES),
            ("Infrastructure", self.INFRA),
            ("Cache & Redis", self.CACHE + self.REDIS),
            ("Database", self.DB),
            ("Kubernetes & Containers", self.KUBE),
            ("Message Broker", self.BROKER),
            (".NET Core & CLR", self.DOTNET),
            ("Java/JVM/JRE", self.JAVA),
            ("Performance & Latency", self.HISTOGRAMS),
            ("General", []),
        ]

        # --- Custom Group Patterns ---
        default_group_patterns = [
            ("http_status", lambda n, l: "http_requests_total" in n and ("status" in l or "code" in l),
                "HTTP Status Codes", "barchart", self.promql_http_status),
            ("http_error_rate", lambda n, l: "http_requests_total" in n,
                "HTTP Error Rate", "gauge", self.promql_http_error_rate),
            ("latency_quantiles", lambda n, l: n.endswith("_bucket"),
                "Latency Quantiles", "timeseries", self.promql_latency_quantiles),
            ("error_metrics", lambda n, l: any(e in n for e in self.ERRORS) and "total" in n,
                "Error Rate Ratio", "gauge", self.promql_error_rate_ratio),
            ("success_rate", lambda n, l: "http_requests_total" in n and ("status" in l or "code" in l),
                "HTTP Success Rate", "gauge", self.promql_http_success_rate),
        ]
        self.GROUP_PATTERNS = (custom_group_patterns or []) + default_group_patterns

        if categories_config:
            self._load_category_config(categories_config)

    # --------------- Utility / Support ---------------
    def _validate_url(self):
        parsed = urlparse(self.metrics_endpoint)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

    def _session_get(self) -> str:
        self._validate_url()
        last_err = None
        for attempt in range(1, self.retries + 1):
            try:
                r = requests.get(self.metrics_endpoint, timeout=self.timeout)
                r.raise_for_status()
                return r.text
            except Exception as e:
                last_err = e
                self.logger.debug(f"Fetch attempt {attempt} failed: {e}")
                if attempt < self.retries:
                    time.sleep(self.retry_backoff * attempt)
        raise RuntimeError(f"Failed to fetch metrics after {self.retries} attempts: {last_err}")

    def _load_category_config(self, path: str):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            custom_rows = []
            for row in data.get("rows", []):
                name = row.get("title")
                patterns = row.get("patterns", [])
                if name:
                    custom_rows.append((name, patterns))
            if custom_rows:
                self.ROW_CATEGORIES = custom_rows + [r for r in self.ROW_CATEGORIES if r[0] == "General"]
        except Exception as e:
            self.logger.warning(f"Failed loading categories config {path}: {e}")

    def _deep_merge(self, base: Dict, ext: Dict):
        for k, v in ext.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                self._deep_merge(base[k], v)
            else:
                base[k] = v

    def prettify_title(self, metric: str) -> str:
        title = re.sub(r"(_total|_sum|_count|_bucket)$", "", metric)
        title = title.replace("_", " ").strip().title()
        return title

    # --------------- Metric Parsing ---------------
    def fetch_raw_metrics(self) -> List[Tuple[str, Tuple[str, ...], Optional[str]]]:
        # REWRITTEN: robust fetching + optional official parser + early max limit + histogram detection
        raw_text = self._session_get()
        metrics: Dict[Tuple[str, Tuple[str, ...]], Optional[str]] = {}
        help_map: Dict[str, str] = {}
        added = 0
        if text_string_to_metric_families:
            try:
                for fam in text_string_to_metric_families(raw_text):
                    help_map[fam.name] = fam.documentation
                    for s in fam.samples:
                        name = s.name
                        label_keys = tuple(sorted(s.labels.keys()))
                        if name.endswith("_bucket"):
                            self.has_histogram = True
                        metrics[(name, label_keys)] = help_map.get(name)
                        # Track *_sum/_count for average calculations
                        if name.endswith("_sum") or name.endswith("_count"):
                            base = re.sub(r"_(sum|count)$", "", name)
                            base_entry = self._metric_base_pairs.setdefault(base, {'sum': False, 'count': False})
                            if name.endswith("_sum"):
                                base_entry['sum'] = True
                            if name.endswith("_count"):
                                base_entry['count'] = True
                        added += 1
                        if added >= self.max_metrics:
                            break
                    if added >= self.max_metrics:
                        break
                sorted_metrics = sorted([(n, l, metrics[(n, l)]) for (n, l) in metrics.keys()])
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"Parsed {len(sorted_metrics)} metrics via prom parser (histograms={self.has_histogram})")
                return sorted_metrics
            except Exception as e:
                self.logger.warning(f"Prometheus parser failed, falling back to regex: {e}")

        # Fallback regex parser (original logic truncated early)
        lines = raw_text.splitlines()
        for line in lines:
            if line.startswith("# HELP"):
                m = re.match(r"# HELP ([a-zA-Z_:][a-zA-Z0-9_:]*) (.+)", line)
                if m:
                    help_map[m.group(1)] = m.group(2)
                continue
            if line.startswith("# TYPE") or not line or line.startswith("#"):
                continue
            m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]+\})?", line)
            if not m:
                continue
            name = m.group(1)
            label_block = m.group(2)
            labels: Tuple[str, ...] = ()
            if label_block:
                labels = tuple(sorted(re.findall(r'(\w+)=', label_block)))
            if name.endswith("_bucket"):
                self.has_histogram = True
            if name.endswith("_sum") or name.endswith("_count"):
                base = re.sub(r"_(sum|count)$", "", name)
                base_entry = self._metric_base_pairs.setdefault(base, {'sum': False, 'count': False})
                if name.endswith("_sum"):
                    base_entry['sum'] = True
                else:
                    base_entry['count'] = True
            metrics[(name, labels)] = help_map.get(name)
            added += 1
            if added >= self.max_metrics:
                break
        sorted_metrics = sorted([(n, l, metrics[(n, l)]) for (n, l) in metrics.keys()])
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Parsed {len(sorted_metrics)} metrics via regex (histograms={self.has_histogram})")
        return sorted_metrics

    # --------------- Service Label Handling ---------------
    def extract_service_labels(self, metrics: List[Tuple[str, Tuple[str, ...], Optional[str]]]) -> Dict[str, str]:
        """
        Returns a mapping: metric_name -> service_label (from candidates) if found, else None.
        """
        metric_service_label = {}
        for metric, labels, _ in metrics:
            found = None
            for label in labels:
                if label in self.SERVICE_LABEL_CANDIDATES:
                    found = label
                    break
            if found:
                metric_service_label[metric] = found
        return metric_service_label

    def get_first_service_label_and_metric(self, metric_service_label: Dict[str, str]) -> Tuple[str, str]:
        """
        Returns (service_label, metric) for the first metric that has a service label.
        """
        # OVERRIDE: allow user override
        if self.service_label_override:
            # pick any metric containing that label
            for metric, label in metric_service_label.items():
                if label == self.service_label_override:
                    return label, metric
            # fallback to 'up'
            return self.service_label_override, "up"

        if metric_service_label:
            metric, label = next(iter(metric_service_label.items()))
            return label, metric
        return "service", "up"

    # --------------- Legend / Units / Classification ---------------
    def create_legend(self, labels: Tuple[str, ...], service_label: Optional[str]) -> str:
        """
        Build a Grafana legendFormat. Use actual label name, e.g. service={{service}}
        """
        legend = ""  # ...existing logic (kept)...
        def wrap(label: str) -> str:
            return f"{label}={{{{{{label}}}}}}"
        if service_label and service_label in labels:
            legend = wrap(service_label)
        elif labels:
            for candidate in ["application", "microservice", "service", "app", "job"]:
                if candidate in labels:
                    legend = wrap(candidate)
                    break
            else:
                legend = ", ".join(wrap(l) for l in labels)
        if legend and len(legend) > self.legend_max_len:
            truncated = legend[: self.legend_max_len - 3] + "..."
            # Avoid truncating inside '{{' '}}' placeholder (simple balance check)
            if legend.count("{{") == legend.count("}}"):
                if truncated.count("{{") == truncated.count("}}"):
                    legend = truncated
        return legend

    def classify_unit(self, name: str, expr: str) -> str:
        # Extended with bytes/sec detection
        name_lower = name.lower()
        if "rate(" in expr and ("_bytes" in name_lower or name_lower.endswith("_bytes_total")):
            return "Bps"
        if name_lower.endswith("_seconds") or name_lower.endswith("_duration_seconds"):
            return "s"
        if any(x in name_lower for x in self.HISTOGRAMS):
            # Default to seconds unless explicit _ms
            if name_lower.endswith("_ms"):
                return "ms"
            return "s"
        if any(k in name_lower for k in self.LATENCY):
            return "s"
        if any(k in name_lower for k in self.THROUGHPUT):
            return "bytes"
        if any(k in name_lower for k in self.RATIOS):
            return "percent"
        return "short"

    # --------------- PromQL Helpers ---------------
    def _has_sum_count_pair(self, metric: str) -> Optional[str]:
        # Detect if this metric is either base_sum or base_count to build average from both
        if metric.endswith("_sum") or metric.endswith("_count"):
            base = re.sub(r"_(sum|count)$", "", metric)
            pair = self._metric_base_pairs.get(base)
            if pair and pair['sum'] and pair['count']:
                return base
        return None

    def _avg_from_sum_count(self, base: str, service_label: Optional[str]) -> str:
        sel = f'{{{service_label}=~"${{service}}"}}' if service_label else ""
        sum_m = f"{base}_sum"
        count_m = f"{base}_count"
        return (
            f"sum(rate({sum_m}{sel}[5m])) / sum(rate({count_m}{sel}[5m]))"
            if service_label else
            f"sum(rate({sum_m}[5m])) / sum(rate({count_m}[5m]))"
        )

    def _label_selector(self, service_label: Optional[str]) -> str:
        """Return common label selector snippet or empty string."""
        return f'{{{service_label}=~"${{service}}"}}' if service_label else ""

    def promql_for_metric(self, metric: str, labels: Tuple[str, ...], service_label: Optional[str]) -> str:
        # Modified: use sum/count avg, rest falls back to existing logic
        base = self._has_sum_count_pair(metric)
        if base:
            return self._avg_from_sum_count(base, service_label)
        name_lower = metric.lower()
        label_selector = self._label_selector(service_label)
        if metric.endswith("_bucket"):
            quantile_vars = "$quantile" if len(self.quantiles) > 1 else str(self.quantiles[0])
            if service_label:
                return (f'histogram_quantile('
                        f'{quantile_vars}, sum(rate({metric}{{{service_label}=~"${{service}}"}}[5m])) '
                        f'by ({service_label}, le))')
            return f'histogram_quantile({quantile_vars}, sum(rate({metric}[5m])) by (le))'
        # HTTP status split
        if "http_requests_total" in metric.lower() and ("status" in labels or "code" in labels):
            status_label = self._http_status_label(labels)
            if service_label:
                return f"sum(rate({metric}{{{service_label}=~\"${{service}}\"}}[1m])) by ({service_label}, {status_label})"
            else:
                return f"sum(rate({metric}[1m])) by ({status_label})"
        if "http_requests_total" in name_lower:
            if service_label:
                return f"sum(rate({metric}{label_selector}[1m]))"
            else:
                return f"sum(rate({metric}[1m]))"
        if any(k in name_lower for k in self.ERRORS) and "total" in name_lower:
            if service_label:
                return (
                    f"sum(rate({metric}{label_selector}[1m])) "
                    f"/ sum(rate(http_requests_total{label_selector}[1m]))"
                )
            else:
                return f"sum(rate({metric}[1m])) / sum(rate(http_requests_total[1m]))"
        if any(k in name_lower for k in self.LATENCY):
            if metric.endswith("_seconds") or metric.endswith("_duration_seconds"):
                if service_label:
                    return f"avg_over_time({metric}{label_selector}[5m])"
                else:
                    return f"avg_over_time({metric}[5m])"
            if service_label:
                return f"avg({metric}{label_selector})"
            else:
                return f"avg({metric})"
        if any(k in name_lower for k in self.THROUGHPUT):
            if service_label:
                return f"sum(rate({metric}{label_selector}[1m]))"
            else:
                return f"sum(rate({metric}[1m]))"
        if any(k in name_lower for k in self.RATIOS):
            if service_label:
                return f"avg({metric}{label_selector})"
            else:
                return f"avg({metric})"
        if any(k in name_lower for k in self.RESOURCES):
            if "cpu" in name_lower or "memory" in name_lower or "heap" in name_lower:
                if service_label:
                    return f"max({metric}{label_selector})"
                else:
                    return f"max({metric})"
            if service_label:
                return f"avg({metric}{label_selector})"
            else:
                return f"avg({metric})"
        if any(p in name_lower for p in self.DB):
            if "query" in name_lower or "sql" in name_lower:
                if service_label:
                    return f"sum(rate({metric}{label_selector}[1m]))"
                else:
                    return f"sum(rate({metric}[1m]))"
            if "connection" in name_lower or "pool" in name_lower:
                if service_label:
                    return f"max({metric}{label_selector})"
                else:
                    return f"max({metric})"
            if "commit" in name_lower or "rollback" in name_lower:
                if service_label:
                    return f"sum(rate({metric}{label_selector}[5m]))"
                else:
                    return f"sum(rate({metric}[5m]))"
        if any(p in name_lower for p in self.BROKER):
            if "consumer" in name_lower or "producer" in name_lower:
                if service_label:
                    return f"sum(rate({metric}{label_selector}[1m]))"
                else:
                    return f"sum(rate({metric}[1m]))"
            if "lag" in name_lower or "offset" in name_lower:
                if service_label:
                    return f"max({metric}{label_selector})"
                else:
                    return f"max({metric})"
            if "queue" in name_lower or "exchange" in name_lower or "topic" in name_lower:
                if service_label:
                    return f"sum({metric}{label_selector})"
                else:
                    return f"sum({metric})"
        if any(p in name_lower for p in self.DOTNET):
            if "exceptions" in name_lower:
                if service_label:
                    return f"sum(rate({metric}{label_selector}[5m]))"
                else:
                    return f"sum(rate({metric}[5m]))"
            if "gc" in name_lower or "allocations" in name_lower:
                if service_label:
                    return f"sum({metric}{label_selector})"
                else:
                    return f"sum({metric})"
            if "threadpool" in name_lower:
                if service_label:
                    return f"max({metric}{label_selector})"
                else:
                    return f"max({metric})"
        if any(p in name_lower for p in self.JAVA):
            if "gc" in name_lower or "collections" in name_lower:
                if service_label:
                    return f"sum({metric}{label_selector})"
                else:
                    return f"sum({metric})"
            if "memory" in name_lower or "heap" in name_lower:
                if service_label:
                    return f"max({metric}{label_selector})"
                else:
                    return f"max({metric})"
            if "threads" in name_lower:
                if service_label:
                    return f"max({metric}{label_selector})"
                else:
                    return f"max({metric})"
        if any(p in name_lower for p in self.REDIS):
            if "hit" in name_lower or "miss" in name_lower:
                if service_label:
                    return f"sum(rate({metric}{label_selector}[5m]))"
                else:
                    return f"sum(rate({metric}[5m]))"
            if "connected_clients" in name_lower:
                if service_label:
                    return f"max({metric}{label_selector})"
                else:
                    return f"max({metric})"
        if any(k in name_lower for k in self.CACHE):
            if "hit" in name_lower or "miss" in name_lower:
                if service_label:
                    return f"sum(rate({metric}{label_selector}[5m]))"
                else:
                    return f"sum(rate({metric}[5m]))"
            if service_label:
                return f"max({metric}{label_selector})"
            else:
                return f"max({metric})"
        if any(k in name_lower for k in self.KUBE):
            if service_label:
                return f"max({metric}{label_selector})"
            else:
                return f"max({metric})"
        if any(k in name_lower for k in self.HISTOGRAMS):
            if service_label:
                return f"sum(rate({metric}{label_selector}[5m]))"
            else:
                return f"sum(rate({metric}[5m]))"
        if any(k in name_lower for k in self.ERRORS):
            if service_label:
                return f"sum(rate({metric}{label_selector}[5m]))"
            else:
                return f"sum(rate({metric}[5m]))"
        if any(k in name_lower for k in self.INFRA):
            if service_label:
                return f"sum({metric}{label_selector})"
            else:
                return f"sum({metric})"
        if service_label:
            return f"avg({metric}{label_selector})"
        return f"avg({metric})"

    # --------------- Panel Creation ---------------
    def create_panel(
        self,
        metric: str,
        labels: Tuple[str, ...],
        help_text: Optional[str],
        panel_id: int,
        x: int,
        y: int,
        service_label: Optional[str],
        width: int
    ) -> Tuple[Dict, str]:
        visual_type = self.classify_visual(metric)
        expr = self.promql_for_metric(metric, labels, service_label)
        unit = self.classify_unit(metric, expr)
        row = self.classify_row(metric)
        title = self.prettify_title(metric)
        legend = self.create_legend(labels, service_label)

        panel: Dict[str, Any] = {
            "id": panel_id,
            "type": visual_type,
            "title": title,
            "description": help_text or "",
            "gridPos": {"x": x, "y": y, "w": width, "h": self.PANEL_HEIGHT},
            "fieldConfig": {
                "defaults": {
                    "unit": unit,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "orange", "value": 70},
                            {"color": "red", "value": 90}
                        ]
                    }
                }
            },
            "targets": [
                {
                    "expr": expr,
                    "legendFormat": legend,
                    "refId": "A",
                    "datasource": "Prometheus"
                }
            ]
        }
        # Deep merge overrides
        override_src = self.panel_overrides.get(metric) or self.panel_overrides.get(visual_type)
        if override_src:
            self._deep_merge(panel, override_src)
        return panel, row

    # --------------- Overrides Validation ---------------
    def validate_panel_overrides(self):
        if not self.validate_overrides:
            return
        allowed_top = {"type", "title", "fieldConfig", "targets", "transformations",
                       "options", "gridPos", "datasource", "description", "id"}
        for key, override in self.panel_overrides.items():
            if not isinstance(override, dict):
                self.logger.warning(f"Override for {key} ignored (not a dict)")
                continue
            for k in override.keys():
                if k not in allowed_top:
                    self.logger.warning(f"Override key '{k}' for {key} not standard")

    # --------------- Additional Variables ---------------
    def discover_extra_label_vars(self, metrics: List[Tuple[str, Tuple[str, ...], Optional[str]]], exclude: Set[str]) -> Dict[str, str]:
        """
        Return mapping: label_name -> example_metric_that_has_it
        Ensures we query label_values(example_metric, label) for a metric that actually contains the label.
        """
        candidate_labels = ["namespace", "pod", "instance", "env", "environment", "cluster"]
        freq: Dict[str, int] = {}
        example_metric: Dict[str, str] = {}
        for metric, labels, _ in metrics:
            for l in labels:
                if l in exclude:
                    continue
                if l in candidate_labels:
                    freq[l] = freq.get(l, 0) + 1
                    example_metric.setdefault(l, metric)
        # threshold: appear in at least 5 metrics
        return {l: example_metric[l] for l, c in freq.items() if c >= 5}

    # --------------- PromQL Group Panel Functions / Helpers ---------------
    def _http_status_label(self, labels: Tuple[str, ...]) -> str:
        """Return the HTTP status label key ('status' or 'code'); default 'status' if none found."""
        return "status" if "status" in labels else ("code" if "code" in labels else "status")

    def promql_http_status(self, metric, labels, service_label):
        status_label = self._http_status_label(labels)
        selector = f'{{{service_label}=~"${{service}}"}}' if service_label else ""
        by_clause = f"by ({service_label}, {status_label})" if service_label else f"by ({status_label})"
        return f"sum(rate({metric}{selector}[1m])) {by_clause}"

    def promql_http_error_rate(self, metric, labels, service_label):
        status_label = self._http_status_label(labels)
        # Build selectors dynamically
        base_selector = f',{service_label}=~"${{service}}"' if service_label else ""
        status_filter = f'{status_label}=~"5.."'
        denom_selector = f'{{{service_label}=~"${{service}}"}}' if service_label else ""
        by_clause = f"by ({service_label})" if service_label else ""
        if service_label:
            return (
                f"sum(rate(http_requests_total{{{status_filter}{base_selector}}}[1m])) {by_clause} "
                f"/ sum(rate(http_requests_total{denom_selector}[1m])) {by_clause}"
            )
        return (
            f"sum(rate(http_requests_total{{{status_filter}}}[1m])) "
            f"/ sum(rate(http_requests_total[1m]))"
        )

    def promql_http_success_rate(self, metric, labels, service_label):
        status_label = self._http_status_label(labels)
        base_selector = f',{service_label}=~"${{service}}"' if service_label else ""
        status_filter = f'{status_label}=~"2.."'
        denom_selector = f'{{{service_label}=~"${{service}}"}}' if service_label else ""
        by_clause = f"by ({service_label})" if service_label else ""
        if service_label:
            return (
                f"sum(rate(http_requests_total{{{status_filter}{base_selector}}}[1m])) {by_clause} "
                f"/ sum(rate(http_requests_total{denom_selector}[1m])) {by_clause}"
            )
        return (
            f"sum(rate(http_requests_total{{{status_filter}}}[1m])) "
            f"/ sum(rate(http_requests_total[1m]))"
        )

    # --------------- Dashboard Generation ---------------
    def generate_dashboard(self) -> Dict:
        try:
            self.validate_panel_overrides()
            metrics = self.fetch_raw_metrics()
        except Exception as e:
            err = {"title": self.dashboard_name, "description": f"Error: {e}", "panels": [], "templating": {"list": []}, "error": str(e)}
            if self.raise_on_error:
                raise
            return err

        metric_service_label = self.extract_service_labels(metrics)
        service_label, service_metric = self.get_first_service_label_and_metric(metric_service_label)
        if not self.enable_service_var:
            service_label = None
        # Disable service variable if chosen metric does not actually expose that label
        if service_label and (service_metric not in metric_service_label or metric_service_label[service_metric] != service_label):
            self.logger.debug("Disabling service variable: selected metric lacks the service label.")
            service_label = None

        templating_vars = []

        # Quantile variable only if histograms exist and enabled
        if self.has_histogram and self.enable_quantile_var and self.quantiles:
            templating_vars.append({
                "name": "quantile",
                "type": "custom",
                "label": "Quantile",
                "hide": 0,
                "options": [{"text": str(q), "value": str(q), "selected": i == 0} for i, q in enumerate(self.quantiles)],
                "query": ",".join(str(q) for q in self.quantiles),
                "current": {"text": str(self.quantiles[0]), "value": str(self.quantiles[0])},
                "includeAll": False,
                "multi": False
            })

        if service_label and self.enable_service_var:
            templating_vars.append({
                "name": "service",
                "type": "query",
                "datasource": "Prometheus",
                "label": service_label.capitalize(),
                "hide": 0,
                "query": f'label_values({service_metric}, {service_label})',
                "current": {"text": "All", "value": "$__all"},
                "includeAll": True,
                "multi": False
            })

        extra_vars_map = self.discover_extra_label_vars(metrics, exclude={service_label} if service_label else set())
        for lbl, ex_metric in extra_vars_map.items():
            # skip if same as service label already added
            if lbl == service_label:
                continue
            templating_vars.append({
                "name": lbl,
                "type": "query",
                "datasource": "Prometheus",
                "label": lbl.capitalize(),
                "hide": 0,
                "query": f'label_values({ex_metric}, {lbl})',
                "current": {"text": "All", "value": "$__all"},
                "includeAll": True,
                "multi": False
            })

        dashboard = {
            "title": self.dashboard_name,
            "description": (
                f"Auto-generated dashboard for endpoint: {self.metrics_endpoint}\n"
                f"Includes up to {self.max_metrics} metrics. "
                f"{'Histogram quantiles: ' + ', '.join(map(str, self.quantiles)) if self.has_histogram else 'No histograms detected.'}"
            ),
            "schemaVersion": 36,
            "version": 1,
            "refresh": "30s",
            "time": {"from": "now-6h", "to": "now"},
            "panels": [],
            "templating": {"list": templating_vars}
        }

        # Build row grouping
        row_map: Dict[str, List[Tuple[str, Tuple[str, ...], Optional[str]]]] = {}
        for metric, labels, help_text in metrics:
            row_key = self.classify_row(metric)
            row_map.setdefault(row_key, []).append((metric, labels, help_text))

        panel_id = 1
        y_cursor = 0
        panel_width = int(24 / max(1, self.columns_per_row))

        for row_title, metric_entries in row_map.items():
            # Row header panel
            dashboard["panels"].append({
                "type": "row",
                "title": row_title,
                "gridPos": {"x": 0, "y": y_cursor, "w": 24, "h": 1},
                "collapsed": False,
                "panels": []
            })
            y_cursor += 1
            for idx, (metric, labels, help_text) in enumerate(metric_entries):
                slabel = metric_service_label.get(metric, service_label)
                x = (idx % self.columns_per_row) * panel_width
                local_y = y_cursor + (idx // self.columns_per_row) * self.PANEL_HEIGHT
                panel, _ = self.create_panel(metric, labels, help_text, panel_id, x, local_y, slabel, panel_width)
                dashboard["panels"].append(panel)
                panel_id += 1
            # Advance cursor after all panels in this row group
            rows_consumed = (len(metric_entries) + self.columns_per_row - 1) // self.columns_per_row
            y_cursor += rows_consumed * self.PANEL_HEIGHT

        return dashboard