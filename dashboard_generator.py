import requests
import re
from typing import List, Dict, Tuple, Optional, Any, Set, Callable

class DashboardGenerator:
    def __init__(
        self,
        metrics_endpoint: str,
        dashboard_name: str = "Microservice Metrics Dashboard",
        quantiles: Optional[List[float]] = None,
        max_metrics: int = 100,
        columns_per_row: int = 2,
        custom_group_patterns: Optional[List[Tuple[str, Callable, str, str, Callable]]] = None,
        panel_overrides: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        self.metrics_endpoint = metrics_endpoint.rstrip("/")
        self.dashboard_name = dashboard_name
        self.quantiles = quantiles if quantiles else [0.5, 0.9, 0.99]
        self.max_metrics = max_metrics
        self.columns_per_row = columns_per_row
        self.panel_overrides = panel_overrides or {}  # <-- store overrides

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

    def fetch_raw_metrics(self) -> List[Tuple[str, Tuple[str, ...], Optional[str]]]:
        try:
            response = requests.get(self.metrics_endpoint)
            response.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Failed to fetch metrics from {self.metrics_endpoint}: {e}")
        lines = response.text.splitlines()
        metrics = {}
        help_map = {}
        for line in lines:
            if line.startswith("# HELP"):
                match = re.match(r"# HELP ([a-zA-Z_:][a-zA-Z0-9_:]*) (.+)", line)
                if match:
                    help_map[match.group(1)] = match.group(2)
                continue
            if line.startswith("# TYPE"):
                continue
            if not line or line.startswith("#"):
                continue
            match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]+\})?", line)
            if match:
                name = match.group(1)
                label_block = match.group(2)
                labels = ()
                if label_block:
                    labels = tuple(re.findall(r'(\w+)=', label_block))
                metrics[(name, labels)] = help_map.get(name)
        return sorted([(name, labels, help_text) for (name, labels), help_text in metrics.items()])

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
        if metric_service_label:
            metric, label = next(iter(metric_service_label.items()))
            return label, metric
        return "service", "up"

    # --- PromQL Group Panel Functions ---
    def promql_http_status(self, metric, labels, service_label):
        selector = f'{{{service_label}=~"${{service}}"}}' if service_label else ""
        by_clause = f"by ({service_label}, status)" if service_label else "by (status)"
        return f"sum(rate({metric}{selector}[1m])) {by_clause}"

    def promql_http_error_rate(self, metric, labels, service_label):
        selector = f',{service_label}=~"${{service}}"' if service_label else ""
        by_clause = f"by ({service_label})" if service_label else ""
        return (
            f"sum(rate(http_requests_total{{status=~\"5..\"{selector}}}[1m])) {by_clause} "
            f"/ sum(rate(http_requests_total{{{service_label}=~\"${{service}}\"}}[1m])) {by_clause}" if service_label else
            "sum(rate(http_requests_total{status=~\"5..\"}[1m])) / sum(rate(http_requests_total[1m]))"
        )


    def promql_http_success_rate(self, metric, labels, service_label):
        selector = f',{service_label}=~\"${{service}}"' if service_label else ""
        by_clause = f"by ({service_label})" if service_label else ""
        return (
            f"sum(rate(http_requests_total{{status=~\"2..\"{selector}}}[1m])) {by_clause} "
            f"/ sum(rate(http_requests_total{{{service_label}=~\"${{service}}\"}}[1m])) {by_clause}" if service_label else
            "sum(rate(http_requests_total{status=~\"2..\"}[1m])) / sum(rate(http_requests_total[1m]))"
        )

    def promql_latency_quantiles(self, metric, labels, service_label):
        quantile_vars = "$quantile" if len(self.quantiles) > 1 else str(self.quantiles[0])
        selector = f'{{{service_label}=~"${{service}}"}}' if service_label else ""
        by_clause = f"by ({service_label}, le)" if service_label else "by (le)"
        return f"histogram_quantile({quantile_vars}, sum(rate({metric}{selector}[5m])) {by_clause})"

    def promql_error_rate_ratio(self, metric, labels, service_label):
        selector = f'{{{service_label}=~"${{service}}"}}' if service_label else ""
        by_clause = f"by ({service_label})" if service_label else ""
        return (
            f"sum(rate({metric}{selector}[1m])) {by_clause} / sum(rate(http_requests_total{selector}[1m])) {by_clause}"
            if service_label else
            f"sum(rate({metric}[1m])) / sum(rate(http_requests_total[1m]))"
        )


    def classify_row(self, name: str) -> str:
        name_lower = name.lower()
        for row, patterns in self.ROW_CATEGORIES:
            if any(p in name_lower for p in patterns):
                return row
        return "General"

    def classify_visual(self, name: str) -> str:
        name_lower = name.lower()
        if any(x in name_lower for x in self.HISTOGRAMS):
            return "heatmap"
        if any(k in name_lower for k in self.ERRORS):
            return "stat"
        if any(k in name_lower for k in self.LATENCY) or re.search(r"_latency(_ms|_seconds)?$", name_lower):
            return "timeseries"
        if any(k in name_lower for k in self.THROUGHPUT):
            return "barchart"
        if any(k in name_lower for k in self.RATIOS):
            return "gauge"
        if any(k in name_lower for k in self.RESOURCES):
            return "stat"
        if any(k in name_lower for k in self.HTTP):
            if any(s in name_lower for s in ["_duration", "_latency", "_time"]):
                return "timeseries"
            if "status" in name_lower or "code" in name_lower:
                return "barchart"
            if "request" in name_lower or "response" in name_lower:
                return "stat"
            return "timeseries"
        if any(k in name_lower for k in self.INFRA):
            return "stat"
        if any(k in name_lower for k in self.CACHE):
            return "stat"
        if any(k in name_lower for k in self.DB):
            return "timeseries"
        if any(k in name_lower for k in self.KUBE):
            return "stat"
        if any(k in name_lower for k in self.BROKER):
            if "lag" in name_lower or "offset" in name_lower:
                return "barchart"
            return "stat"
        if any(k in name_lower for k in self.DOTNET):
            return "stat"
        if any(k in name_lower for k in self.JAVA):
            return "stat"
        if any(k in name_lower for k in self.REDIS):
            return "stat"
        return "timeseries"

    def create_legend(self, labels: Tuple[str, ...], service_label: Optional[str]) -> str:
        """
        Build a Grafana legendFormat. Use actual label name, e.g. service={{service}}
        """
        def wrap(label: str) -> str:
            return f"{label}={{{{{{label}}}}}}"

        if service_label and service_label in labels:
            return wrap(service_label)
        if labels:
            # Prefer a canonical label if present
            for candidate in ["application", "microservice", "service", "app", "job"]:
                if candidate in labels:
                    return wrap(candidate)
            return ", ".join(wrap(l) for l in labels)
        return ""

    def classify_unit(self, name: str) -> str:
        name_lower = name.lower()
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

    def promql_for_metric(self, metric: str, labels: Tuple[str, ...], service_label: Optional[str]) -> str:
        name_lower = metric.lower()
        label_selector = f'{{{service_label}=~"${{service}}"}}' if service_label else ""
        # Panel override
        if metric.endswith("_bucket"):
            quantile_vars = "$quantile" if len(self.quantiles) > 1 else str(self.quantiles[0])
            if service_label:
                return (
                    f'histogram_quantile('
                    f'{quantile_vars}, sum(rate({metric}{{{service_label}=~"${{service}}"}}[5m])) '
                    f'by ({service_label}, le))'
                )
            return f'histogram_quantile({quantile_vars}, sum(rate({metric}[5m])) by (le))'
        # HTTP status split
        if "http_requests_total" in metric.lower() and ("status" in labels or "code" in labels):
            if service_label:
                return f"sum(rate({metric}{{{service_label}=~\"${{service}}\"}}[1m])) by ({service_label}, status)"
            else:
                return f"sum(rate({metric}[1m])) by (status)"
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
        unit = self.classify_unit(metric)
        row = self.classify_row(metric)
        title = self.prettify_title(metric)
        legend = self.create_legend(labels, service_label)
        expr = self.promql_for_metric(metric, labels, service_label)

        panel: Dict[str, Any] = {
            "id": panel_id,
            "type": visual_type,
            "title": title,
            "description": help_text or "",
            "gridPos": {"x": x, "y": y, "w": width, "h": 8},
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

        # Apply overrides: exact metric name > panel type
        override_src = self.panel_overrides.get(metric) or self.panel_overrides.get(visual_type)
        if override_src:
            # Shallow merge (simple); deep merge only for dict children
            for k, v in override_src.items():
                if isinstance(v, dict) and isinstance(panel.get(k), dict):
                    panel[k].update(v)
                else:
                    panel[k] = v

        return panel, row

    def generate_dashboard(self) -> Dict:
        try:
            metrics = self.fetch_raw_metrics()
        except Exception as e:
            return {
                "title": self.dashboard_name,
                "description": f"Error: {e}",
                "panels": [],
                "templating": {"list": []}
            }

        metric_service_label = self.extract_service_labels(metrics)
        # Pick the first found label for the dashboard variable
        service_label, service_metric = self.get_first_service_label_and_metric(metric_service_label)

        dashboard = {
            "title": self.dashboard_name,
            "description": (
                f"Auto-generated dashboard for endpoint: {self.metrics_endpoint}\n"
                f"Includes up to {self.max_metrics} metrics. "
                f"Histogram quantiles: {', '.join(str(q) for q in self.quantiles)}"
            ),
            "schemaVersion": 36,
            "version": 1,
            "refresh": "30s",
            "time": {"from": "now-6h", "to": "now"},
            "panels": [],
            "templating": {
                "list": [
                    {
                        "name": "quantile",
                        "type": "custom",
                        "label": "Quantile",
                        "hide": 0,
                        "options": [{"text": str(q), "value": str(q), "selected": i == 0} for i, q in enumerate(self.quantiles)],
                        "query": ",".join(str(q) for q in self.quantiles),
                        "current": {"text": str(self.quantiles[0]), "value": str(self.quantiles[0])},
                        "includeAll": False,
                        "multi": False
                    },
                    {
                        "name": "service",
                        "type": "query",
                        "datasource": "Prometheus",
                        "label": service_label.capitalize(),
                        "hide": 0,
                        "query": f'label_values({service_metric}, {service_label})',
                        "current": {"text": "All", "value": "$__all"},
                        "includeAll": True,
                        "multi": False
                    }
                ]
            }
        }

        row_map: Dict[str, List[Tuple[str, Tuple[str, ...], Optional[str]]]] = {}
        panel_id = 1
        panel_width = int(24 / max(1, self.columns_per_row))

        for metric, labels, help_text in metrics[:self.max_metrics]:
            row_key = self.classify_row(metric)
            if row_key not in row_map:
                row_map[row_key] = []
            row_map[row_key].append((metric, labels, help_text))

        y_offset = 0
        for row_title, metric_entries in row_map.items():
            dashboard["panels"].append({
                "type": "row",
                "title": row_title,
                "gridPos": {"x": 0, "y": y_offset, "w": 24, "h": 1},
                "collapsed": False,
                "panels": []
            })
            y_offset += 1
            for i, (metric, labels, help_text) in enumerate(metric_entries):
                slabel = metric_service_label.get(metric, service_label)
                x = (i % self.columns_per_row) * panel_width
                y = y_offset + (i // self.columns_per_row) * 8
                panel, _ = self.create_panel(metric, labels, help_text, panel_id, x, y, slabel, panel_width)
                dashboard["panels"].append(panel)
                panel_id += 1
            y_offset = y + 8 if metric_entries else y_offset

        return dashboard