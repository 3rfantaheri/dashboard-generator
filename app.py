import os
import json
import argparse
from dashboard_generator import DashboardGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate Grafana dashboard from Prometheus metrics endpoint.")
    parser.add_argument("--metrics_endpoint", required=True, help="Prometheus metrics endpoint URL (e.g. http://localhost:8080/metrics)")
    parser.add_argument("--dashboard-name", default="Microservice Metrics Dashboard", help="Dashboard name/title")
    parser.add_argument("--output", default="output/dashboard.json", help="Output dashboard JSON path")
    parser.add_argument("--quantiles", default="0.5,0.9,0.99", help="Histogram quantiles (comma-separated)")
    parser.add_argument("--max-metrics", type=int, default=10000, help="Maximum number of metrics to include")
    parser.add_argument("--columns-per-row", type=int, default=2, help="Number of panels per row")
    parser.add_argument("--panel-overrides", default=None, help="Path to JSON file with panel overrides (optional)")
    parser.add_argument("--timeout", type=float, default=5.0, help="HTTP timeout seconds")
    parser.add_argument("--retries", type=int, default=3, help="Retry attempts for fetching metrics")
    parser.add_argument("--retry-backoff", type=float, default=0.5, help="Backoff multiplier between retries")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--service-label-override", default=None, help="Force a specific service label (e.g. app)")
    parser.add_argument("--categories-config", default=None, help="Path to JSON category config")
    parser.add_argument("--no-service-var", action="store_true", help="Disable service variable templating")
    parser.add_argument("--no-quantiles", action="store_true", help="Disable quantile variable even if histograms exist")
    parser.add_argument("--legend-max-len", type=int, default=60, help="Max legend length before truncation")
    parser.add_argument("--no-override-validation", action="store_true", help="Skip validation of panel overrides")
    parser.add_argument("--raise-on-error", action="store_true", help="Raise exceptions instead of embedding error in dashboard JSON")
    args = parser.parse_args()

    quantiles = [float(q) for q in args.quantiles.split(",") if q.strip()]

    panel_overrides = {}
    if args.panel_overrides:
        with open(args.panel_overrides, "r") as f:
            panel_overrides = json.load(f)

    generator = DashboardGenerator(
        metrics_endpoint=args.metrics_endpoint,
        dashboard_name=args.dashboard_name,
        quantiles=quantiles,
        max_metrics=args.max_metrics,
        columns_per_row=args.columns_per_row,
        panel_overrides=panel_overrides,
        timeout=args.timeout,
        retries=args.retries,
        retry_backoff=args.retry_backoff,
        verbose=args.verbose,
        service_label_override=args.service_label_override,
        categories_config=args.categories_config,
        validate_overrides=not args.no_override_validation,
        legend_max_len=args.legend_max_len,
        raise_on_error=args.raise_on_error,
        enable_service_var=not args.no_service_var,
        enable_quantile_var=not args.no_quantiles
    )
    dashboard = generator.generate_dashboard()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(dashboard, f, indent=2)

    print(f"Dashboard successfully generated at {args.output}")

if __name__ == "__main__":
    main()