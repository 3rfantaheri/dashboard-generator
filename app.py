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
        panel_overrides=panel_overrides
    )
    dashboard = generator.generate_dashboard()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(dashboard, f, indent=2)

    print(f"Dashboard successfully generated at {args.output}")

if __name__ == "__main__":
    main()