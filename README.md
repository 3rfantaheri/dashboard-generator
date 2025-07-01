# Dashboard Generator

Automatically generate a Grafana dashboard JSON from a Prometheus metrics endpoint.

## Features

- **Automatic metric discovery**: Fetches and parses all metrics from a Prometheus endpoint.
- **classification**: Categorizes metrics into rows by topic (HTTP, Database, Message Broker, .NET, Java, Redis, etc.).
- **Pattern-based visualization**: Assigns the best visualization type (timeseries, stat, gauge, heatmap, barchart) based on metric patterns.
- **Parametric dashboard**: Dashboard name, quantiles, and max metrics are configurable via command-line arguments.
- **Descriptive output**: Dashboard visuals include a description.
- **Ready-to-import**: Outputs a JSON file compatible with Grafana's import feature.


## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py --metrics_endpoint http://localhost:8080/metrics \
               --dashboard-name "My Service Dashboard" \
               --output output/dashboard.json \
               --quantiles 0.5,0.9,0.99 \
```

- `--metrics_endpoint`: Prometheus metrics endpoint URL (required)
- `--dashboard-name`: Dashboard name/title (default: "Microservice Metrics Dashboard")
- `--output`: Output dashboard JSON path (default: `output/dashboard.json`)
- `--quantiles`: Histogram quantiles as comma-separated values (default: `0.5,0.9,0.99`)
- `--max-metrics`: Maximum number of metrics to include (default: `10000`)

## Metric Classification Logic

- **HTTP & API**: Request/response count, latency, status code, etc.
- **Errors & Failures**: Error, fail, panic, exception, timeout, etc.
- **Timing & Latency**: Latency, duration, elapsed, wait, etc.
- **Throughput & Bandwidth**: Bytes, throughput, ops, messages, etc.
- **Ratios & Utilization**: Percent, utilization, load, efficiency, etc.
- **Runtime & Resources**: CPU, memory, heap, threads, GC, etc.
- **Infrastructure**: Connection, queue, session, cluster, etc.
- **Cache & Redis**: Cache, hit, miss, evict, redis, etc.
- **Database**: DB, SQL, query, connection, pool, etc.
- **Kubernetes & Containers**: Pod, container, deployment, etc.
- **Message Broker**: RabbitMQ, Kafka, queue, topic, lag, etc.
- **.NET Core & CLR**: dotnet, clr, aspnet, gc, threadpool, etc.
- **Java/JVM/JRE**: jvm, java, gc, memory, threads, etc.
- **Performance & Latency**: _bucket, _sum, _count, _quantile, etc.
- **General**: Any metric not matching above patterns.

## Notes

- Import the resulting JSON into Grafana using the "Import Dashboard".
- Histogram panels use the `$quantile` variable for interactive quantile selection.
- The generator is extensible: add more patterns or row categories as needed.

---