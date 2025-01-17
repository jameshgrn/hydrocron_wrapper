Yes, you can query an API endpoint with DuckDB. DuckDB provides functionality to interact with HTTP APIs and process the returned data[1][3]. Here's how you can do it:

1. Use the `httpfs` extension to fetch data from an HTTP endpoint.
2. Parse the JSON response using `read_json_auto` function.

For example, you can query an API and create a table from the results with a single SQL statement[3]:

```sql
CREATE TABLE api_data AS SELECT * FROM read_json_auto('https://api-endpoint.com/data')
```

For APIs requiring authentication or custom headers, DuckDB 1.1 introduced support for custom HTTP headers[3]. You can create an `http` secret to include authentication tokens or other required headers in your API requests.

Additionally, DuckDB can be extended to act as an HTTP API server itself, allowing you to query your DuckDB instance through HTTP requests[1]. This functionality is provided by the `httpserver` extension, which can be installed and used to start an HTTP server for your DuckDB instance.

Citations:
[1] https://github.com/quackscience/duckdb-extension-httpserver
[2] https://www.cloudquery.io/blog/exploring-api-data-with-duckdb
[3] https://motherduck.com/blog/duckdb-110-hidden-gems/
[4] https://duckdb.org/docs/api/python/overview.html
[5] https://duckdb.org/docs/api/c/query.html