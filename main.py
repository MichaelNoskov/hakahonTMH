import argparse
import json
import sys
from pathlib import Path

import polars as pl

DISPLACEMENT_PATH = "locomotives_displacement.csv"
STATION_INFO_PATH = "station_info.csv"

DISPLACEMENT_REQUIRED = ["locomotive_series", "locomotive_number", "datetime", "station", "depo_station"]
STATION_INFO_REQUIRED = ["station", "station_name", "latitude", "longitude"]


class ValidationError(Exception):
    pass


def _validate_columns(df: pl.DataFrame, required: list[str], source: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValidationError(
            f"{source}: отсутствуют обязательные поля: {missing}. "
            f"Требуются: {required}. Найдены: {list(df.columns)}"
        )


def _load_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, try_parse_dates=True)


def _load_json(path: Path) -> pl.DataFrame:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return pl.DataFrame(data)
    if isinstance(data, dict) and "data" in data:
        return pl.DataFrame(data["data"])
    if isinstance(data, dict) and "records" in data:
        return pl.DataFrame(data["records"])
    raise ValidationError(f"JSON должен быть массивом объектов или dict с ключом 'data'/'records'")


def load_displacement(path: str | Path | None = None) -> pl.DataFrame:
    p = Path(path) if path else Path(DISPLACEMENT_PATH)
    if not p.exists():
        raise FileNotFoundError(f"Файл не найден: {p}")

    suffix = p.suffix.lower()
    if suffix == ".csv":
        df = _load_csv(p)
    elif suffix == ".json":
        df = _load_json(p)
    else:
        raise ValidationError(f"Поддерживаются только .csv и .json, получено: {suffix}")

    _validate_columns(df, DISPLACEMENT_REQUIRED, str(p))

    return (
        df.with_columns([
            pl.col("locomotive_number").cast(pl.Int32, strict=False),
            pl.col("station").cast(pl.Int32, strict=False),
            pl.col("depo_station").cast(pl.Int32, strict=False),
        ])
        .filter(
            pl.col("locomotive_number").is_not_null()
            & pl.col("station").is_not_null()
            & pl.col("depo_station").is_not_null()
        )
        .sort(["locomotive_series", "locomotive_number", "datetime"])
    )


def load_station_info(path: str | Path | None = None) -> pl.DataFrame:
    p = Path(path) if path else Path(STATION_INFO_PATH)
    if not p.exists():
        raise FileNotFoundError(f"Файл не найден: {p}")

    suffix = p.suffix.lower()
    if suffix == ".csv":
        df = pl.read_csv(p)
    elif suffix == ".json":
        df = _load_json(p)
    else:
        raise ValidationError(f"Поддерживаются только .csv и .json, получено: {suffix}")

    _validate_columns(df, STATION_INFO_REQUIRED, str(p))

    return (
        df.with_columns([
            pl.col("station").cast(pl.Int32, strict=False),
            pl.col("latitude").cast(pl.Float64, strict=False),
            pl.col("longitude").cast(pl.Float64, strict=False),
        ])
        .filter(pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null())
    )


def build_trips(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        (pl.col("station") == pl.col("depo_station")).cast(pl.Int8).alias("is_depo")
    )
    return df.with_columns(
        pl.col("is_depo").cum_sum().over(["locomotive_number"]).alias("trip_id")
    )


def build_edges(df: pl.DataFrame, depo: int | None = None, terminal: int | None = None) -> pl.DataFrame:
    if depo is not None:
        trip_depo = (
            df.filter(pl.col("is_depo") == 1)
            .select(["locomotive_series", "locomotive_number", "trip_id", pl.col("station").alias("depot")])
        )
        trips_of_depo = trip_depo.filter(pl.col("depot") == depo).select(
            ["locomotive_series", "locomotive_number", "trip_id"]
        )
        if terminal is not None:
            trips_visiting_term = (
                df.filter(pl.col("station") == terminal)
                .select(["locomotive_series", "locomotive_number", "trip_id"])
                .unique()
            )
            trips_of_depo = trips_of_depo.join(
                trips_visiting_term, on=["locomotive_series", "locomotive_number", "trip_id"], how="inner"
            )
        df_trip = df.join(trips_of_depo, on=["locomotive_series", "locomotive_number", "trip_id"], how="inner")
    else:
        df_trip = df
    df_trip = df_trip.with_columns(
        pl.col("station").shift(-1).over(["locomotive_number", "trip_id"]).alias("next_station")
    )
    edges = (
        df_trip
        .filter(pl.col("next_station").is_not_null())
        .select([pl.col("station").alias("from_st"), pl.col("next_station").alias("to_st")])
        .unique()
    )
    return edges


def _shortest_path(edges: pl.DataFrame, start: int, end: int) -> list[int]:
    edge_list = [(int(r["from_st"]), int(r["to_st"])) for r in edges.iter_rows(named=True)]
    adj: dict[int, list[int]] = {}
    for f, t in edge_list:
        adj.setdefault(f, []).append(t)

    from collections import deque

    parent: dict[int, int | None] = {start: None}
    q: deque[int] = deque([start])
    while q:
        v = q.popleft()
        if v == end:
            path: list[int] = []
            cur = end
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]
        for u in adj.get(v, []):
            if u not in parent:
                parent[u] = v
                q.append(u)
    return []


def _find_scc_representatives(edges: pl.DataFrame, all_stations: pl.Series) -> list:
    nodes = set(all_stations.to_list())
    edge_list = [(int(r["from_st"]), int(r["to_st"])) for r in edges.iter_rows(named=True)]
    adj: dict[int, list[int]] = {n: [] for n in nodes}
    for f, t in edge_list:
        if f in adj:
            adj[f].append(t)

    visited = set()
    order: list[int] = []

    def dfs1(v: int) -> None:
        visited.add(v)
        for u in adj.get(v, []):
            if u not in visited:
                dfs1(u)
        order.append(v)

    for n in nodes:
        if n not in visited:
            dfs1(n)

    rev_adj: dict[int, list[int]] = {n: [] for n in nodes}
    for f, t in edge_list:
        rev_adj[t].append(f)

    visited.clear()
    scc_reps: list[int] = []

    def dfs2(v: int, comp: list[int]) -> None:
        visited.add(v)
        comp.append(v)
        for u in rev_adj.get(v, []):
            if u not in visited:
                dfs2(u, comp)

    for n in reversed(order):
        if n not in visited:
            comp: list[int] = []
            dfs2(n, comp)
            if len(comp) > 1:
                scc_reps.append(comp[0])

    return scc_reps


def find_terminals(edges: pl.DataFrame, all_stations: pl.Series) -> pl.Series:
    from_stations = edges.select("from_st").unique().to_series()
    dead_ends = all_stations.filter(~all_stations.is_in(from_stations.to_list()))
    out_degree = edges.group_by("from_st").len()
    od1 = out_degree.filter(pl.col("len") == 1).select("from_st")
    edges_od1 = edges.join(od1, on="from_st", how="inner")
    reverse_exists = edges_od1.join(
        edges.select(
            pl.col("from_st").alias("rev_to"),
            pl.col("to_st").alias("rev_from"),
        ),
        left_on=["to_st", "from_st"],
        right_on=["rev_to", "rev_from"],
        how="inner",
    )
    turnarounds = reverse_exists.select("from_st").unique().to_series()
    result = pl.concat([dead_ends, turnarounds]).unique()
    if result.is_empty():
        cycle_reps = _find_scc_representatives(edges, all_stations)
        if cycle_reps:
            result = pl.Series("station", cycle_reps).cast(all_stations.dtype)
    return result


def compute_depo_branches(
    edges: pl.DataFrame,
    depo_stations: pl.Series,
    terminals: pl.Series,
) -> dict[int, list[tuple[list[int], int]]]:
    edges = edges.with_columns([
        pl.col("from_st").cast(pl.Int64),
        pl.col("to_st").cast(pl.Int64),
    ])
    depo_list = depo_stations.cast(pl.Int64).unique().to_list()
    term_list = terminals.cast(pl.Int64).unique().to_list()

    result: dict[int, list[tuple[list[int], int]]] = {}
    for depo in depo_list:
        branches: list[tuple[list[int], int]] = []
        for term in term_list:
            path = _shortest_path(edges, int(depo), int(term))
            if path:
                branches.append((path, int(term)))
        if branches:
            result[int(depo)] = branches
    return result


def compute_popular_branch_per_locomotive(
    df: pl.DataFrame,
    depo_branches: dict[int, list[tuple[list[int], int]]],
) -> dict[tuple[str, int], int]:
    branch_rows = []
    for depo, branches in depo_branches.items():
        for branch_idx, (stations, _) in enumerate(branches):
            for s in stations:
                branch_rows.append({"depo": depo, "branch_idx": branch_idx, "station": s})
    if not branch_rows:
        return {}
    branch_df = pl.DataFrame(branch_rows)

    df_trip = df.filter(pl.col("is_depo") == 0)
    trip_depo = (
        df.filter(pl.col("is_depo") == 1)
        .select(["locomotive_series", "locomotive_number", "trip_id", pl.col("station").alias("depot")])
    )
    trip_stations = (
        df_trip
        .group_by(["locomotive_series", "locomotive_number", "trip_id"])
        .agg(pl.col("station"))
    )
    trip_with_depo = trip_stations.join(
        trip_depo,
        on=["locomotive_series", "locomotive_number", "trip_id"],
        how="inner",
    )
    trip_exploded = trip_with_depo.explode("station")

    visits = (
        trip_exploded.join(
            branch_df,
            left_on=["depot", "station"],
            right_on=["depo", "station"],
            how="inner",
        )
        .select(["locomotive_series", "locomotive_number", "trip_id", "depot", "branch_idx"])
        .unique()
    )
    counts = visits.group_by(["locomotive_series", "locomotive_number", "depot", "branch_idx"]).len()
    best_branch = (
        counts.with_columns(
            pl.col("len").rank(method="dense", descending=True).over(["locomotive_series", "locomotive_number", "depot"]).alias("rn")
        )
        .filter(pl.col("rn") == 1)
        .select(["locomotive_series", "locomotive_number", "depot", "branch_idx"])
    )
    return {
        (str(r["locomotive_series"]), r["locomotive_number"], r["depot"]): r["branch_idx"]
        for r in best_branch.iter_rows(named=True)
    }


def visualize_depo(
    depo_station: int,
    depo_branches: dict[int, list[tuple[set[int], int]]],
    station_info: pl.DataFrame,
    df: pl.DataFrame,
    edges: pl.DataFrame,
    popular_branch: dict[tuple[str, int, int], int],
    top_n_locos: int = 5,
    output_path: str | Path = "map_visualization.html",
) -> None:
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        print("Для визуализации установите: pip install plotly")
        return

    branches = depo_branches.get(depo_station, [])
    if not branches:
        print(f"Нет веток для депо {depo_station}")
        return

    loco_counts = (
        df.filter(
            (pl.col("depo_station") == depo_station)
            & (pl.col("station") == depo_station)
        )
        .group_by(["locomotive_series", "locomotive_number"])
        .len()
        .sort("len", descending=True)
        .head(top_n_locos)
    )

    if loco_counts.is_empty():
        print(f"Нет локомотивов для депо {depo_station}")
        return

    stations_to_show = set()
    for path, _ in branches:
        stations_to_show.update(path)
    stations_to_show.add(depo_station)

    station_ids = list(stations_to_show)
    coords = (
        station_info
        .filter(pl.col("station").is_in(station_ids))
        .select(["station", "station_name", "latitude", "longitude"])
    )

    if coords.is_empty():
        print("Нет координат для отображаемых станций")
        return

    trip_depot = (
        df.filter(pl.col("is_depo") == 1)
        .select(["locomotive_series", "locomotive_number", "trip_id", pl.col("station").alias("trip_depot")])
    )
    visit_counts = (
        df.filter(pl.col("is_depo") == 0)
        .join(trip_depot, on=["locomotive_series", "locomotive_number", "trip_id"], how="inner")
        .filter(
            (pl.col("trip_depot") == depo_station) & pl.col("station").is_in(station_ids)
        )
        .group_by("station")
        .len()
    )

    plot_df = coords.join(visit_counts, on="station", how="left").with_columns(
        pl.col("len").fill_null(0).alias("visits")
    )

    coord_map = {int(r["station"]): (float(r["latitude"]), float(r["longitude"])) for r in plot_df.iter_rows(named=True)}

    all_branch_stations = set()
    for path, _ in branches:
        all_branch_stations.update(path)

    fig = go.Figure()
    colors = [
        "#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed",
        "#0891b2", "#65a30d", "#be123c", "#4f46e5",
    ]

    for i, (path, term) in enumerate(branches):
        path_coords = [p for p in path if p in coord_map]
        if len(path_coords) >= 2:
            lats = [coord_map[p][0] for p in path_coords]
            lons = [coord_map[p][1] for p in path_coords]
            fig.add_trace(go.Scattermap(
                lat=lats, lon=lons, mode="lines",
                line=dict(width=4, color=colors[i % len(colors)]),
                name=f"Ветка → {term}", legendgroup=f"branch_{i}", showlegend=True,
            ))

    fig.add_trace(go.Scattermap(
        lat=plot_df["latitude"],
        lon=plot_df["longitude"],
        mode="markers+text",
        marker=dict(
            size=plot_df["visits"].clip(1, 30) + 4,
            color=plot_df["visits"],
            colorscale=[
                [0, "#e0f2fe"], [0.3, "#38bdf8"], [0.6, "#0284c7"], [1, "#0c4a6e"],
            ],
            showscale=True,
            colorbar=dict(
                title=dict(text="Посещений", font=dict(size=11)),
                thickness=15,
                len=0.4,
                x=1.02,
                y=0.92,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#e5e7eb",
                borderwidth=1,
            ),
            opacity=0.9,
        ),
        text=[
            f"🚉 {name}<br>📊 {v} посещ."
            for name, v in zip(plot_df["station_name"], plot_df["visits"])
        ],
        textposition="top center",
        textfont=dict(size=8, color="#374151"),
        name="Станции",
    ))

    depo_coords = plot_df.filter(pl.col("station") == depo_station)
    if not depo_coords.is_empty():
        d = depo_coords.to_dicts()[0]
        depo_name = d.get("station_name", "") or ""
        depo_label = f"🏭 Депо {depo_station}" + (f"<br>{depo_name}" if depo_name else "")
        fig.add_trace(go.Scattermap(
            lat=[d["latitude"]], lon=[d["longitude"]],
            mode="markers+text",
            marker=dict(
                size=44,
                color="#b91c1c",
                symbol="circle",
                opacity=1,
            ),
            text=[depo_label],
            textposition="top center",
            textfont=dict(size=13, color="#7f1d1d"),
            name="Депо (корень веток)",
            legendgroup="depo",
        ))

    depo_row = plot_df.filter(pl.col("station") == depo_station)
    center_lat = float(depo_row["latitude"][0]) if not depo_row.is_empty() else float(plot_df["latitude"].mean())
    center_lon = float(depo_row["longitude"][0]) if not depo_row.is_empty() else float(plot_df["longitude"].mean())

    loco_lines = []
    for r in loco_counts.iter_rows(named=True):
        series, num = r["locomotive_series"], r["locomotive_number"]
        branch_idx = popular_branch.get((str(series), num, depo_station))
        term = branches[branch_idx][1] if branch_idx is not None and branch_idx < len(branches) else "—"
        loco_lines.append(f"• {series} {num} → ветка к {term}")
    loco_info_html = (
        "<b>Популярные ветки локомотивов</b><br>" + "<br>".join(loco_lines)
        if loco_lines
        else ""
    )

    layout_kw = dict(
        map=dict(
            style="carto-positron",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=4,
        ),
        title=dict(
            x=0.5,
            xanchor="center",
            font=dict(size=18, color="#111827"),
        ),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e5e7eb",
            borderwidth=1,
            font=dict(size=11),
            x=1.02,
            y=0.50,
            yanchor="top",
        ),
        height=750,
        margin=dict(t=80, r=20, b=20, l=20),
        paper_bgcolor="#f8fafc",
    )
    if loco_info_html:
        layout_kw["annotations"] = [
            dict(
                text=loco_info_html,
                x=0.02,
                y=0.02,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="bottom",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="#cbd5e1",
                borderwidth=1,
                borderpad=10,
                font=dict(size=11, color="#334155"),
                align="left",
            )
        ]
    fig.update_layout(**layout_kw)

    out = Path(output_path)
    fig.write_html(str(out))
    print(f"Карта сохранена: {out.absolute()}")


def visualize_all_depos(
    depo_branches: dict[int, list[tuple[set[int], int]]],
    station_info: pl.DataFrame,
    df: pl.DataFrame,
    popular_branch: dict[tuple[str, int, int], int],
    top_n_locos: int = 5,
    output_path: str | Path = "map_visualization.html",
) -> None:
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        print("Для визуализации установите: pip install plotly")
        return

    depo_list = [d for d, b in depo_branches.items() if b]
    if not depo_list:
        print("Нет депо с ветками для визуализации")
        return

    colors = [
        "#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed",
        "#0891b2", "#65a30d", "#be123c", "#4f46e5",
    ]
    all_traces = []
    depo_trace_ranges = {}
    depo_layouts = {}
    first_depo = depo_list[0]

    for depo_station in depo_list:
        is_first = depo_station == first_depo
        branches = depo_branches[depo_station]

        loco_counts = (
            df.filter(
                (pl.col("depo_station") == depo_station)
                & (pl.col("station") == depo_station)
            )
            .group_by(["locomotive_series", "locomotive_number"])
            .len()
            .sort("len", descending=True)
            .head(top_n_locos)
        )
        stations_to_show = set()
        for path, _ in branches:
            stations_to_show.update(path)
        stations_to_show.add(depo_station)
        station_ids = list(stations_to_show)
        coords = station_info.filter(pl.col("station").is_in(station_ids)).select(
            ["station", "station_name", "latitude", "longitude"]
        )
        if coords.is_empty():
            continue
        trip_depot = (
            df.filter(pl.col("is_depo") == 1)
            .select(["locomotive_series", "locomotive_number", "trip_id", pl.col("station").alias("trip_depot")])
        )
        visit_counts = (
            df.filter(pl.col("is_depo") == 0)
            .join(trip_depot, on=["locomotive_series", "locomotive_number", "trip_id"], how="inner")
            .filter(
                (pl.col("trip_depot") == depo_station) & pl.col("station").is_in(station_ids)
            )
            .group_by("station")
            .len()
        )
        plot_df = coords.join(visit_counts, on="station", how="left").with_columns(
            pl.col("len").fill_null(0).alias("visits")
        )
        coord_map = {int(r["station"]): (float(r["latitude"]), float(r["longitude"])) for r in plot_df.iter_rows(named=True)}

        trace_start = len(all_traces)

        for i, (path, term) in enumerate(branches):
            path_coords = [p for p in path if p in coord_map]
            if len(path_coords) >= 2:
                lats = [coord_map[p][0] for p in path_coords]
                lons = [coord_map[p][1] for p in path_coords]
                all_traces.append(go.Scattermap(
                    lat=lats,
                    lon=lons,
                    mode="lines",
                    line=dict(
                        width=4,
                        color=colors[i % len(colors)],
                    ),
                    name=f"Ветка → {term}",
                    legendgroup=f"branch_{depo_station}_{i}",
                    visible=is_first,
                ))

        all_traces.append(go.Scattermap(
            lat=plot_df["latitude"],
            lon=plot_df["longitude"],
            mode="markers+text",
            marker=dict(
                size=plot_df["visits"].clip(1, 30) + 4,
                color=plot_df["visits"],
                colorscale=[
                    [0, "#e0f2fe"], [0.3, "#38bdf8"], [0.6, "#0284c7"], [1, "#0c4a6e"],
                ],
            showscale=True,
            colorbar=dict(
                title=dict(text="Посещений", font=dict(size=11)),
                thickness=15,
                len=0.4,
                x=1.02,
                y=0.92,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#e5e7eb",
                borderwidth=1,
                tickfont=dict(size=10),
            ),
                opacity=0.9,
            ),
            text=[
                f"🚉 {name}<br>📊 {v} посещ."
                for name, v in zip(plot_df["station_name"], plot_df["visits"])
            ],
            textposition="top center",
            textfont=dict(size=8, color="#374151", family="Arial, sans-serif"),
            name="Станции",
            legendgroup=f"stations_{depo_station}",
            visible=is_first,
        ))

        depo_coords = plot_df.filter(pl.col("station") == depo_station)
        if not depo_coords.is_empty():
            d = depo_coords.to_dicts()[0]
            depo_name = d.get("station_name", "") or ""
            depo_label = f"🏭 Депо {depo_station}" + (f"<br>{depo_name}" if depo_name else "")
            all_traces.append(go.Scattermap(
                lat=[d["latitude"]], lon=[d["longitude"]],
                mode="markers+text",
                marker=dict(
                    size=44,
                    color="#b91c1c",
                    symbol="circle",
                    opacity=1,
                ),
                text=[depo_label],
                textposition="top center",
                textfont=dict(size=13, color="#7f1d1d", family="Arial, sans-serif"),
                name="Депо",
                legendgroup=f"depo_{depo_station}",
                visible=is_first,
            ))

        depo_row = plot_df.filter(pl.col("station") == depo_station)
        center_lat = float(depo_row["latitude"][0]) if not depo_row.is_empty() else float(plot_df["latitude"].mean())
        center_lon = float(depo_row["longitude"][0]) if not depo_row.is_empty() else float(plot_df["longitude"].mean())
        loco_lines = []
        for r in loco_counts.iter_rows(named=True):
            branch_idx = popular_branch.get((str(r["locomotive_series"]), r["locomotive_number"], depo_station))
            term = branches[branch_idx][1] if branch_idx is not None and branch_idx < len(branches) else "—"
            loco_lines.append(f"• {r['locomotive_series']} {r['locomotive_number']} → ветка к {term}")

        loco_info_html = (
            "<b>Популярные ветки локомотивов</b><br>"
            + "<br>".join(loco_lines)
            if loco_lines
            else ""
        )

        depo_trace_ranges[depo_station] = (trace_start, len(all_traces))
        depo_layouts[depo_station] = dict(
            map=dict(
                center=dict(lat=center_lat, lon=center_lon),
                zoom=4,
                style="carto-positron",
            ),
            title=dict(
                x=0.5,
                xanchor="center",
                font=dict(size=18, color="#111827", family="Arial, sans-serif"),
            ),
            annotations=[
                dict(
                    text=loco_info_html,
                    x=0.02,
                    y=0.02,
                    xref="paper",
                    yref="paper",
                    xanchor="left",
                    yanchor="bottom",
                    showarrow=False,
                    bgcolor="rgba(255,255,255,0.95)",
                    bordercolor="#cbd5e1",
                    borderwidth=1,
                    borderpad=10,
                    font=dict(size=11, color="#334155", family="Arial, sans-serif"),
                    align="left",
                )
            ]
            if loco_info_html
            else [],
        )

    depo_names = (
        station_info.filter(pl.col("station").is_in(depo_list))
        .select(["station", "station_name"])
        .to_dict(as_series=False)
    )
    depo_name_map = dict(zip(depo_names["station"], depo_names["station_name"]))

    n_total = len(all_traces)
    buttons = []
    for depo in depo_list:
        vis = [False] * n_total
        s, e = depo_trace_ranges[depo]
        for i in range(s, e):
            vis[i] = True
        layout_upd = depo_layouts[depo]
        name = depo_name_map.get(depo, "")
        label = f"{depo} — {name}" if name else str(depo)
        buttons.append(dict(
            label=label,
            method="update",
            args=[
                {"visible": vis},
                layout_upd,
            ],
        ))

    fig = go.Figure(data=all_traces)
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.02,
                xanchor="left",
                y=0.98,
                yanchor="top",
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="#cbd5e1",
                borderwidth=1.5,
                font=dict(size=12, color="#334155", family="Arial, sans-serif"),
                pad=dict(t=8, r=8, b=8, l=8),
                active=0,
            )
        ],
        **depo_layouts[first_depo],
        showlegend=True,
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e5e7eb",
            borderwidth=1,
            font=dict(size=11, family="Arial, sans-serif"),
            x=1.02,
            y=0.50,
            yanchor="top",
            xanchor="left",
        ),
        height=750,
        margin=dict(t=80, r=20, b=20, l=20),
        paper_bgcolor="#f8fafc",
        font=dict(family="Arial, sans-serif", size=12),
    )

    out = Path(output_path)
    fig.write_html(str(out))
    print(f"Карта с выбором депо сохранена: {out.absolute()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Анализ перемещений локомотивов: ветки депо, популярные ветки, визуализация."
    )
    parser.add_argument(
        "-d",
        "--displacement",
        type=str,
        default=None,
        help="Путь к файлу перемещений (.csv или .json). По умолчанию: locomotives_displacement.csv",
    )
    parser.add_argument(
        "-s",
        "--stations",
        type=str,
        default=None,
        help="Путь к файлу станций (.csv или .json). По умолчанию: station_info.csv",
    )
    parser.add_argument(
        "-p",
        "--depo",
        type=int,
        default=None,
        help="Номер депо для визуализации. По умолчанию — первое депо с ветками.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Загрузка данных...")
    try:
        df = load_displacement(args.displacement)
        station_info = load_station_info(args.stations)
    except (ValidationError, FileNotFoundError) as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Данные загружены! Всего записей перемещений: {df.height:,}")

    df = build_trips(df)

    depo_stations = (
        df.select("depo_station")
        .unique()
        .filter(pl.col("depo_station").is_not_null())
        .to_series()
    )
    depo_branches: dict[int, list[tuple[list[int], int]]] = {}
    for depo in depo_stations.to_list():
        edges_d = build_edges(df, depo)
        if edges_d.is_empty():
            continue
        all_st_d = pl.concat([edges_d["from_st"], edges_d["to_st"]]).unique()
        terminals_d = find_terminals(edges_d, all_st_d)
        branches_list = []
        for term in terminals_d.to_list():
            edges_branch = build_edges(df, depo, term)
            if edges_branch.is_empty():
                continue
            all_st_b = pl.concat([edges_branch["from_st"], edges_branch["to_st"]]).unique()
            terms_b = find_terminals(edges_branch, all_st_b)
            branches_b = compute_depo_branches(edges_branch, pl.Series([depo]), terms_b)
            for path, t in branches_b.get(depo, []):
                if t == term:
                    branches_list.append((path, int(term)))
                    break
        branches_list = [
            (path, term)
            for path, term in branches_list
            if not any(
                set(path) < set(other_path)
                for other_path, _ in branches_list
                if other_path != path
            )
        ]
        depo_branches[depo] = branches_list
    print(f"Граф построен! Депо: {len(depo_branches)}, веток: {sum(len(b) for b in depo_branches.values())}")

    for depo, branches in list(depo_branches.items())[:]:
        terms = [t for _, t in branches]
        print(f"  Депо {depo}, конечные станции веток: {terms[:10]}{'...' if len(terms) > 10 else ''}")

    print("\nВычисление популярных веток по локомотивам...")
    popular_branch = compute_popular_branch_per_locomotive(df, depo_branches)
    print(f"Локомотивов с определённой популярной веткой: {len(popular_branch)}")

    depo_list = [d for d in depo_stations.to_list() if depo_branches.get(d)]
    if args.depo is not None:
        if args.depo in depo_branches:
            depo_list = [args.depo]
        else:
            print(f"Депо {args.depo} не найдено. Доступные: {depo_list}", file=sys.stderr)
    if not depo_list:
        print("Нет депо для визуализации")
    else:
        print(f"\nВизуализация с выбором депо ({len(depo_list)} депо)...")
        depo_branches_filtered = {d: depo_branches[d] for d in depo_list}
        visualize_all_depos(depo_branches_filtered, station_info, df, popular_branch, output_path="map_visualization.html")


if __name__ == "__main__":
    main()
