"""Plotly chart helpers for league_analysis.

Each helper returns an HTML string (div) or None if data is insufficient.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional


def make_stacked_roster_chart(pivot_df, title: str, div_id: str, height: int = 500) -> Optional[str]:
    if pivot_df is None or pivot_df.empty:
        return None
    sparse_note = ""
    if pivot_df.shape[0] < 2:
        sparse_note = "<p><em>Only one season of data is available for this view; trend lines will appear as single points.</em></p>"
    fig = go.Figure()
    for position in pivot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=pivot_df.index,
                y=pivot_df[position],
                mode="lines+markers",
                name=position,
                stackgroup="one",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Number of Players",
        template="plotly_white",
        height=height,
        hovermode="x unified",
    )
    html = fig.to_html(include_plotlyjs="cdn", div_id=div_id)
    if sparse_note:
        html = sparse_note + html
    return html


def make_competitiveness_chart(combined) -> Optional[str]:
    if combined is None or combined.empty or "cv" not in combined:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=combined["year"],
            y=combined["cv"],
            mode="lines+markers",
            name="CV",
            line=dict(color="#007bff", width=3),
            marker=dict(size=10),
            hovertemplate="Year: %{x}<br>CV: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_hline(
        y=combined["cv"].mean(),
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Mean: {combined['cv'].mean():.2f}%",
    )
    best_idx = combined["cv"].idxmin()
    worst_idx = combined["cv"].idxmax()
    fig.add_annotation(
        x=combined.loc[best_idx, "year"],
        y=combined.loc[best_idx, "cv"],
        text=f"Most Competitive<br>{combined.loc[best_idx, 'cv']:.2f}%",
        showarrow=True,
        arrowhead=2,
        arrowcolor="green",
        bgcolor="lightgreen",
    )
    fig.add_annotation(
        x=combined.loc[worst_idx, "year"],
        y=combined.loc[worst_idx, "cv"],
        text=f"Least Competitive<br>{combined.loc[worst_idx, 'cv']:.2f}%",
        showarrow=True,
        arrowhead=2,
        arrowcolor="red",
        bgcolor="lightcoral",
    )
    fig.update_layout(
        title="League Competitiveness Over Time<br><sub>Lower CV = More Competitive (wins distributed evenly)</sub>",
        xaxis_title="Year",
        yaxis_title="Coefficient of Variation (%)",
        template="plotly_white",
        height=500,
    )
    return fig.to_html(include_plotlyjs="cdn", div_id="cv_chart")


def make_matchup_fairness_chart(combined) -> Optional[str]:
    if combined is None or combined.empty:
        return None
    if "avg_differential" not in combined.columns or "close_matchups_pct" not in combined.columns:
        return None
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Average Score Differential (Lower = Closer Games)",
            "Close Matchups Percentage (Higher = More Competitive)",
        ),
        vertical_spacing=0.15,
    )
    fig.add_trace(
        go.Bar(
            x=combined["year"],
            y=combined["avg_differential"],
            name="Avg Differential",
            marker_color="#28a745",
            hovertemplate="Year: %{x}<br>Avg Diff: %{y:.2f} pts<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=combined["year"],
            y=combined["close_matchups_pct"],
            mode="lines+markers",
            name="Close Matchups %",
            line=dict(color="#dc3545", width=3),
            marker=dict(size=8),
            hovertemplate="Year: %{x}<br>Close Games: %{y:.1f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=combined["avg_differential"].mean(),
        line_dash="dash",
        line_color="gray",
        row=1,
        col=1,
    )  # type: ignore[arg-type]
    fig.add_hline(
        y=combined["close_matchups_pct"].mean(),
        line_dash="dash",
        line_color="gray",
        row=2,
        col=1,
    )  # type: ignore[arg-type]
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_yaxes(title_text="Points", row=1, col=1)
    fig.update_yaxes(title_text="Percentage (%)", row=2, col=1)
    fig.update_layout(height=700, template="plotly_white", showlegend=False)
    return fig.to_html(include_plotlyjs="cdn", div_id="fairness_chart")


def make_close_vs_blowouts_chart(combined) -> Optional[str]:
    if combined is None or combined.empty:
        return None
    if "close_matchups_pct" not in combined.columns or "blowouts_pct" not in combined.columns:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=combined["year"],
            y=combined["close_matchups_pct"],
            mode="lines+markers",
            name="Close Matchups %",
            line=dict(color="#28a745", width=3),
            marker=dict(size=8),
            hovertemplate="Year: %{x}<br>Close: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=combined["year"],
            y=combined["blowouts_pct"],
            mode="lines+markers",
            name="Blowouts %",
            line=dict(color="#dc3545", width=3),
            marker=dict(size=8),
            hovertemplate="Year: %{x}<br>Blowouts: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="Close Matchups vs Blowouts Over Time",
        xaxis_title="Year",
        yaxis_title="Percentage of Matchups",
        template="plotly_white",
        height=450,
        legend_title="Matchup Type",
    )
    return fig.to_html(include_plotlyjs="cdn", div_id="close_vs_blowout_chart")


def make_settings_vs_competitiveness_chart(combined) -> Optional[str]:
    if combined is None or combined.empty:
        return None
    if "rec_points" not in combined.columns or "cv" not in combined.columns:
        return None
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=combined["year"],
            y=combined["rec_points"],
            name="PPR Points",
            marker_color="rgba(0, 123, 255, 0.6)",
            hovertemplate="Year: %{x}<br>PPR: %{y:.1f} pts/rec<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=combined["year"],
            y=combined["cv"],
            mode="lines+markers",
            name="CV",
            line=dict(color="#ff6b6b", width=3),
            marker=dict(size=10),
            hovertemplate="Year: %{x}<br>CV: %{y:.2f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="PPR Points per Reception", secondary_y=False)
    fig.update_yaxes(title_text="Coefficient of Variation (%)", secondary_y=True)
    fig.update_layout(
        title="PPR Setting vs League Competitiveness<br><sub>Shows correlation between scoring changes and competitive balance</sub>",
        template="plotly_white",
        height=500,
    )
    return fig.to_html(include_plotlyjs="cdn", div_id="settings_chart")


def make_multi_metric_dashboard(combined) -> Optional[str]:
    needed = ["cv", "avg_differential", "close_matchups_pct", "blowouts_pct"]
    if combined is None or combined.empty or not all(col in combined.columns for col in needed):
        return None
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Win Diversity (CV)", "Avg Score Differential", "Close Matchups %", "Blowouts %"),
    )
    fig.add_trace(
        go.Scatter(x=combined["year"], y=combined["cv"], mode="lines+markers", name="CV", line=dict(color="#007bff")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=combined["year"], y=combined["avg_differential"], mode="lines+markers", name="Avg Diff", line=dict(color="#28a745")),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=combined["year"], y=combined["close_matchups_pct"], mode="lines+markers", name="Close %", line=dict(color="#ffc107")),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=combined["year"], y=combined["blowouts_pct"], mode="lines+markers", name="Blowouts %", line=dict(color="#dc3545")),
        row=2,
        col=2,
    )
    fig.update_layout(height=700, template="plotly_white", showlegend=False)
    return fig.to_html(include_plotlyjs="cdn", div_id="dashboard")


def make_recent_comp_chart(recent_years) -> Optional[str]:
    if recent_years is None or recent_years.empty or "cv" not in recent_years.columns:
        return None
    colors = [
        "#28a745" if cv == recent_years["cv"].min() else "#dc3545" if cv == recent_years["cv"].max() else "#007bff"
        for cv in recent_years["cv"]
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=recent_years["year"],
            y=recent_years["cv"],
            marker_color=colors,
            text=recent_years["cv"].round(2),
            textposition="auto",
            hovertemplate="Year: %{x}<br>CV: %{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="Recent Year Competitiveness Comparison (2023-2025)<br><sub>Green = Best, Red = Worst</sub>",
        xaxis_title="Year",
        yaxis_title="Coefficient of Variation (%)",
        template="plotly_white",
        height=400,
    )
    return fig.to_html(include_plotlyjs="cdn", div_id="recent_chart")


def make_finish_std_chart(finish_diversity) -> Optional[str]:
    if finish_diversity is None or finish_diversity.empty or "Finish Std" not in finish_diversity.columns:
        return None
    fig_finish_std = go.Figure()
    fig_finish_std.add_trace(
        go.Bar(
            x=finish_diversity["Owner"],
            y=finish_diversity["Finish Std"],
            marker_color="#6c63ff",
            hovertemplate="Owner: %{x}<br>Finish Std: %{y:.2f}<extra></extra>",
        )
    )
    fig_finish_std.update_layout(
        title="Finish Volatility by Team (Std Dev of Final Standing)",
        xaxis_title="Owner",
        yaxis_title="Finish Std Dev",
        template="plotly_white",
        height=400,
    )
    return fig_finish_std.to_html(include_plotlyjs="cdn", div_id="finish_std_chart")
