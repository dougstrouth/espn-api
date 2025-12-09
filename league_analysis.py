#!/usr/bin/env python3
"""
ESPN Fantasy Football League Comparison Tool (2016-2025)
Analyzes matchups, rosters, settings, and fairness metrics across multiple years.
Run this script to compare league data and correlate settings changes with fairness.
"""

import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import defaultdict
from espn_api.football import League

# =============================================================================
# CONFIGURATION
# =============================================================================
ESPN_S2 = 'AEBvr5lKQnBF4qZ6iAb0dQxlVe%2FFjmDvoodlVueyffeuuWHthnUO5U8URkyOro9P95i5hHi7uVO8rsMDwEfrhM1ocAAbtLKAS2DWchuOib5sE5%2FVGoas05b1tnifeeCOZ0Q5ANyjF3JLYRdrq39OR1a%2BucdWXw7XZELwdX4a7bacXJwa%2FD1izL1Y%2F1KZ%2BxjkC%2BC1at7XtQsf7HHxll8m1WcrclJUtEFNZKoSDRu6alVqBzaf3x6c7WtEzaOplUbSFNhklgDzmJjrK9HHvGnzKmcvJ8I4hhCtm9fNRuIAyu9R2eeMvuyyjxYMRQgw%2F5YkW3twSXx0uVXA3VEl4Zn%2BVV%2FR'
SWID = '{9A39BAD6-0F58-4406-B9BA-D60F587406D2}'
LEAGUE_ID = 247704
YEARS = range(2016, 2026)  # 2016-2025 (range end is exclusive)

# =============================================================================
# LOAD LEAGUES
# =============================================================================
print("=" * 80)
print("Loading leagues (2016-2025)...")
print("=" * 80)

leagues = {}

for year in YEARS:
    try:
        league = League(league_id=LEAGUE_ID, year=year, espn_s2=ESPN_S2, swid=SWID)
        leagues[year] = league
        print(f"✓ {year} League loaded: {len(league.teams)} teams")
    except Exception as e:
        print(f"✗ Could not load {year} league: {e}")
        leagues[year] = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def extract_settings(league_obj, year):
    """Extract key settings from league"""
    settings = league_obj.settings
    
    # Convert position_slot_counts dict to JSON string for CSV storage
    position_slots = getattr(settings, 'position_slot_counts', {})
    
    # Extract key scoring rules
    scoring_format = getattr(settings, 'scoring_format', [])
    scoring_dict = {item.get('abbr', item.get('label', 'unknown')): item.get('points', 0) 
                    for item in scoring_format}
    
    return {
        'year': year,
        'name': settings.name,
        'team_count': settings.team_count,
        'reg_season_count': settings.reg_season_count,
        'playoff_team_count': settings.playoff_team_count,
        'scoring_type': settings.scoring_type,
        'faab': settings.faab,
        'acquisition_budget': settings.acquisition_budget,
        'trade_deadline': settings.trade_deadline,
        'veto_votes_required': settings.veto_votes_required,
        'keeper_count': settings.keeper_count,
        'tie_rule': settings.tie_rule,
        'playoff_seed_tie_rule': settings.playoff_seed_tie_rule,
        'position_slot_counts': json.dumps(position_slots),  # Store as JSON string
        'scoring_format_count': len(scoring_format),
        # Key scoring settings
        'pass_td_points': scoring_dict.get('PTD', scoring_dict.get('Pass TD', 0)),
        'rush_td_points': scoring_dict.get('RTD', scoring_dict.get('Rush TD', 0)),
        'rec_td_points': scoring_dict.get('RETD', scoring_dict.get('TD Reception', 0)),
        'rec_points': scoring_dict.get('REC', scoring_dict.get('Each reception', 0)),  # PPR setting
    }


def get_all_matchups(league_obj, year):
    """Fetch all matchups for a given league"""
    matchups_by_week = {}
    max_weeks = league_obj.settings.reg_season_count
    
    for week in range(1, max_weeks + 1):
        try:
            # Use box_scores for 2019+, scoreboard for pre-2019
            if year >= 2019:
                matchups = league_obj.box_scores(week)
            else:
                matchups = league_obj.scoreboard(week)
            matchups_by_week[week] = matchups
        except Exception as e:
            print(f"  Error fetching week {week} for {year}: {e}")
    
    return matchups_by_week


def build_matchup_dataframe(matchups_by_week, year):
    """Convert matchups to DataFrame for analysis"""
    data = []
    for week, matchups in matchups_by_week.items():
        for matchup in matchups:
            data.append({
                'year': year,
                'week': week,
                'home_team': matchup.home_team.team_name,
                'away_team': matchup.away_team.team_name,
                'home_score': matchup.home_score,
                'away_score': matchup.away_score,
                'winner': matchup.home_team.team_name if matchup.home_score > matchup.away_score else matchup.away_team.team_name,
                'score_differential': abs(matchup.home_score - matchup.away_score),
                'total_points': matchup.home_score + matchup.away_score
            })
    return pd.DataFrame(data)


def analyze_roster_composition(league_obj, year):
    """Analyze roster diversity by position"""
    roster_data = []
    position_counts = defaultdict(int)
    
    for team in league_obj.teams:
        for player in team.roster:
            roster_data.append({
                'year': year,
                'team': team.team_name,
                'player': player.name,
                'position': player.position,
                'projected_points': getattr(player, 'projected_total_points', 0)
            })
            position_counts[player.position] += 1
    
    return pd.DataFrame(roster_data)


def analyze_projected_fairness(matchups_df):
    """Analyze matchup competitiveness (close games vs blowouts)"""
    if matchups_df.empty:
        return {'avg_differential': None, 'close_matchups_pct': None, 'blowouts_pct': None}
    
    avg_differential = matchups_df['score_differential'].mean()
    close_matchups = (matchups_df['score_differential'] < 5).sum()
    blowouts = (matchups_df['score_differential'] > 20).sum()
    
    return {
        'avg_differential': avg_differential,
        'close_matchups_pct': (close_matchups / len(matchups_df)) * 100,
        'blowouts_pct': (blowouts / len(matchups_df)) * 100
    }


def analyze_win_diversity(matchups_df):
    """Calculate Coefficient of Variation for win distribution"""
    if matchups_df.empty:
        return {'cv': None, 'avg_wins': None, 'std_wins': None}
    
    win_counts = matchups_df.groupby('winner').size()
    avg_wins = win_counts.mean()
    std_wins = win_counts.std()
    cv = (std_wins / avg_wins) * 100 if avg_wins > 0 else None
    
    return {
        'cv': cv,
        'avg_wins': avg_wins,
        'std_wins': std_wins,
        'min_wins': win_counts.min(),
        'max_wins': win_counts.max()
    }


def extract_power_rankings(league_obj, year, week=None):
    """Extract power rankings for a given week (or final week if not specified)"""
    try:
        if week is None:
            week = league_obj.settings.reg_season_count
        
        rankings = league_obj.power_rankings(week=week)
        data = []
        for rank, (score, team) in enumerate(rankings, 1):
            data.append({
                'year': year,
                'week': week,
                'rank': rank,
                'team': team.team_name,
                'power_score': float(score)
            })
        return pd.DataFrame(data)
    except Exception as e:
        print(f"  Error fetching power rankings for {year} week {week}: {e}")
        return pd.DataFrame()


# =============================================================================
# ANALYSIS SECTION 1: SETTINGS EXTRACTION
# =============================================================================
print("\n" + "=" * 80)
print("SETTINGS EXTRACTION (CSV-FIRST)")
print("=" * 80)

settings_df = pd.DataFrame()

try:
    settings_df = pd.read_csv('settings_all_years.csv')
    print(f"✓ Loaded settings_all_years.csv ({len(settings_df)} years)")
except FileNotFoundError:
    settings_list = []
    for year, league in leagues.items():
        if league:
            settings_list.append(extract_settings(league, year))
    settings_df = pd.DataFrame(settings_list)
    print(f"Extracted settings for {len(settings_df)} years")

if not settings_df.empty:
    print("\nSettings Summary:")
    print(settings_df[['year', 'team_count', 'faab', 'acquisition_budget', 'playoff_team_count']].to_string(index=False))
    
    # Analyze roster slot changes
    print("\nRoster Slot Configuration Changes:")
    for idx, row in settings_df.iterrows():
        try:
            slots = json.loads(row['position_slot_counts'])
            if slots:
                print(f"\n{int(row['year'])}:")
                for pos, count in sorted(slots.items()):
                    print(f"  {pos}: {count}")
        except (json.JSONDecodeError, KeyError):
            pass
    
    # Show scoring changes
    if 'rec_points' in settings_df.columns:
        print("\nScoring Format Changes (Key Settings):")
        scoring_cols = ['year', 'rec_points', 'pass_td_points', 'rush_td_points', 'rec_td_points']
        available_scoring = [c for c in scoring_cols if c in settings_df.columns]
        if len(available_scoring) > 1:
            print(settings_df[available_scoring].to_string(index=False))

# =============================================================================
# ANALYSIS SECTION 2: MATCHUP DATA (CSV-FIRST)
# =============================================================================
print("\n" + "=" * 80)
print("MATCHUP DATA COLLECTION (CSV-FIRST)")
print("=" * 80)

matchups_by_year = {}
matchups_df_by_year = {}

for year in YEARS:
    df = pd.DataFrame()
    try:
        df = pd.read_csv(f'matchups_{year}.csv')
        print(f"✓ Loaded matchups_{year}.csv")
    except FileNotFoundError:
        if leagues.get(year):
            matchups = get_all_matchups(leagues[year], year)
            matchups_by_year[year] = matchups
            df = build_matchup_dataframe(matchups, year)
            print(f"Fetched {len(df)} matchups for {year}")
        else:
            print(f"✗ Skipping {year} matchups (no league and no CSV)")
    
    matchups_df_by_year[year] = df

all_matchups_df = pd.concat([df for df in matchups_df_by_year.values() if not df.empty], ignore_index=True)

# =============================================================================
# ANALYSIS SECTION 3: ROSTER COMPOSITION (CSV-FIRST)
# =============================================================================
print("\n" + "=" * 80)
print("ROSTER COMPOSITION ANALYSIS (CSV-FIRST)")
print("=" * 80)

roster_df_by_year = {}

for year in YEARS:
    df = pd.DataFrame()
    try:
        df = pd.read_csv(f'roster_{year}.csv')
        print(f"✓ Loaded roster_{year}.csv")
    except FileNotFoundError:
        if leagues.get(year):
            df = analyze_roster_composition(leagues[year], year)
            print(f"Analyzed roster composition for {year}")
        else:
            print(f"✗ Skipping {year} roster analysis")
    
    roster_df_by_year[year] = df

all_rosters_df = pd.concat([df for df in roster_df_by_year.values() if not df.empty], ignore_index=True)

# Show player pool trends
if not all_rosters_df.empty:
    player_counts = all_rosters_df.groupby('year')['player'].nunique()
    print("\nUnique Players by Year:")
    print(player_counts.to_string())

# =============================================================================
# ANALYSIS SECTION 4: POWER RANKINGS (CSV-FIRST)
# =============================================================================
print("\n" + "=" * 80)
print("POWER RANKINGS ANALYSIS (CSV-FIRST)")
print("=" * 80)

power_rankings_df_by_year = {}

for year in YEARS:
    df = pd.DataFrame()
    try:
        df = pd.read_csv(f'power_rankings_{year}.csv')
        print(f"✓ Loaded power_rankings_{year}.csv")
    except FileNotFoundError:
        if leagues.get(year):
            # Get final week power rankings
            df = extract_power_rankings(leagues[year], year)
            if not df.empty:
                print(f"Fetched power rankings for {year}")
        else:
            print(f"✗ Skipping {year} power rankings")
    
    power_rankings_df_by_year[year] = df

all_power_rankings_df = pd.concat([df for df in power_rankings_df_by_year.values() if not df.empty], ignore_index=True)

# Analyze power rankings spread (competitive balance)
if not all_power_rankings_df.empty:
    power_spread = all_power_rankings_df.groupby('year')['power_score'].agg(['std', 'min', 'max'])
    power_spread['range'] = power_spread['max'] - power_spread['min']
    print("\nPower Rankings Spread by Year (lower std/range = more competitive):")
    print(power_spread.to_string())

# =============================================================================
# ANALYSIS SECTION 5: FAIRNESS METRICS
# =============================================================================
print("\n" + "=" * 80)
print("PROJECTED FAIRNESS ANALYSIS (ALL YEARS)")
print("=" * 80)

fairness_list = []
for year, df in matchups_df_by_year.items():
    if not df.empty:
        fairness = analyze_projected_fairness(df)
        fairness['year'] = year
        fairness_list.append(fairness)

fairness_df = pd.DataFrame(fairness_list)
if not fairness_df.empty:
    fairness_df = fairness_df.sort_values('year')
    print("\nFairness Metrics by Year:")
    print(fairness_df[['year', 'avg_differential', 'close_matchups_pct', 'blowouts_pct']].to_string(index=False))

# =============================================================================
# ANALYSIS SECTION 6: WIN DISTRIBUTION DIVERSITY
# =============================================================================
print("\n" + "=" * 80)
print("WIN DISTRIBUTION DIVERSITY (ALL YEARS)")
print("=" * 80)

diversity_list = []
for year, df in matchups_df_by_year.items():
    if not df.empty:
        diversity = analyze_win_diversity(df)
        diversity['year'] = year
        diversity_list.append(diversity)

diversity_df = pd.DataFrame(diversity_list)
if not diversity_df.empty:
    diversity_df = diversity_df.sort_values('year')
    print("\nWin Diversity by Year (lower CV = more fair):")
    print(diversity_df[['year', 'cv', 'min_wins', 'max_wins']].to_string(index=False))

# =============================================================================
# ANALYSIS SECTION 7: SETTINGS VS FAIRNESS CORRELATION
# =============================================================================
print("\n" + "=" * 80)
print("SETTINGS VS FAIRNESS TRENDS")
print("=" * 80)

if not settings_df.empty and not fairness_df.empty:
    combined = settings_df.merge(fairness_df, on='year')
    combined = combined.merge(diversity_df[['year', 'cv']], on='year', how='left')
    
    # Add power rankings spread if available
    if not all_power_rankings_df.empty:
        power_spread = all_power_rankings_df.groupby('year')['power_score'].agg(['std']).reset_index()
        power_spread.columns = ['year', 'power_std']
        combined = combined.merge(power_spread, on='year', how='left')
    
    combined = combined.sort_values('year')
    
    print("\nYear-over-Year Summary:")
    cols = ['year', 'rec_points', 'faab', 'acquisition_budget', 'playoff_team_count', 'avg_differential', 'close_matchups_pct', 'cv', 'power_std']
    available_cols = [c for c in cols if c in combined.columns]
    print(combined[available_cols].to_string(index=False))
    
    # Identify setting changes
    print("\nSetting Changes Detected:")
    for col in ['faab', 'acquisition_budget', 'playoff_team_count', 'team_count', 'rec_points']:
        if col in combined.columns:
            changes = combined[combined[col] != combined[col].shift(1)]
            if len(changes) > 1:  # More than just first year
                print(f"\n{col}:")
                for _, row in changes.iterrows():
                    print(f"  {int(row['year'])}: {row[col]}")
    
    # Check for roster slot changes
    if 'position_slot_counts' in settings_df.columns:
        print("\nRoster Slot Changes:")
        prev_slots = None
        for idx, row in settings_df.sort_values('year').iterrows():
            try:
                slots = json.loads(row['position_slot_counts'])
                if prev_slots is not None and slots != prev_slots:
                    print(f"\n{int(row['year'])} changed from previous year:")
                    all_positions = set(list(slots.keys()) + list(prev_slots.keys()))
                    for pos in sorted(all_positions):
                        old_count = prev_slots.get(pos, 0)
                        new_count = slots.get(pos, 0)
                        if old_count != new_count:
                            print(f"  {pos}: {old_count} → {new_count}")
                prev_slots = slots
            except (json.JSONDecodeError, KeyError):
                pass
else:
    print("Insufficient data for correlation analysis")

# =============================================================================
# FINAL SUMMARY REPORT WITH VISUALIZATIONS
# =============================================================================
print("\n" + "=" * 80)
print("GENERATING FINAL SUMMARY REPORT WITH VISUALIZATIONS")
print("=" * 80)

if not settings_df.empty and not fairness_df.empty and not diversity_df.empty:
    combined = settings_df.merge(fairness_df, on='year')
    combined = combined.merge(diversity_df[['year', 'cv']], on='year', how='left')
    if not all_power_rankings_df.empty:
        power_spread = all_power_rankings_df.groupby('year')['power_score'].agg(['std']).reset_index()
        power_spread.columns = ['year', 'power_std']
        combined = combined.merge(power_spread, on='year', how='left')
    combined = combined.sort_values('year')
    
    # Create HTML report with embedded visualizations
    html_lines = []
    html_lines.append("<!DOCTYPE html>")
    html_lines.append("<html><head>")
    html_lines.append("<title>ESPN Fantasy Football League Analysis (2016-2025)</title>")
    html_lines.append("<style>")
    html_lines.append("body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }")
    html_lines.append("h1 { color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }")
    html_lines.append("h2 { color: #555; margin-top: 30px; border-bottom: 2px solid #28a745; padding-bottom: 5px; }")
    html_lines.append(".metric-card { background: white; padding: 20px; margin: 15px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }")
    html_lines.append(".stat { display: inline-block; margin: 10px 20px; }")
    html_lines.append(".stat-label { font-weight: bold; color: #666; }")
    html_lines.append(".stat-value { font-size: 1.3em; color: #007bff; }")
    html_lines.append(".min { color: #28a745; }")
    html_lines.append(".max { color: #dc3545; }")
    html_lines.append(".mean { color: #ffc107; }")
    html_lines.append("table { border-collapse: collapse; width: 100%; background: white; margin: 15px 0; }")
    html_lines.append("th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }")
    html_lines.append("th { background-color: #007bff; color: white; }")
    html_lines.append("</style>")
    html_lines.append("</head><body>")
    
    html_lines.append("<h1>ESPN Fantasy Football League Analysis (2016-2025)</h1>")
    
    # Executive Summary with Min/Max/Mean
    html_lines.append("<h2>Executive Summary: Key Metrics Over Time</h2>")
    
    metrics_to_track = [
        ('cv', 'League Competitiveness (CV)', '%', 'Lower is more competitive'),
        ('avg_differential', 'Average Score Differential', 'points', 'Lower means closer games'),
        ('close_matchups_pct', 'Close Matchups (<5 pts)', '%', 'Higher is better'),
        ('blowouts_pct', 'Blowouts (>20 pts)', '%', 'Lower is better'),
        ('power_std', 'Power Rankings Spread', 'points', 'Lower means more balanced')
    ]
    
    for metric, label, unit, interpretation in metrics_to_track:
        if metric in combined.columns:
            values = combined[metric].dropna()
            if len(values) > 0:
                min_val = values.min()
                max_val = values.max()
                mean_val = values.mean()
                min_year = int(combined.loc[values.idxmin(), 'year'])
                max_year = int(combined.loc[values.idxmax(), 'year'])
                
                html_lines.append(f"<div class='metric-card'>")
                html_lines.append(f"<h3>{label}</h3>")
                html_lines.append(f"<p><em>{interpretation}</em></p>")
                html_lines.append(f"<div class='stat'><span class='stat-label'>Min:</span> <span class='stat-value min'>{min_val:.2f} {unit}</span> (Year: {min_year})</div>")
                html_lines.append(f"<div class='stat'><span class='stat-label'>Max:</span> <span class='stat-value max'>{max_val:.2f} {unit}</span> (Year: {max_year})</div>")
                html_lines.append(f"<div class='stat'><span class='stat-label'>Mean:</span> <span class='stat-value mean'>{mean_val:.2f} {unit}</span></div>")
                html_lines.append("</div>")
    
    # Setting Changes
    html_lines.append("<h2>Major Setting Changes</h2>")
    html_lines.append("<div class='metric-card'>")
    
    for col in ['rec_points', 'acquisition_budget', 'playoff_team_count']:
        if col in settings_df.columns:
            changes = settings_df[settings_df[col] != settings_df[col].shift(1)]
            if len(changes) > 1:
                html_lines.append(f"<h3>{col.replace('_', ' ').title()}</h3>")
                html_lines.append("<ul>")
                for _, row in changes.iterrows():
                    html_lines.append(f"<li>{int(row['year'])}: {row[col]}</li>")
                html_lines.append("</ul>")
    
    html_lines.append("</div>")
    
    # Create Plotly visualizations
    print("\nGenerating interactive charts...")
    
    # Chart 1: Competitiveness Trend (CV over time)
    if 'cv' in combined.columns:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=combined['year'],
            y=combined['cv'],
            mode='lines+markers',
            name='CV',
            line=dict(color='#007bff', width=3),
            marker=dict(size=10)
        ))
        fig1.add_hline(y=combined['cv'].mean(), line_dash="dash", line_color="orange",
                      annotation_text=f"Mean: {combined['cv'].mean():.2f}%")
        fig1.update_layout(
            title='League Competitiveness Over Time (Lower = More Competitive)',
            xaxis_title='Year',
            yaxis_title='Coefficient of Variation (%)',
            template='plotly_white',
            height=500
        )
        html_lines.append("<h2>Competitiveness Trend</h2>")
        html_lines.append(fig1.to_html(include_plotlyjs='cdn', div_id='cv_chart'))
    
    # Chart 2: Matchup Fairness Metrics
    if 'avg_differential' in combined.columns and 'close_matchups_pct' in combined.columns:
        fig2 = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Average Score Differential', 'Close Matchups Percentage'),
            vertical_spacing=0.15
        )
        
        fig2.add_trace(go.Bar(
            x=combined['year'],
            y=combined['avg_differential'],
            name='Avg Differential',
            marker_color='#28a745'
        ), row=1, col=1)
        
        fig2.add_trace(go.Scatter(
            x=combined['year'],
            y=combined['close_matchups_pct'],
            mode='lines+markers',
            name='Close Matchups %',
            line=dict(color='#dc3545', width=3),
            marker=dict(size=8)
        ), row=2, col=1)
        
        fig2.update_xaxes(title_text="Year", row=2, col=1)
        fig2.update_yaxes(title_text="Points", row=1, col=1)
        fig2.update_yaxes(title_text="Percentage (%)", row=2, col=1)
        fig2.update_layout(height=700, template='plotly_white', showlegend=False)
        
        html_lines.append("<h2>Matchup Fairness Analysis</h2>")
        html_lines.append(fig2.to_html(include_plotlyjs='cdn', div_id='fairness_chart'))
    
    # Chart 3: Settings vs Competitiveness Correlation
    if 'rec_points' in combined.columns and 'cv' in combined.columns:
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig3.add_trace(go.Bar(
            x=combined['year'],
            y=combined['rec_points'],
            name='PPR Points',
            marker_color='rgba(0, 123, 255, 0.6)'
        ), secondary_y=False)
        
        fig3.add_trace(go.Scatter(
            x=combined['year'],
            y=combined['cv'],
            mode='lines+markers',
            name='CV',
            line=dict(color='#ff6b6b', width=3),
            marker=dict(size=10)
        ), secondary_y=True)
        
        fig3.update_xaxes(title_text="Year")
        fig3.update_yaxes(title_text="PPR Points per Reception", secondary_y=False)
        fig3.update_yaxes(title_text="Coefficient of Variation (%)", secondary_y=True)
        fig3.update_layout(
            title='PPR Setting vs League Competitiveness',
            template='plotly_white',
            height=500
        )
        
        html_lines.append("<h2>Settings Impact on Competitiveness</h2>")
        html_lines.append(fig3.to_html(include_plotlyjs='cdn', div_id='settings_chart'))
    
    # Chart 4: Multi-metric Dashboard
    if all(col in combined.columns for col in ['cv', 'avg_differential', 'close_matchups_pct', 'blowouts_pct']):
        fig4 = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Win Diversity (CV)', 'Avg Score Differential', 
                          'Close Matchups %', 'Blowouts %')
        )
        
        fig4.add_trace(go.Scatter(x=combined['year'], y=combined['cv'], 
                                 mode='lines+markers', name='CV',
                                 line=dict(color='#007bff')), row=1, col=1)
        
        fig4.add_trace(go.Scatter(x=combined['year'], y=combined['avg_differential'],
                                 mode='lines+markers', name='Avg Diff',
                                 line=dict(color='#28a745')), row=1, col=2)
        
        fig4.add_trace(go.Scatter(x=combined['year'], y=combined['close_matchups_pct'],
                                 mode='lines+markers', name='Close %',
                                 line=dict(color='#ffc107')), row=2, col=1)
        
        fig4.add_trace(go.Scatter(x=combined['year'], y=combined['blowouts_pct'],
                                 mode='lines+markers', name='Blowouts %',
                                 line=dict(color='#dc3545')), row=2, col=2)
        
        fig4.update_layout(height=700, template='plotly_white', showlegend=False)
        
        html_lines.append("<h2>Multi-Metric Dashboard</h2>")
        html_lines.append(fig4.to_html(include_plotlyjs='cdn', div_id='dashboard'))
    
    # Year-by-year table
    html_lines.append("<h2>Year-by-Year Detailed Metrics</h2>")
    html_lines.append("<div class='metric-card'>")
    cols = ['year', 'rec_points', 'acquisition_budget', 'avg_differential', 'close_matchups_pct', 'blowouts_pct', 'cv', 'power_std']
    available_cols = [c for c in cols if c in combined.columns]
    html_lines.append(combined[available_cols].to_html(index=False, border=0))
    html_lines.append("</div>")
    
    html_lines.append("</body></html>")
    
    # Save HTML report
    with open('league_analysis_summary.html', 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_lines))
    
    print("\n✓ Interactive HTML report saved to league_analysis_summary.html")
    
    # Also save CSV
    combined.to_csv('league_analysis_summary.csv', index=False)
    print("✓ Summary data saved to league_analysis_summary.csv")
else:
    print("Insufficient data for summary report")

# =============================================================================
# CSV EXPORT
# =============================================================================
print("\n" + "=" * 80)
print("CSV EXPORT")
print("=" * 80)

def export_data_to_csv():
    """Export analysis data to CSV files"""
    for year, df in matchups_df_by_year.items():
        if not df.empty:
            df.to_csv(f'matchups_{year}.csv', index=False)
    
    for year, df in roster_df_by_year.items():
        if not df.empty:
            df.to_csv(f'roster_{year}.csv', index=False)
    
    for year, df in power_rankings_df_by_year.items():
        if not df.empty:
            df.to_csv(f'power_rankings_{year}.csv', index=False)
    
    if not settings_df.empty:
        settings_df.to_csv('settings_all_years.csv', index=False)
    
    if not fairness_df.empty:
        fairness_df.to_csv('fairness_all_years.csv', index=False)
    
    if not diversity_df.empty:
        diversity_df.to_csv('diversity_all_years.csv', index=False)
    
    if not all_power_rankings_df.empty:
        all_power_rankings_df.to_csv('power_rankings_all_years.csv', index=False)
    
    print("\n✓ All data exported to CSV files")

export_data_to_csv()
