#!/usr/bin/env python3
"""
ESPN Fantasy Football League Comparison Tool (2016-2025)
Analyzes matchups, rosters, settings, and fairness metrics across multiple years.
Run this script to compare league data and correlate settings changes with fairness.
"""

import pandas as pd
import json
from collections import defaultdict
from espn_api.football import League
from report_charts import (
    make_stacked_roster_chart,
    make_competitiveness_chart,
    make_matchup_fairness_chart,
    make_close_vs_blowouts_chart,
    make_settings_vs_competitiveness_chart,
    make_multi_metric_dashboard,
    make_recent_comp_chart,
    make_finish_std_chart,
)

# =============================================================================
# CONFIGURATION
# =============================================================================
ESPN_S2 = 'AEBvr5lKQnBF4qZ6iAb0dQxlVe%2FFjmDvoodlVueyffeuuWHthnUO5U8URkyOro9P95i5hHi7uVO8rsMDwEfrhM1ocAAbtLKAS2DWchuOib5sE5%2FVGoas05b1tnifeeCOZ0Q5ANyjF3JLYRdrq39OR1a%2BucdWXw7XZELwdX4a7bacXJwa%2FD1izL1Y%2F1KZ%2BxjkC%2BC1at7XtQsf7HHxll8m1WcrclJUtEFNZKoSDRu6alVqBzaf3x6c7WtEzaOplUbSFNhklgDzmJjrK9HHvGnzKmcvJ8I4hhCtm9fNRuIAyu9R2eeMvuyyjxYMRQgw%2F5YkW3twSXx0uVXA3VEl4Zn%2BVV%2FR'
SWID = '{9A39BAD6-0F58-4406-B9BA-D60F587406D2}'
LEAGUE_ID = 247704
YEARS = range(2015, 2026)  # 2016-2025 (range end is exclusive)

# =============================================================================
# LOAD LEAGUES
# =============================================================================
print("=" * 80)
print("Loading leagues (2015-2025)...")
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


def _normalize_owner_name(owner_entry):
    """Return a hashable owner name from various ESPN owner representations."""
    if owner_entry is None:
        return None
    # If already a string or int, return as string
    if isinstance(owner_entry, (str, int)):
        return str(owner_entry)
    # If dict-like with name fields
    if isinstance(owner_entry, dict):
        # Prefer displayName when present, otherwise fallback sequence
        if owner_entry.get('displayName'):
            return str(owner_entry.get('displayName'))
        for key in ['nickname', 'username', 'name']:
            val = owner_entry.get(key)
            if val:
                return str(val)
        # Fallback to first/last if present
        parts = []
        for key in ['firstName', 'lastName', 'fullName']:
            val = owner_entry.get(key)
            if val:
                parts.append(str(val))
        if parts:
            return ' '.join(parts)
        return str(owner_entry)
    # Fallback: string representation
    return str(owner_entry)


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
    should_refetch = False
    
    try:
        df = pd.read_csv(f'roster_{year}.csv')
        # Check if CSV has expected columns
        if df.empty or ('player' not in df.columns and 'player_name' not in df.columns):
            print(f"⚠ roster_{year}.csv is empty or missing expected columns, refetching...")
            should_refetch = True
        else:
            # Normalize column names - handle both 'player' and 'player_name'
            if 'player_name' in df.columns and 'player' not in df.columns:
                df = df.rename(columns={'player_name': 'player'})
            print(f"✓ Loaded roster_{year}.csv")
    except FileNotFoundError:
        should_refetch = True
    
    if should_refetch and leagues.get(year):
        df = analyze_roster_composition(leagues[year], year)
        df.to_csv(f'roster_{year}.csv', index=False)
        print(f"✓ Fetched and saved roster_{year}.csv")
    elif should_refetch:
        print(f"✗ Skipping {year} roster analysis (no league available)")
    
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
    
    # Calculate actual year range from data
    min_year = int(combined['year'].min()) if not combined.empty else 2016
    max_year = int(combined['year'].max()) if not combined.empty else 2025
    year_range_str = f"{min_year}-{max_year}"
    
    # Create HTML report with embedded visualizations
    html_lines = []
    html_lines.append("<!DOCTYPE html>")
    html_lines.append("<html><head>")
    html_lines.append(f"<title>ESPN Fantasy Football League Analysis ({year_range_str})</title>")
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
    
    html_lines.append(f"<h1>ESPN Fantasy Football League Analysis ({year_range_str})</h1>")
    
    # Quick Stats Overview
    html_lines.append("<div class='metric-card'>")
    html_lines.append("<h2>League Overview</h2>")
    total_matchups = len(all_matchups_df) if not all_matchups_df.empty else 0
    total_seasons = len(combined)
    html_lines.append(f"<p><strong>Total Seasons Analyzed:</strong> {total_seasons} seasons ({year_range_str})</p>")
    html_lines.append(f"<p><strong>Total Regular Season Matchups:</strong> {total_matchups}</p>")
    if not settings_df.empty:
        team_counts = settings_df['team_count'].value_counts().to_dict()
        format_summary = ', '.join([f"{count} teams ({years} seasons)" for count, years in sorted(team_counts.items())])
        html_lines.append(f"<p><strong>League Formats:</strong> {format_summary}</p>")
    html_lines.append("</div>")
    
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
                
                html_lines.append("<div class='metric-card'>")
                html_lines.append(f"<h3>{label}</h3>")
                html_lines.append(f"<p><em>{interpretation}</em></p>")
                html_lines.append(f"<div class='stat'><span class='stat-label'>Min:</span> <span class='stat-value min'>{min_val:.2f} {unit}</span> (Year: {min_year})</div>")
                html_lines.append(f"<div class='stat'><span class='stat-label'>Max:</span> <span class='stat-value max'>{max_val:.2f} {unit}</span> (Year: {max_year})</div>")
                html_lines.append(f"<div class='stat'><span class='stat-label'>Mean:</span> <span class='stat-value mean'>{mean_val:.2f} {unit}</span></div>")
                html_lines.append("</div>")
    
    # Roster Composition Analysis
    html_lines.append("<h2>Roster Composition Analysis</h2>")
    html_lines.append("<div class='metric-card'>")
    
    if not all_rosters_df.empty:
        # Position distribution over time
        position_by_year = all_rosters_df.groupby(['year', 'position']).size().reset_index(name='count')
        print("[DEBUG] position_by_year head:")
        print(position_by_year.head())
        html_lines.append("<h3>Player Pool Size by Year</h3>")
        player_counts = all_rosters_df.groupby('year')['player'].nunique()
        print("[DEBUG] player_counts:")
        print(player_counts)
        html_lines.append("<table>")
        html_lines.append("<tr><th>Year</th><th>Unique Players</th></tr>")
        for year, count in player_counts.items():
            html_lines.append(f"<tr><td>{int(year)}</td><td>{count}</td></tr>")
        html_lines.append("</table>")

        html_lines.append("<h3>Position Distribution Trends</h3>")
        html_lines.append("<p><em>Shows how roster composition has evolved over time</em></p>")

        # Create position distribution chart
        try:
            pivot_positions = position_by_year.pivot(index='year', columns='position', values='count').fillna(0)
            print("[DEBUG] pivot_positions head:")
            print(pivot_positions.head())
            chart_html = make_stacked_roster_chart(
                pivot_positions,
                title='Position Distribution by Year (Stacked)',
                div_id='roster_chart',
                height=500,
            )
            if chart_html:
                html_lines.append(chart_html)
            else:
                html_lines.append("<p><em>Position distribution data is empty. Check roster CSVs and extraction logic.</em></p>")
        except Exception as e:
            print(f"[ERROR] Failed to generate roster chart: {e}")
            html_lines.append(f"<p><em>Error generating roster chart: {e}</em></p>")
    else:
        print("[ERROR] all_rosters_df is empty. No roster data available.")
        html_lines.append("<p><em>No roster data available</em></p>")
    
    # 14-Team Seasons Roster Breakdown
    if not all_rosters_df.empty and 'team_count' in settings_df.columns:
        fourteen_years = settings_df[settings_df['team_count'] == 14]['year'].tolist()
        roster_14_df = all_rosters_df[all_rosters_df['year'].isin(fourteen_years)]
        # Drop years with zero players (likely missing/ongoing data)
        player_counts_14 = roster_14_df.groupby('year')['player'].nunique()
        valid_years_14 = player_counts_14[player_counts_14 > 0].index.tolist()
        roster_14_df = roster_14_df[roster_14_df['year'].isin(valid_years_14)]
        if not roster_14_df.empty:
            html_lines.append("<h3>14-Team Seasons: Roster Composition</h3>")
            position_by_year_14 = roster_14_df.groupby(['year', 'position']).size().reset_index(name='count')
            print("[DEBUG] 14-team position_by_year head:")
            print(position_by_year_14.head())
            player_counts_14 = roster_14_df.groupby('year')['player'].nunique()
            print("[DEBUG] 14-team player_counts:")
            print(player_counts_14)
            html_lines.append("<table>")
            html_lines.append("<tr><th>Year</th><th>Unique Players</th></tr>")
            for year, count in player_counts_14.items():
                html_lines.append(f"<tr><td>{int(year)}</td><td>{count}</td></tr>")
            html_lines.append("</table>")
            if len(player_counts_14) < 2:
                html_lines.append("<p><em>Only one 14-team season with roster data is available; stacked trend will plot as single points.</em></p>")

            try:
                pivot_positions_14 = position_by_year_14.pivot(index='year', columns='position', values='count').fillna(0)
                print("[DEBUG] 14-team pivot_positions head:")
                print(pivot_positions_14.head())
                chart_html = make_stacked_roster_chart(
                    pivot_positions_14,
                    title='14-Team Seasons: Position Distribution (Stacked)',
                    div_id='roster_chart_14',
                    height=450,
                )
                if chart_html:
                    html_lines.append(chart_html)
                else:
                    html_lines.append("<p><em>14-team position distribution is empty. Verify roster data for those seasons.</em></p>")
            except Exception as e:
                print(f"[ERROR] Failed to generate 14-team roster chart: {e}")
                html_lines.append(f"<p><em>Error generating 14-team roster chart: {e}</em></p>")
        else:
            html_lines.append("<p><em>No roster data for 14-team seasons.</em></p>")
    
    
    html_lines.append("</div>")

    # Team finish diversity by team_id
    html_lines.append("<h2>Team Finish Diversity (by Team ID)</h2>")
    html_lines.append("<div class='metric-card'>")
    team_finishes = []
    for year, league in leagues.items():
        if league:
            for team in league.teams:
                finish = getattr(team, 'final_standing', None)
                if finish is None:
                    finish = getattr(team, 'standing', None)
                # Prefer owner/manager name; fallback to team name
                owner_list = getattr(team, 'owners', None)
                owner_single = getattr(team, 'owner', None)
                owner_name = None
                if owner_list:
                    owner_name = _normalize_owner_name(owner_list[0])
                elif owner_single:
                    owner_name = _normalize_owner_name(owner_single)
                else:
                    owner_name = team.team_name
                # Exclude ongoing seasons where finish is 0
                if finish is not None and finish != 0 and owner_name is not None:
                    team_finishes.append({
                        'year': year,
                        'owner': owner_name,
                        'team_name': team.team_name,
                        'final_standing': finish
                    })
    team_finishes_df = pd.DataFrame(team_finishes)
    if not team_finishes_df.empty:
        finish_diversity = team_finishes_df.groupby('owner')['final_standing'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
        finish_diversity = finish_diversity.rename(columns={
            'owner': 'Owner',
            'mean': 'Avg Finish',
            'std': 'Finish Std',
            'min': 'Best',
            'max': 'Worst',
            'count': 'Seasons'
        })
        # Round numeric values for readability
        for col in ['Avg Finish', 'Finish Std']:
            finish_diversity[col] = finish_diversity[col].round(2)
        html_lines.append(finish_diversity.to_html(index=False, border=0))

        chart_html = make_finish_std_chart(finish_diversity)
        if chart_html:
            html_lines.append(chart_html)
    else:
        html_lines.append("<p><em>No team finish data available.</em></p>")
    html_lines.append("</div>")
    
    # Create Plotly visualizations
    print("\nGenerating interactive charts...")
    
    # Chart 1: Competitiveness Trend (CV over time)
    chart_html = make_competitiveness_chart(combined)
    if chart_html:
        html_lines.append("<h2>Competitiveness Trend</h2>")
        html_lines.append("<p><strong>Interpretation:</strong> Lower values indicate more competitive balance (wins spread across teams). Higher values suggest domination by fewer teams.</p>")
        html_lines.append(chart_html)
    
    # Chart 2: Matchup Fairness Metrics
    chart_html = make_matchup_fairness_chart(combined)
    if chart_html:
        html_lines.append("<h2>Matchup Fairness Analysis</h2>")
        html_lines.append("<p><strong>Interpretation:</strong> Lower score differentials and higher percentages of close matchups indicate more competitive/exciting games.</p>")
        html_lines.append(chart_html)
    
    # Chart 2b: Close vs Blowouts Trend
    chart_html = make_close_vs_blowouts_chart(combined)
    if chart_html:
        html_lines.append("<h2>Close vs Blowouts</h2>")
        html_lines.append("<p><strong>Interpretation:</strong> Higher close-matchup percentage with lower blowout percentage indicates healthier balance.</p>")
        html_lines.append(chart_html)
    
    # Chart 3: Settings vs Competitiveness Correlation
    chart_html = make_settings_vs_competitiveness_chart(combined)
    if chart_html:
        html_lines.append("<h2>Settings Impact on Competitiveness</h2>")
        html_lines.append("<p><strong>Interpretation:</strong> Compare PPR changes (bars) with CV trend (line) to see if scoring rule changes affected competitive balance.</p>")
        html_lines.append(chart_html)
    
    # Chart 4: Multi-metric Dashboard
    chart_html = make_multi_metric_dashboard(combined)
    if chart_html:
        html_lines.append("<h2>Multi-Metric Dashboard</h2>")
        html_lines.append("<p><strong>Interpretation:</strong> Compare all key metrics across years to identify patterns and outliers.</p>")
        html_lines.append(chart_html)
    
    # Recent Trends Analysis (2023-2025)
    html_lines.append("<h2>Recent Trends (2023-2025 Deep Dive)</h2>")
    html_lines.append("<div class='metric-card'>")
    
    recent_years = combined[combined['year'] >= 2023].copy()
    if len(recent_years) > 0:
        html_lines.append("<h3>Recent Performance Summary</h3>")
        html_lines.append("<p><em>Focused analysis on the most recent seasons</em></p>")
        
        for metric, label, unit, better in [
            ('cv', 'Competitiveness (CV)', '%', 'lower'),
            ('avg_differential', 'Average Score Differential', 'points', 'lower'),
            ('close_matchups_pct', 'Close Matchups', '%', 'higher'),
            ('power_std', 'Power Rankings Spread', 'points', 'lower')
        ]:
            if metric in recent_years.columns:
                values = recent_years[metric].dropna()
                if len(values) > 0:
                    trend = "improving" if (values.iloc[-1] < values.iloc[0]) == (better == 'lower') else "declining"
                    change = values.iloc[-1] - values.iloc[0]
                    html_lines.append(f"<p><strong>{label}:</strong> ")
                    html_lines.append(f"{values.iloc[0]:.2f} ({int(recent_years.iloc[0]['year'])}) → ")
                    html_lines.append(f"{values.iloc[-1]:.2f} ({int(recent_years.iloc[-1]['year'])}) ")
                    html_lines.append(f"<span style='color: {'green' if trend == 'improving' else 'red'};'>({change:+.2f} - {trend})</span></p>")
        
        chart_html = make_recent_comp_chart(recent_years)
        if chart_html:
            html_lines.append(chart_html)
    
    html_lines.append("</div>")
    
    # Year-over-Year Changes Analysis
    html_lines.append("<h2>Year-over-Year Trend Analysis</h2>")
    html_lines.append("<div class='metric-card'>")
    html_lines.append("<h3>Biggest Changes Between Seasons</h3>")
    if len(combined) > 1:
        changes = combined.copy()
        for metric in ['cv', 'avg_differential', 'close_matchups_pct', 'blowouts_pct']:
            if metric in changes.columns:
                changes[f'{metric}_change'] = changes[metric].diff()
        
        # Find biggest improvements/declines
        if 'cv_change' in changes.columns:
            max_improvement = changes.loc[changes['cv_change'].idxmin()] if changes['cv_change'].notna().any() else None
            max_decline = changes.loc[changes['cv_change'].idxmax()] if changes['cv_change'].notna().any() else None
            if max_improvement is not None:
                html_lines.append(f"<p><strong>Most Competitive Improvement:</strong> {int(max_improvement['year'])} (CV decreased by {abs(max_improvement['cv_change']):.2f}%)</p>")
            if max_decline is not None:
                html_lines.append(f"<p><strong>Biggest Competitive Decline:</strong> {int(max_decline['year'])} (CV increased by {max_decline['cv_change']:.2f}%)</p>")
        
        if 'close_matchups_pct_change' in changes.columns:
            best_close = changes.loc[changes['close_matchups_pct_change'].idxmax()] if changes['close_matchups_pct_change'].notna().any() else None
            if best_close is not None and best_close['close_matchups_pct_change'] > 0:
                html_lines.append(f"<p><strong>Best Year for Close Games:</strong> {int(best_close['year'])} (+{best_close['close_matchups_pct_change']:.2f}% close matchups vs prior year)</p>")
    html_lines.append("</div>")
    
    # Key Takeaways
    html_lines.append("<h2>Key Insights & Recommendations</h2>")
    html_lines.append("<div class='metric-card'>")
    html_lines.append("<h3>League Health Summary</h3>")
    
    # Competitive balance insight
    if 'cv' in combined.columns:
        recent_cv = combined[combined['year'] >= 2023]['cv'].mean() if len(combined[combined['year'] >= 2023]) > 0 else None
        historical_cv = combined[combined['year'] < 2023]['cv'].mean() if len(combined[combined['year'] < 2023]) > 0 else None
        if recent_cv and historical_cv:
            if recent_cv < historical_cv:
                html_lines.append(f"<p>✅ <strong>Competitive Balance:</strong> Recent seasons ({recent_cv:.1f}% CV) are MORE competitive than historical average ({historical_cv:.1f}% CV)</p>")
            else:
                html_lines.append(f"<p>⚠️ <strong>Competitive Balance:</strong> Recent seasons ({recent_cv:.1f}% CV) are LESS competitive than historical average ({historical_cv:.1f}% CV)</p>")
    
    # Blowout trend
    if 'blowouts_pct' in combined.columns:
        recent_blowouts = combined[combined['year'] >= 2023]['blowouts_pct'].mean() if len(combined[combined['year'] >= 2023]) > 0 else None
        if recent_blowouts:
            if recent_blowouts > 60:
                html_lines.append(f"<p>⚠️ <strong>Matchup Quality:</strong> High blowout rate in recent seasons ({recent_blowouts:.1f}%). Consider roster/scoring adjustments.</p>")
            elif recent_blowouts < 50:
                html_lines.append(f"<p>✅ <strong>Matchup Quality:</strong> Good balance with {recent_blowouts:.1f}% blowouts in recent seasons.</p>")
    
    # Team count impact
    if not settings_df.empty and 'team_count' in settings_df.columns:
        team_count_years = settings_df.groupby('team_count')['year'].apply(list).to_dict()
        if len(team_count_years) > 1:
            html_lines.append("<h3>League Format Changes</h3>")
            for team_count, years in sorted(team_count_years.items()):
                year_list = ', '.join([str(int(y)) for y in sorted(years)])
                html_lines.append(f"<p><strong>{team_count}-Team Format:</strong> {year_list}</p>")
    
    html_lines.append("</div>")
    
    # Year-by-year table (readable labels)
    html_lines.append("<h2>Year-by-Year Detailed Metrics</h2>")
    html_lines.append("<div class='metric-card'>")
    cols = [
        'year', 'team_count', 'playoff_team_count', 'rec_points',
        'avg_differential', 'close_matchups_pct', 'blowouts_pct', 'cv', 'power_std'
    ]
    available_cols = [c for c in cols if c in combined.columns]
    display_df = combined[available_cols].copy()
    rename_map = {
        'year': 'Year',
        'team_count': 'Teams',
        'playoff_team_count': 'Playoff Teams',
        'rec_points': 'PPR pts/rec',
        'avg_differential': 'Avg Diff (pts)',
        'close_matchups_pct': 'Close % (<5)',
        'blowouts_pct': 'Blowouts % (>20)',
        'cv': 'Win CV (%)',
        'power_std': 'Power Std'
    }
    display_df = display_df.rename(columns=rename_map)
    # Round numeric columns to keep the table readable
    for col in display_df.columns:
        if col != 'Year':
            display_df[col] = display_df[col].apply(lambda v: f"{v:.2f}" if isinstance(v, (int, float, float)) else v)
    html_lines.append(display_df.to_html(index=False, border=0))
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
