"""
Factory Analytics Dashboard
Combines Operator Duration Analysis and Efficiency Calculator into one unified interface
"""

import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import timedelta
import os
import re
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

# Settings file path for persistence
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard_settings.json')


class DataManager:
    """Shared data manager for both tools"""
    
    def __init__(self):
        self.df = None
        self.df_filtered = None
        self.file_path = None
        self.machines = []
        self.operators = []
        self.shifts = []
        self.target_times = {}  # Shared target times for all stations
        
        # Centralized filter settings (global defaults)
        self.filter_min_seconds = 3.0  # Default: filter out < 3 seconds
        self.filter_max_seconds = 1800.0  # Default: filter out > 30 minutes
        self.filters_enabled = True
        
        # Per-cell duration filters: {cell_num: {'min': float, 'max': float}}
        # If a cell has a custom filter, it overrides the global filter for that cell
        self.cell_duration_filters = {}
        
        # Advanced filter settings
        self.filter_date_start = None
        self.filter_date_end = None
        self.filter_shifts = []  # Empty = all shifts
        self.filter_operators = []  # Empty = all operators
        self.filter_machines = []  # Empty = all machines
        self.filter_cells = []  # Empty = all cells
        
        # Operator filtering
        self.filter_min_records = 0  # Minimum records per operator (0 = no minimum)
        self.excluded_operators = []  # List of operators to exclude
        
        # Per-station duration filters: {station_name: {'min': float, 'max': float}}
        self.station_duration_filters = {}
        
        # Last station per cell for quantity calculation
        self.last_stations = {}  # {cell_num: station_name}
        
        # Load saved settings
        self._load_settings()
    
    def _load_settings(self):
        """Load saved settings from JSON file"""
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, 'r') as f:
                    settings = json.load(f)
                
                # Load target times
                self.target_times = settings.get('target_times', {})
                
                # Load cell duration filters
                self.cell_duration_filters = settings.get('cell_duration_filters', {})
                
                # Load station duration filters
                self.station_duration_filters = settings.get('station_duration_filters', {})
                
                # Load global filter settings
                self.filter_min_seconds = settings.get('filter_min_seconds', 3.0)
                self.filter_max_seconds = settings.get('filter_max_seconds', 1800.0)
                self.filter_min_records = settings.get('filter_min_records', 0)
                self.excluded_operators = settings.get('excluded_operators', [])
                
                # Load last stations
                self.last_stations = settings.get('last_stations', {})
        except Exception as e:
            print(f"Could not load settings: {e}")
    
    def save_settings(self):
        """Save current settings to JSON file"""
        try:
            settings = {
                'target_times': self.target_times,
                'cell_duration_filters': self.cell_duration_filters,
                'station_duration_filters': self.station_duration_filters,
                'filter_min_seconds': self.filter_min_seconds,
                'filter_max_seconds': self.filter_max_seconds,
                'filter_min_records': self.filter_min_records,
                'excluded_operators': self.excluded_operators,
                'last_stations': self.last_stations,
            }
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"Could not save settings: {e}")
    
    def get_target_time(self, station):
        """Get target time for a station, default 30 if not set"""
        return self.target_times.get(station, 30.0)
    
    def set_target_time(self, station, value):
        """Set target time for a station"""
        self.target_times[station] = float(value)
    
    def set_duration_filters(self, min_seconds, max_seconds, enabled=True):
        """Set the global duration filter settings (applies to cells without custom filters)"""
        self.filter_min_seconds = float(min_seconds)
        self.filter_max_seconds = float(max_seconds)
        self.filters_enabled = enabled
        self._apply_filters()
    
    def set_cell_duration_filter(self, cell_num, min_seconds, max_seconds):
        """Set duration filter for a specific cell"""
        self.cell_duration_filters[cell_num] = {
            'min': float(min_seconds),
            'max': float(max_seconds)
        }
        self._apply_filters()
    
    def get_cell_duration_filter(self, cell_num):
        """Get duration filter for a specific cell, or global defaults if not set"""
        if cell_num in self.cell_duration_filters:
            return self.cell_duration_filters[cell_num]
        return {'min': self.filter_min_seconds, 'max': self.filter_max_seconds}
    
    def clear_cell_duration_filter(self, cell_num):
        """Remove custom duration filter for a cell (will use global defaults)"""
        if cell_num in self.cell_duration_filters:
            del self.cell_duration_filters[cell_num]
            self._apply_filters()
    
    def clear_all_cell_duration_filters(self):
        """Clear all per-cell duration filters"""
        self.cell_duration_filters = {}
        self._apply_filters()
    
    def set_advanced_filters(self, date_start=None, date_end=None, shifts=None, 
                              operators=None, machines=None, cells=None):
        """Set advanced filter settings"""
        self.filter_date_start = date_start
        self.filter_date_end = date_end
        self.filter_shifts = shifts or []
        self.filter_operators = operators or []
        self.filter_machines = machines or []
        self.filter_cells = cells or []
        self._apply_filters()
    
    def _apply_filters(self):
        """Apply current filter settings to create df_filtered"""
        if self.df is None:
            return
        
        df = self.df.copy()
        
        if self.filters_enabled:
            # Duration filter - priority: station > cell > global
            df['_Cell'] = df['Machine Name'].apply(self.extract_cell_number)
            
            def check_duration(row):
                machine = row['Machine Name']
                cell = row['_Cell']
                duration = row['Duration_Seconds']
                
                # Check station-specific filter first (highest priority)
                if machine in self.station_duration_filters:
                    f = self.station_duration_filters[machine]
                    return f['min'] <= duration <= f['max']
                # Check cell-specific filter
                elif cell and cell in self.cell_duration_filters:
                    f = self.cell_duration_filters[cell]
                    return f['min'] <= duration <= f['max']
                # Use global filter
                else:
                    return self.filter_min_seconds <= duration <= self.filter_max_seconds
            
            mask = df.apply(check_duration, axis=1)
            df = df[mask]
            df = df.drop('_Cell', axis=1)
            
            # Date range filter
            if self.filter_date_start and 'ParsedDate' in df.columns:
                df = df[df['ParsedDate'] >= pd.Timestamp(self.filter_date_start)]
            if self.filter_date_end and 'ParsedDate' in df.columns:
                df = df[df['ParsedDate'] <= pd.Timestamp(self.filter_date_end) + pd.Timedelta(days=1)]
            
            # Shift filter
            if self.filter_shifts and 'Shift' in df.columns:
                df = df[df['Shift'].isin(self.filter_shifts)]
            
            # Operator filter
            if self.filter_operators:
                df = df[df['Operator Name'].isin(self.filter_operators)]
            
            # Machine filter
            if self.filter_machines:
                df = df[df['Machine Name'].isin(self.filter_machines)]
            
            # Cell filter - filter machines that belong to selected cells
            if self.filter_cells:
                cell_machines = []
                for machine in df['Machine Name'].unique():
                    cell = self.extract_cell_number(machine)
                    if cell in self.filter_cells:
                        cell_machines.append(machine)
                if cell_machines:
                    df = df[df['Machine Name'].isin(cell_machines)]
            
            # Excluded operators filter
            if self.excluded_operators:
                df = df[~df['Operator Name'].isin(self.excluded_operators)]
            
            # Minimum records filter - exclude operators with fewer records than threshold
            if self.filter_min_records > 0:
                op_counts = df['Operator Name'].value_counts()
                valid_operators = op_counts[op_counts >= self.filter_min_records].index.tolist()
                df = df[df['Operator Name'].isin(valid_operators)]
        
        self.df_filtered = df
    
    def get_filtered_df(self):
        """Get the filtered dataframe (applies filters if not already applied)"""
        if self.df_filtered is None and self.df is not None:
            self._apply_filters()
        return self.df_filtered
    
    def get_filter_stats(self):
        """Get statistics about current filters"""
        if self.df is None:
            return None
        
        total = len(self.df)
        filtered = len(self.df_filtered) if self.df_filtered is not None else total
        removed = total - filtered
        
        return {
            'total': total,
            'filtered': filtered,
            'removed': removed,
            'min_seconds': self.filter_min_seconds,
            'max_seconds': self.filter_max_seconds,
            'enabled': self.filters_enabled
        }
    
    # ============================================================
    # CENTRALIZED METRICS CALCULATIONS
    # All tabs should use these methods for consistency
    # ============================================================
    
    def calc_efficiency(self, avg_duration, target_time):
        """
        Calculate efficiency from average duration and target time.
        Formula: (target_time / avg_duration) * 100
        
        Returns: efficiency percentage (100 = exactly on target)
        """
        if avg_duration <= 0 or target_time <= 0:
            return 0
        return (target_time / avg_duration) * 100
    
    def calc_efficiency_for_durations(self, durations, target_time):
        """
        Calculate efficiency from a series of durations and target time.
        Uses average duration for calculation.
        
        Args:
            durations: pandas Series or list of duration values in seconds
            target_time: target time in seconds
            
        Returns: efficiency percentage
        """
        if durations is None or len(durations) == 0:
            return 0
        valid = [d for d in durations if d > 0]
        if not valid:
            return 0
        avg_duration = sum(valid) / len(valid)
        return self.calc_efficiency(avg_duration, target_time)
    
    def calc_weighted_efficiency(self, data_by_machine):
        """
        Calculate weighted efficiency across multiple machines.
        Each machine's contribution is weighted by number of records.
        
        Args:
            data_by_machine: dict of {machine: {'durations': [...], 'target': float}}
            
        Returns: weighted efficiency percentage
        """
        total_target_time = 0
        total_duration = 0
        
        for machine, data in data_by_machine.items():
            durations = [d for d in data['durations'] if d > 0]
            if durations:
                target = data['target']
                total_target_time += target * len(durations)
                total_duration += sum(durations)
        
        if total_duration <= 0:
            return 0
        return (total_target_time / total_duration) * 100
    
    def get_station_efficiency(self, station, df=None):
        """
        Get efficiency metrics for a single station.
        
        Returns: dict with efficiency, avg_duration, records, etc.
        """
        if df is None:
            df = self.get_filtered_df()
        if df is None:
            return None
        
        station_data = df[df['Machine Name'] == station]
        if len(station_data) == 0:
            return None
        
        valid_durations = station_data['Duration_Seconds'].dropna()
        valid_durations = valid_durations[valid_durations > 0]
        
        if len(valid_durations) == 0:
            return None
        
        target = self.get_target_time(station)
        avg_duration = valid_durations.mean()
        efficiency = self.calc_efficiency(avg_duration, target)
        
        return {
            'station': station,
            'target': target,
            'avg_duration': avg_duration,
            'efficiency': efficiency,
            'records': len(valid_durations),
            'operators': station_data['Operator Name'].nunique()
        }
    
    def get_operator_efficiency(self, operator, df=None):
        """
        Get efficiency metrics for a single operator across all their machines.
        Uses weighted average.
        
        Returns: dict with efficiency, avg_duration, records, etc.
        """
        if df is None:
            df = self.get_filtered_df()
        if df is None:
            return None
        
        op_data = df[df['Operator Name'] == operator]
        if len(op_data) == 0:
            return None
        
        # Build data by machine for weighted calculation
        data_by_machine = {}
        for machine in op_data['Machine Name'].unique():
            m_data = op_data[op_data['Machine Name'] == machine]
            valid_dur = m_data['Duration_Seconds'][m_data['Duration_Seconds'] > 0].tolist()
            if valid_dur:
                data_by_machine[machine] = {
                    'durations': valid_dur,
                    'target': self.get_target_time(machine)
                }
        
        if not data_by_machine:
            return None
        
        efficiency = self.calc_weighted_efficiency(data_by_machine)
        avg_duration = op_data['Duration_Seconds'][op_data['Duration_Seconds'] > 0].mean()
        
        return {
            'operator': operator,
            'efficiency': efficiency,
            'avg_duration': avg_duration,
            'records': len(op_data[op_data['Duration_Seconds'] > 0]),
            'machines': len(data_by_machine)
        }
    
    def get_operator_station_efficiency(self, operator, station, df=None):
        """
        Get efficiency for a specific operator at a specific station.
        
        Returns: dict with efficiency, avg_duration, records
        """
        if df is None:
            df = self.get_filtered_df()
        if df is None:
            return None
        
        data = df[(df['Operator Name'] == operator) & (df['Machine Name'] == station)]
        if len(data) == 0:
            return None
        
        valid_durations = data['Duration_Seconds'].dropna()
        valid_durations = valid_durations[valid_durations > 0]
        
        if len(valid_durations) == 0:
            return None
        
        target = self.get_target_time(station)
        avg_duration = valid_durations.mean()
        efficiency = self.calc_efficiency(avg_duration, target)
        
        return {
            'operator': operator,
            'station': station,
            'target': target,
            'avg_duration': avg_duration,
            'efficiency': efficiency,
            'records': len(valid_durations)
        }
    
    # ============================================================
    
    def extract_cell_number(self, station_name):
        """
        Extract cell identifier from station name.
        Handles multiple formats:
        - '221 W01 Press' → '221' (cell number is first numeric word)
        - 'ST6.TB FOA' → 'Line' (ST# format = stations within one line/cell)
        - 'OP20_InspectA' → 'Line' (OP# format = stations within one line/cell)
        - 'Thatcher' → 'Other' (standalone name)
        """
        if not station_name:
            return None
        
        name = str(station_name).strip()
        
        # Format 1: First word is all digits (e.g., "221 W01 Press")
        # This indicates a cell number followed by station info
        parts = name.split()
        if parts and parts[0].isdigit():
            return parts[0]
        
        # Format 2: Station prefix "ST#." (e.g., "ST6.TB FOA", "ST8.Door")
        # These are station numbers within a single production line/cell
        if re.match(r'^ST\d+\.', name):
            return 'Line'
        
        # Format 3: Operation prefix "OP#_" (e.g., "OP20_InspectA")
        # These are operation numbers within a single line/cell
        if re.match(r'^OP\d+[_\.]', name):
            return 'Line'
        
        # Format 4: First word is cell number with letters (e.g., "221A W01")
        if parts and len(parts) > 1:
            first = parts[0]
            # Check if it's mostly numeric with optional letter suffix
            if re.match(r'^\d+[A-Za-z]?$', first):
                return re.match(r'^(\d+)', first).group(1)
        
        # Fallback: group standalone names under "Other"
        return 'Other'
    
    def get_cells_and_stations(self):
        """Group stations by their cell number"""
        cells = {}
        for machine in self.machines:
            cell_num = self.extract_cell_number(machine)
            if cell_num:
                if cell_num not in cells:
                    cells[cell_num] = []
                cells[cell_num].append(machine)
        return cells
        
    def load_data(self, file_path: str) -> bool:
        """Load data from Excel or CSV file"""
        try:
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel files.")
            
            self.file_path = file_path
            
            # Convert Duration to seconds
            if 'Duration' in self.df.columns:
                self.df['Duration_Seconds'] = self.df['Duration'].apply(self._parse_duration)
            
            # Filter out blanks
            if 'Machine Name' in self.df.columns:
                self.df = self.df[self.df['Machine Name'].notna()]
                self.df = self.df[~self.df['Machine Name'].astype(str).str.strip().isin(['', '(blank)', '(blanks)'])]
            
            if 'Operator Name' in self.df.columns:
                self.df = self.df[self.df['Operator Name'].notna()]
                self.df = self.df[~self.df['Operator Name'].astype(str).str.strip().isin(['', '(blank)', '(blanks)'])]
            
            # Get unique values
            self.machines = sorted(self.df['Machine Name'].dropna().unique().tolist())
            self.operators = sorted(self.df['Operator Name'].dropna().unique().tolist())
            
            # Get shifts if available
            if 'Shift' in self.df.columns:
                self.shifts = sorted(self.df['Shift'].dropna().unique().tolist())
            else:
                self.shifts = []
            
            # Parse date columns for trend analysis
            self._parse_dates()
            
            # Reset advanced filters on new data load
            self.filter_date_start = None
            self.filter_date_end = None
            self.filter_shifts = []
            self.filter_cells = []
            self.filter_operators = []
            self.filter_machines = []
            self.cell_duration_filters = {}  # Reset per-cell duration filters
            
            self.df_filtered = self.df.copy()
            return True
            
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")
    
    def _parse_duration(self, duration_val):
        """Convert duration to seconds"""
        if pd.isna(duration_val):
            return 0
        
        try:
            if isinstance(duration_val, timedelta):
                return duration_val.total_seconds()
            
            duration_str = str(duration_val)
            parts = duration_str.split(':')
            
            if len(parts) == 3:
                h, m, s = map(float, parts)
                return h * 3600 + m * 60 + s
            elif len(parts) == 2:
                m, s = map(float, parts)
                return m * 60 + s
            else:
                return float(duration_str)
        except:
            return 0
    
    def _parse_dates(self):
        """Parse date/time columns for trend analysis"""
        if self.df is None:
            return
        
        # Common date column names
        date_cols = ['Date', 'date', 'DATE', 'Timestamp', 'timestamp', 'Start Time', 'Start Date',
                     'End Time', 'End Date', 'DateTime', 'Time', 'Created', 'RecordDate']
        
        date_col = None
        for col in date_cols:
            if col in self.df.columns:
                date_col = col
                break
        
        if date_col:
            try:
                self.df['ParsedDate'] = pd.to_datetime(self.df[date_col], errors='coerce')
                self.df['DateOnly'] = self.df['ParsedDate'].dt.date
            except:
                pass
        
        # If no explicit date column, try to infer from other columns
        if 'ParsedDate' not in self.df.columns:
            for col in self.df.columns:
                if self.df[col].dtype == 'datetime64[ns]':
                    self.df['ParsedDate'] = self.df[col]
                    self.df['DateOnly'] = self.df['ParsedDate'].dt.date
                    break
    
    def get_trend_data(self, group_by='cell'):
        """Get efficiency trend data over time"""
        df = self.get_filtered_df()
        if df is None or 'ParsedDate' not in df.columns:
            return None
        
        df = df.dropna(subset=['ParsedDate'])
        if len(df) == 0:
            return None
        
        df['DateOnly'] = df['ParsedDate'].dt.date
        
        trends = {}
        
        if group_by == 'cell':
            for machine in self.machines:
                cell_num = self.extract_cell_number(machine)
                if cell_num:
                    if cell_num not in trends:
                        trends[cell_num] = {'dates': [], 'efficiencies': []}
                    
                    target = self.get_target_time(machine)
                    machine_data = df[df['Machine Name'] == machine]
                    
                    for date, date_group in machine_data.groupby('DateOnly'):
                        valid_dur = date_group['Duration_Seconds'][date_group['Duration_Seconds'] > 0]
                        if len(valid_dur) > 0:
                            # Use correct efficiency: target / avg_duration
                            avg_dur = valid_dur.mean()
                            eff = (target / avg_dur) * 100 if avg_dur > 0 else 0
                            trends[cell_num]['dates'].append(date)
                            trends[cell_num]['efficiencies'].append(eff)
            
            # Average by date for each cell
            for cell_num in trends:
                date_effs = {}
                for d, e in zip(trends[cell_num]['dates'], trends[cell_num]['efficiencies']):
                    if d not in date_effs:
                        date_effs[d] = []
                    date_effs[d].append(e)
                
                trends[cell_num]['dates'] = sorted(date_effs.keys())
                trends[cell_num]['efficiencies'] = [np.mean(date_effs[d]) for d in trends[cell_num]['dates']]
        
        elif group_by == 'operator':
            for operator in self.operators:
                op_data = df[df['Operator Name'] == operator]
                if len(op_data) == 0:
                    continue
                
                trends[operator] = {'dates': [], 'efficiencies': []}
                
                for date, date_group in op_data.groupby('DateOnly'):
                    # Calculate weighted efficiency across all machines for this date
                    total_target_time = 0
                    total_duration = 0
                    for machine in date_group['Machine Name'].unique():
                        target = self.get_target_time(machine)
                        m_data = date_group[date_group['Machine Name'] == machine]
                        valid_dur = m_data['Duration_Seconds'][m_data['Duration_Seconds'] > 0]
                        if len(valid_dur) > 0:
                            total_target_time += target * len(valid_dur)
                            total_duration += valid_dur.sum()
                    
                    if total_duration > 0:
                        eff = (total_target_time / total_duration) * 100
                        trends[operator]['dates'].append(date)
                        trends[operator]['efficiencies'].append(eff)
        
        return trends
    
    def apply_duration_filter(self, min_seconds: float, max_seconds: float):
        """Apply duration filter to data"""
        if self.df is not None:
            self.df_filtered = self.df[
                (self.df['Duration_Seconds'] >= min_seconds) &
                (self.df['Duration_Seconds'] <= max_seconds)
            ].copy()
    
    def get_shift_analysis(self):
        """Analyze performance by shift"""
        df = self.get_filtered_df()
        if df is None or 'Shift' not in df.columns:
            return None
        
        shift_stats = []
        for shift in sorted(df['Shift'].unique()):
            shift_data = df[df['Shift'] == shift]
            
            # Calculate efficiency using weighted average (total target / total duration)
            total_target_time = 0
            total_duration = 0
            for machine in shift_data['Machine Name'].unique():
                target = self.get_target_time(machine)
                m_data = shift_data[shift_data['Machine Name'] == machine]
                valid_dur = m_data['Duration_Seconds'][m_data['Duration_Seconds'] > 0]
                if len(valid_dur) > 0:
                    total_target_time += target * len(valid_dur)
                    total_duration += valid_dur.sum()
            
            avg_eff = (total_target_time / total_duration * 100) if total_duration > 0 else 0
            
            shift_stats.append({
                'shift': shift,
                'records': len(shift_data),
                'operators': shift_data['Operator Name'].nunique(),
                'avg_duration': shift_data['Duration_Seconds'].mean(),
                'avg_efficiency': avg_eff,
                'quantity': shift_data['Quantity'].sum() if 'Quantity' in shift_data.columns else 0
            })
        
        return shift_stats
    
    def get_productivity_metrics(self):
        """Calculate productivity metrics - uses last station records for quantity"""
        df = self.get_filtered_df()
        if df is None:
            return None
        
        metrics = {
            'total_records': len(df),
            'total_operators': df['Operator Name'].nunique(),
            'total_machines': df['Machine Name'].nunique(),
            'avg_duration': df['Duration_Seconds'].mean(),
            'median_duration': df['Duration_Seconds'].median(),
            'std_duration': df['Duration_Seconds'].std(),
        }
        
        # Add quantity metrics - use only last station records if configured
        if 'Quantity' in df.columns:
            # Filter to only last stations if any are defined
            if self.last_stations:
                # Get list of last stations
                last_station_list = list(self.last_stations.values())
                qty_df = df[df['Machine Name'].isin(last_station_list)]
                
                if len(qty_df) > 0:
                    metrics['total_quantity'] = qty_df['Quantity'].sum()
                    total_hours = qty_df['Duration_Seconds'].sum() / 3600
                    metrics['parts_per_hour'] = metrics['total_quantity'] / total_hours if total_hours > 0 else 0
                    metrics['last_station_records'] = len(qty_df)
                else:
                    # No records at last stations, use all
                    metrics['total_quantity'] = df['Quantity'].sum()
                    total_hours = df['Duration_Seconds'].sum() / 3600
                    metrics['parts_per_hour'] = metrics['total_quantity'] / total_hours if total_hours > 0 else 0
            else:
                # No last stations defined, use all records
                metrics['total_quantity'] = df['Quantity'].sum()
                total_hours = df['Duration_Seconds'].sum() / 3600
                metrics['parts_per_hour'] = metrics['total_quantity'] / total_hours if total_hours > 0 else 0
        
        return metrics
    
    def get_outliers(self, threshold=2.0):
        """Identify statistical outliers using z-score"""
        df = self.get_filtered_df()
        if df is None or len(df) < 3:
            return None
        
        mean = df['Duration_Seconds'].mean()
        std = df['Duration_Seconds'].std()
        
        if std == 0:
            return None
        
        df_copy = df.copy()
        df_copy['z_score'] = (df_copy['Duration_Seconds'] - mean) / std
        
        outliers = df_copy[abs(df_copy['z_score']) > threshold].copy()
        outliers['outlier_type'] = outliers['z_score'].apply(
            lambda x: 'Too Fast' if x < 0 else 'Too Slow'
        )
        
        # Include timestamp columns if available
        cols = ['Operator Name', 'Machine Name', 'Duration_Seconds', 'z_score', 'outlier_type']
        if 'ParsedDate' in outliers.columns:
            cols.insert(0, 'ParsedDate')
        elif 'Date' in outliers.columns:
            cols.insert(0, 'Date')
        
        return outliers[[c for c in cols if c in outliers.columns]]
    
    def get_operator_rankings(self, sort_by='efficiency'):
        """Get operator rankings with various metrics"""
        df = self.get_filtered_df()
        if df is None:
            return None
        
        rankings = []
        for operator in self.operators:
            op_data = df[df['Operator Name'] == operator]
            if len(op_data) == 0:
                continue
            
            # Calculate efficiency using weighted average (target / avg_duration)
            # This gives accurate efficiency based on actual average performance
            total_target_time = 0
            total_duration = 0
            for machine in op_data['Machine Name'].unique():
                target = self.get_target_time(machine)
                m_data = op_data[op_data['Machine Name'] == machine]
                valid_dur = m_data['Duration_Seconds'][m_data['Duration_Seconds'] > 0]
                if len(valid_dur) > 0:
                    total_target_time += target * len(valid_dur)
                    total_duration += valid_dur.sum()
            
            avg_eff = (total_target_time / total_duration * 100) if total_duration > 0 else 0
            
            rankings.append({
                'operator': operator,
                'records': len(op_data),
                'machines': op_data['Machine Name'].nunique(),
                'avg_duration': op_data['Duration_Seconds'].mean(),
                'efficiency': avg_eff,
                'consistency': 100 - op_data['Duration_Seconds'].std() / op_data['Duration_Seconds'].mean() * 100 if op_data['Duration_Seconds'].mean() > 0 else 0,
                'quantity': op_data['Quantity'].sum() if 'Quantity' in op_data.columns else 0
            })
        
        # Sort by requested metric
        sort_key = sort_by if sort_by in ['efficiency', 'records', 'quantity', 'consistency'] else 'efficiency'
        reverse = sort_key in ['efficiency', 'records', 'quantity', 'consistency']
        rankings.sort(key=lambda x: x[sort_key], reverse=reverse)
        
        return rankings


class ModernStyle:
    """Modern styling constants"""
    
    # Colors
    PRIMARY = "#2563eb"
    PRIMARY_DARK = "#1d4ed8"
    SECONDARY = "#64748b"
    SUCCESS = "#10b981"
    WARNING = "#f59e0b"
    DANGER = "#ef4444"
    BG_LIGHT = "#f8fafc"
    BG_WHITE = "#ffffff"
    TEXT_DARK = "#1e293b"
    TEXT_MUTED = "#64748b"
    BORDER = "#e2e8f0"
    
    @staticmethod
    def configure_styles():
        """Configure ttk styles for modern look"""
        style = ttk.Style()
        
        # Try to use a modern theme
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use('clam')
        
        # Configure main styles
        style.configure('Title.TLabel', font=('Segoe UI', 22, 'bold'), foreground=ModernStyle.TEXT_DARK)
        style.configure('Subtitle.TLabel', font=('Segoe UI', 12), foreground=ModernStyle.TEXT_MUTED)
        style.configure('Header.TLabel', font=('Segoe UI', 12, 'bold'), foreground=ModernStyle.TEXT_DARK)
        style.configure('Card.TFrame', background=ModernStyle.BG_WHITE)
        style.configure('Modern.TButton', font=('Segoe UI', 11), padding=(15, 8))
        style.configure('Primary.TButton', font=('Segoe UI', 11, 'bold'), padding=(20, 10))
        style.configure('TNotebook', background=ModernStyle.BG_LIGHT)
        style.configure('TNotebook.Tab', font=('Segoe UI', 11), padding=(20, 10))
        style.configure('TLabelframe.Label', font=('Segoe UI', 11, 'bold'))
        
        # Configure Entry and Spinbox to be clearly editable
        style.configure('TEntry', fieldbackground='white', foreground='black', font=('Segoe UI', 11))
        style.configure('TSpinbox', fieldbackground='white', foreground='black', arrowsize=13, font=('Segoe UI', 11))
        style.map('TEntry', 
                  fieldbackground=[('disabled', '#e0e0e0'), ('readonly', '#f0f0f0')],
                  foreground=[('disabled', 'gray')])
        style.map('TSpinbox',
                  fieldbackground=[('disabled', '#e0e0e0'), ('readonly', '#f0f0f0')],
                  foreground=[('disabled', 'gray')])
        
        # Configure Treeview (tables) with larger fonts
        style.configure('Treeview', 
                        font=('Segoe UI', 11),
                        rowheight=28)
        style.configure('Treeview.Heading', 
                        font=('Segoe UI', 11, 'bold'),
                        padding=(5, 8))
        
        # Configure Labels with larger default font
        style.configure('TLabel', font=('Segoe UI', 11))
        style.configure('TCheckbutton', font=('Segoe UI', 10))
        style.configure('TRadiobutton', font=('Segoe UI', 10))
        style.configure('TCombobox', font=('Segoe UI', 11))
        
        return style


class TargetTimesTab(ttk.Frame):
    """Target Times Configuration Tab - Centralized target time settings"""
    
    def __init__(self, parent, data_manager):
        super().__init__(parent)
        self.data_manager = data_manager
        self.entries = {}
        
        self._setup_ui()
    
    def _setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(header_frame, text="⏱️ Target Times Configuration", style='Title.TLabel').pack(side=tk.LEFT)
        ttk.Label(header_frame, text="Set target cycle times for all stations (used across all tabs)",
                  style='Subtitle.TLabel').pack(side=tk.LEFT, padx=(20, 0))
        
        # Content area - use PanedWindow for resizable layout
        content_frame = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Instructions (compact)
        left_panel = ttk.Frame(content_frame)
        content_frame.add(left_panel, weight=1)
        
        # Instructions card
        info_frame = ttk.LabelFrame(left_panel, text="📖 Instructions", padding="15")
        info_frame.pack(fill=tk.X, pady=(0, 15))
        
        info_text = tk.Text(info_frame, wrap=tk.WORD, height=10, width=38,
                           font=('Segoe UI', 11), bg='#f8fafc', relief='flat')
        info_text.pack(fill=tk.X)
        
        instructions = """Set the target cycle time (in seconds) for each station in your factory.

These target times are used to calculate:
• Operator efficiency (Efficiency Calculator tab)
• Cell efficiency (Cell Efficiency tab)
• Duration analysis comparisons

📊 Formula used:
Efficiency = (Target Time ÷ Actual Time) × 100%

💡 Tips:
• Set realistic targets based on ideal conditions
• Consider machine capabilities and product type
• Update targets as processes improve"""
        
        info_text.insert('1.0', instructions)
        info_text.config(state='disabled')
        
        # Status
        self.status_label = ttk.Label(left_panel, text="Load data to configure target times", 
                                       style='Subtitle.TLabel')
        self.status_label.pack(pady=10)
        
        # Save button
        ttk.Button(left_panel, text="💾 Save All Target Times", 
                   command=self._save_all_targets, style='Primary.TButton').pack(fill=tk.X, pady=10)
        
        # Right panel - Target time inputs (takes more space)
        right_panel = ttk.Frame(content_frame)
        content_frame.add(right_panel, weight=3)
        
        # Scrollable frame for stations
        stations_frame = ttk.LabelFrame(right_panel, text="📋 Station Target Times (seconds)", padding="10")
        stations_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas and scrollbar
        canvas = tk.Canvas(stations_frame, highlightthickness=0)
        scrollbar_y = ttk.Scrollbar(stations_frame, orient="vertical", command=canvas.yview)
        scrollbar_x = ttk.Scrollbar(stations_frame, orient="horizontal", command=canvas.xview)
        self.scroll_frame = ttk.Frame(canvas)
        
        self.scroll_frame.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        # Pack scrollbars and canvas
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Store canvas reference for mousewheel
        self.stations_canvas = canvas
        
        # Bind mousewheel to canvas only when mouse is over canvas (not over entry widgets)
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind("<MouseWheel>", _on_mousewheel)
        self.scroll_frame.bind("<MouseWheel>", _on_mousewheel)
    
    def refresh_data(self):
        """Refresh UI when data is loaded"""
        # Clear existing entries
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
        self.entries.clear()
        
        if not self.data_manager.machines:
            ttk.Label(self.scroll_frame, text="No data loaded. Please load a data file first.",
                     style='Subtitle.TLabel').pack(pady=20)
            return
        
        # Get cells and stations
        cells = self.data_manager.get_cells_and_stations()
        
        # Create entries grouped by cell
        row = 0
        for cell_num in sorted(cells.keys()):
            # Cell header
            cell_header = ttk.Frame(self.scroll_frame)
            cell_header.grid(row=row, column=0, columnspan=4, sticky='w', pady=(15, 5))
            
            ttk.Label(cell_header, text=f"🏭 Cell {cell_num}", 
                     font=('Segoe UI', 11, 'bold')).pack(side=tk.LEFT)
            ttk.Label(cell_header, text=f"({len(cells[cell_num])} stations)",
                     style='Subtitle.TLabel').pack(side=tk.LEFT, padx=10)
            
            row += 1
            
            # Station entries in grid (2 columns)
            col = 0
            for station in cells[cell_num]:
                station_short = station.replace(cell_num, '').strip()
                
                frame = ttk.Frame(self.scroll_frame)
                frame.grid(row=row, column=col, sticky='w', padx=10, pady=3)
                
                ttk.Label(frame, text=station_short, width=20).pack(side=tk.LEFT)
                
                entry = ttk.Entry(frame, width=8)
                # Load existing value or default
                current_val = self.data_manager.target_times.get(station, 30)
                entry.insert(0, str(int(current_val) if current_val == int(current_val) else current_val))
                entry.pack(side=tk.LEFT, padx=5)
                
                ttk.Label(frame, text="s").pack(side=tk.LEFT)
                
                self.entries[station] = entry
                
                col += 1
                if col >= 2:
                    col = 0
                    row += 1
            
            if col != 0:
                row += 1
        
        # Update status
        num_stations = len(self.entries)
        num_cells = len(cells)
        self.status_label.config(text=f"✅ {num_stations} stations in {num_cells} cells")
        
        # Auto-save existing values
        self._save_all_targets(silent=True)
    
    def _save_all_targets(self, silent=False):
        """Save all target times to the data manager and persist to file"""
        saved = 0
        for station, entry in self.entries.items():
            try:
                value = float(entry.get())
                self.data_manager.set_target_time(station, value)
                saved += 1
            except ValueError:
                pass
        
        # Persist to file
        self.data_manager.save_settings()
        
        if not silent:
            messagebox.showinfo("Saved", f"Target times saved for {saved} stations.\n\n"
                               "Settings will persist across sessions.")


class DataFiltersTab(ttk.Frame):
    """Data Filters Tab - Filter data by duration range"""
    
    def __init__(self, parent, data_manager, on_filter_change=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.on_filter_change = on_filter_change  # Callback to refresh other tabs
        
        self._setup_ui()
    
    def _setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(header_frame, text="🔍 Data Filters", style='Title.TLabel').pack(side=tk.LEFT)
        ttk.Label(header_frame, text="Filter data by duration range (applies to all tabs)",
                  style='Subtitle.TLabel').pack(side=tk.LEFT, padx=(20, 0))
        
        # Initialize filter variables with defaults
        self.filters_enabled = tk.BooleanVar(value=True)
        self.min_value = tk.StringVar(value="3")
        self.min_unit = tk.StringVar(value="seconds")
        self.max_value = tk.StringVar(value="30")
        self.max_unit = tk.StringVar(value="minutes")
        
        # Content area - Grid layout
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.columnconfigure(2, weight=1)
        content_frame.rowconfigure(0, weight=1)
        content_frame.rowconfigure(1, weight=1)
        
        # ===== ROW 0: Duration Filters (Cell, Station, Stats) =====
        
        # Per-Cell Duration Filters (Col 0)
        cell_filter_frame = ttk.LabelFrame(content_frame, text="🏭 Per-Cell Duration Filters", padding="10")
        cell_filter_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        
        cell_grid = ttk.Frame(cell_filter_frame)
        cell_grid.pack(fill=tk.X, pady=5)
        
        ttk.Label(cell_grid, text="Cell:").grid(row=0, column=0, sticky='w', padx=2)
        self.cell_duration_var = tk.StringVar(value="")
        self.cell_duration_combo = ttk.Combobox(cell_grid, textvariable=self.cell_duration_var,
                                                 values=[], state="readonly", width=12)
        self.cell_duration_combo.grid(row=0, column=1, sticky='w', padx=2)
        self.cell_duration_combo.bind('<<ComboboxSelected>>', self._on_cell_duration_select)
        
        ttk.Label(cell_grid, text="Min:").grid(row=1, column=0, sticky='w', padx=2, pady=2)
        self.cell_min_var = tk.StringVar(value="3")
        ttk.Entry(cell_grid, textvariable=self.cell_min_var, width=8).grid(row=1, column=1, sticky='w', padx=2)
        
        ttk.Label(cell_grid, text="Max:").grid(row=2, column=0, sticky='w', padx=2, pady=2)
        self.cell_max_var = tk.StringVar(value="1800")
        ttk.Entry(cell_grid, textvariable=self.cell_max_var, width=8).grid(row=2, column=1, sticky='w', padx=2)
        
        cell_btn_frame = ttk.Frame(cell_filter_frame)
        cell_btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(cell_btn_frame, text="Apply", command=self._apply_cell_duration_filter).pack(side=tk.LEFT, padx=2)
        ttk.Button(cell_btn_frame, text="Clear", command=self._clear_cell_duration_filter).pack(side=tk.LEFT, padx=2)
        ttk.Button(cell_btn_frame, text="Clear All", command=self._clear_all_cell_duration_filters).pack(side=tk.LEFT, padx=2)
        
        self.cell_filters_display = tk.Text(cell_filter_frame, height=3, font=('Consolas', 9), bg='#f1f5f9', state='disabled')
        self.cell_filters_display.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Per-Station Duration Filters (Col 1)
        station_filter_frame = ttk.LabelFrame(content_frame, text="🔧 Per-Station Duration Filters", padding="10")
        station_filter_frame.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        
        station_grid = ttk.Frame(station_filter_frame)
        station_grid.pack(fill=tk.X, pady=5)
        
        ttk.Label(station_grid, text="Station:").grid(row=0, column=0, sticky='w', padx=2)
        self.station_duration_var = tk.StringVar(value="")
        self.station_duration_combo = ttk.Combobox(station_grid, textvariable=self.station_duration_var,
                                                    values=[], state="readonly", width=20)
        self.station_duration_combo.grid(row=0, column=1, sticky='w', padx=2)
        self.station_duration_combo.bind('<<ComboboxSelected>>', self._on_station_duration_select)
        
        ttk.Label(station_grid, text="Min:").grid(row=1, column=0, sticky='w', padx=2, pady=2)
        self.station_min_var = tk.StringVar(value="3")
        ttk.Entry(station_grid, textvariable=self.station_min_var, width=8).grid(row=1, column=1, sticky='w', padx=2)
        
        ttk.Label(station_grid, text="Max:").grid(row=2, column=0, sticky='w', padx=2, pady=2)
        self.station_max_var = tk.StringVar(value="1800")
        ttk.Entry(station_grid, textvariable=self.station_max_var, width=8).grid(row=2, column=1, sticky='w', padx=2)
        
        station_btn_frame = ttk.Frame(station_filter_frame)
        station_btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(station_btn_frame, text="Apply", command=self._apply_station_duration_filter).pack(side=tk.LEFT, padx=2)
        ttk.Button(station_btn_frame, text="Clear", command=self._clear_station_duration_filter).pack(side=tk.LEFT, padx=2)
        ttk.Button(station_btn_frame, text="Clear All", command=self._clear_all_station_duration_filters).pack(side=tk.LEFT, padx=2)
        
        self.station_filters_display = tk.Text(station_filter_frame, height=3, font=('Consolas', 9), bg='#f1f5f9', state='disabled')
        self.station_filters_display.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Filter Statistics (Col 2)
        stats_frame = ttk.LabelFrame(content_frame, text="📊 Filter Statistics", padding="10")
        stats_frame.grid(row=0, column=2, sticky='nsew', padx=5, pady=5)
        
        self.stats_display = ttk.Frame(stats_frame)
        self.stats_display.pack(fill=tk.BOTH, expand=True)
        
        self.no_data_label = ttk.Label(self.stats_display, text="Load data to see statistics", style='Subtitle.TLabel')
        self.no_data_label.pack(pady=30)
        
        self.total_label = None
        self.kept_label = None
        self.removed_label = None
        self.range_label = None
        
        # ===== ROW 1: Advanced Filters & Exclude Operators =====
        
        # Advanced Filters (Col 0-1)
        adv_frame = ttk.LabelFrame(content_frame, text="🎯 Advanced Filters", padding="10")
        adv_frame.grid(row=1, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)
        
        adv_grid = ttk.Frame(adv_frame)
        adv_grid.pack(fill=tk.BOTH, expand=True)
        adv_grid.columnconfigure(0, weight=1)
        adv_grid.columnconfigure(1, weight=1)
        adv_grid.columnconfigure(2, weight=1)
        
        # Column 0: Date Range & Shifts
        col0 = ttk.Frame(adv_grid)
        col0.grid(row=0, column=0, sticky='nsew', padx=10)
        
        ttk.Label(col0, text="📅 Date Range:", font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        date_row = ttk.Frame(col0)
        date_row.pack(fill=tk.X, pady=2)
        self.date_start = tk.StringVar(value="")
        self.date_end = tk.StringVar(value="")
        ttk.Entry(date_row, textvariable=self.date_start, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Label(date_row, text="to").pack(side=tk.LEFT, padx=2)
        ttk.Entry(date_row, textvariable=self.date_end, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Label(col0, text="(YYYY-MM-DD)", style='Subtitle.TLabel').pack(anchor='w')
        
        ttk.Label(col0, text="🔄 Shifts:", font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(10, 0))
        self.shift_listbox = tk.Listbox(col0, height=4, selectmode=tk.MULTIPLE, exportselection=False, width=15)
        self.shift_listbox.pack(fill=tk.X, pady=2)
        ttk.Label(col0, text="(Ctrl+click to multi-select)", style='Subtitle.TLabel').pack(anchor='w')
        
        # Column 1: Cell, Operator, Station
        col1 = ttk.Frame(adv_grid)
        col1.grid(row=0, column=1, sticky='nsew', padx=10)
        
        ttk.Label(col1, text="🏭 Filter by Cell:", font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        self.cell_var = tk.StringVar(value="All")
        self.cell_combo = ttk.Combobox(col1, textvariable=self.cell_var, values=["All"], state="readonly", width=20)
        self.cell_combo.pack(fill=tk.X, pady=2)
        
        ttk.Label(col1, text="👤 Filter by Operator:", font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(10, 0))
        self.operator_var = tk.StringVar(value="All")
        self.operator_combo = ttk.Combobox(col1, textvariable=self.operator_var, values=["All"], state="readonly", width=20)
        self.operator_combo.pack(fill=tk.X, pady=2)
        
        ttk.Label(col1, text="🔧 Filter by Station:", font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(10, 0))
        self.machine_var = tk.StringVar(value="All")
        self.machine_combo = ttk.Combobox(col1, textvariable=self.machine_var, values=["All"], state="readonly", width=20)
        self.machine_combo.pack(fill=tk.X, pady=2)
        
        # Column 2: Min Records & Buttons
        col2 = ttk.Frame(adv_grid)
        col2.grid(row=0, column=2, sticky='nsew', padx=10)
        
        ttk.Label(col2, text="📊 Min Records:", font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        min_rec_row = ttk.Frame(col2)
        min_rec_row.pack(fill=tk.X, pady=2)
        self.min_records_var = tk.StringVar(value="0")
        ttk.Spinbox(min_rec_row, from_=0, to=1000, textvariable=self.min_records_var, width=8).pack(side=tk.LEFT)
        ttk.Label(min_rec_row, text="per operator", style='Subtitle.TLabel').pack(side=tk.LEFT, padx=5)
        
        ttk.Button(col2, text="🔄 Clear Advanced Filters", command=self._clear_advanced_filters).pack(fill=tk.X, pady=(20, 5))
        ttk.Button(col2, text="✅ Apply All Filters", command=self._apply_filters, style='Primary.TButton').pack(fill=tk.X, pady=5)
        
        # Exclude Operators (Col 2)
        exclude_frame = ttk.LabelFrame(content_frame, text="🚫 Exclude Operators", padding="10")
        exclude_frame.grid(row=1, column=2, sticky='nsew', padx=5, pady=5)
        
        ttk.Label(exclude_frame, text="Select operators to exclude:", style='Subtitle.TLabel').pack(anchor='w')
        
        exclude_list_frame = ttk.Frame(exclude_frame)
        exclude_list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.exclude_listbox = tk.Listbox(exclude_list_frame, height=6, selectmode=tk.MULTIPLE,
                                           exportselection=False, font=('Segoe UI', 9))
        exclude_scroll = ttk.Scrollbar(exclude_list_frame, orient=tk.VERTICAL, command=self.exclude_listbox.yview)
        self.exclude_listbox.configure(yscrollcommand=exclude_scroll.set)
        
        self.exclude_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        exclude_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        exclude_btn_frame = ttk.Frame(exclude_frame)
        exclude_btn_frame.pack(fill=tk.X)
        ttk.Button(exclude_btn_frame, text="Select All", command=self._select_all_exclude).pack(side=tk.LEFT, padx=2)
        ttk.Button(exclude_btn_frame, text="Clear All", command=self._clear_all_exclude).pack(side=tk.LEFT, padx=2)
    
    def _apply_preset(self, min_val, min_unit, max_val, max_unit):
        """Apply a quick preset"""
        self.min_value.set(str(min_val))
        self.min_unit.set(min_unit)
        self.max_value.set(str(max_val))
        self.max_unit.set(max_unit)
        self.filters_enabled.set(True)
        self._apply_filters()
    
    def _on_enable_toggle(self):
        """Handle enable/disable toggle"""
        pass  # Will apply on Apply button click
    
    def _get_seconds(self, value, unit):
        """Convert value to seconds"""
        val = float(value)
        if unit == "minutes":
            val *= 60
        return val
    
    def _on_cell_duration_select(self, event=None):
        """When a cell is selected, show its current filter values"""
        cell = self.cell_duration_var.get()
        if cell:
            filter_vals = self.data_manager.get_cell_duration_filter(cell)
            self.cell_min_var.set(str(int(filter_vals['min'])))
            self.cell_max_var.set(str(int(filter_vals['max'])))
    
    def _apply_cell_duration_filter(self):
        """Apply duration filter to selected cell"""
        cell = self.cell_duration_var.get()
        if not cell:
            messagebox.showwarning("No Cell", "Please select a cell first.")
            return
        
        try:
            min_sec = float(self.cell_min_var.get())
            max_sec = float(self.cell_max_var.get())
            
            if min_sec >= max_sec:
                messagebox.showerror("Error", "Minimum must be less than maximum.")
                return
            
            self.data_manager.set_cell_duration_filter(cell, min_sec, max_sec)
            self._update_cell_filters_display()
            self._update_stats_display()
            
            if self.on_filter_change:
                self.on_filter_change()
            
            messagebox.showinfo("Applied", f"Duration filter applied to Cell {cell}:\n"
                                           f"Min: {min_sec}s, Max: {max_sec}s")
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers.")
    
    def _clear_cell_duration_filter(self):
        """Clear duration filter for selected cell"""
        cell = self.cell_duration_var.get()
        if not cell:
            messagebox.showwarning("No Cell", "Please select a cell first.")
            return
        
        self.data_manager.clear_cell_duration_filter(cell)
        self._update_cell_filters_display()
        self._update_stats_display()
        
        if self.on_filter_change:
            self.on_filter_change()
        
        # Reset to global defaults
        self.cell_min_var.set(str(int(self.data_manager.filter_min_seconds)))
        self.cell_max_var.set(str(int(self.data_manager.filter_max_seconds)))
        
        messagebox.showinfo("Cleared", f"Cell {cell} will now use global filter settings.")
    
    def _clear_all_cell_duration_filters(self):
        """Clear all per-cell duration filters"""
        self.data_manager.clear_all_cell_duration_filters()
        self._update_cell_filters_display()
        self._update_stats_display()
        
        if self.on_filter_change:
            self.on_filter_change()
        
        messagebox.showinfo("Cleared", "All per-cell filters cleared. Using global settings.")
    
    def _update_cell_filters_display(self):
        """Update the display showing current per-cell filters"""
        self.cell_filters_display.config(state='normal')
        self.cell_filters_display.delete('1.0', tk.END)
        
        filters = self.data_manager.cell_duration_filters
        if not filters:
            self.cell_filters_display.insert('1.0', "No per-cell filters set. Using global settings.")
        else:
            lines = []
            for cell, vals in sorted(filters.items()):
                lines.append(f"Cell {cell}: {vals['min']:.0f}s - {vals['max']:.0f}s")
            self.cell_filters_display.insert('1.0', "\n".join(lines))
        
        self.cell_filters_display.config(state='disabled')
    
    def _on_station_duration_select(self, event=None):
        """When a station is selected, show its current filter values"""
        station = self.station_duration_var.get()
        if station:
            if station in self.data_manager.station_duration_filters:
                f = self.data_manager.station_duration_filters[station]
                self.station_min_var.set(str(int(f['min'])))
                self.station_max_var.set(str(int(f['max'])))
            else:
                # Use global defaults
                self.station_min_var.set(str(int(self.data_manager.filter_min_seconds)))
                self.station_max_var.set(str(int(self.data_manager.filter_max_seconds)))
    
    def _apply_station_duration_filter(self):
        """Apply duration filter to selected station"""
        station = self.station_duration_var.get()
        if not station:
            messagebox.showwarning("No Station", "Please select a station first.")
            return
        
        try:
            min_sec = float(self.station_min_var.get())
            max_sec = float(self.station_max_var.get())
            
            if min_sec >= max_sec:
                messagebox.showerror("Error", "Minimum must be less than maximum.")
                return
            
            self.data_manager.station_duration_filters[station] = {'min': min_sec, 'max': max_sec}
            self.data_manager.save_settings()
            self._update_station_filters_display()
            self._update_stats_display()
            
            if self.on_filter_change:
                self.on_filter_change()
            
            messagebox.showinfo("Applied", f"Duration filter applied to {station}:\n"
                                           f"Min: {min_sec}s, Max: {max_sec}s")
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers.")
    
    def _clear_station_duration_filter(self):
        """Clear duration filter for selected station"""
        station = self.station_duration_var.get()
        if not station:
            messagebox.showwarning("No Station", "Please select a station first.")
            return
        
        if station in self.data_manager.station_duration_filters:
            del self.data_manager.station_duration_filters[station]
            self.data_manager.save_settings()
        
        self._update_station_filters_display()
        self._update_stats_display()
        
        if self.on_filter_change:
            self.on_filter_change()
        
        # Reset to global defaults
        self.station_min_var.set(str(int(self.data_manager.filter_min_seconds)))
        self.station_max_var.set(str(int(self.data_manager.filter_max_seconds)))
        
        messagebox.showinfo("Cleared", f"Station {station} will now use global/cell filter settings.")
    
    def _clear_all_station_duration_filters(self):
        """Clear all per-station duration filters"""
        self.data_manager.station_duration_filters = {}
        self.data_manager.save_settings()
        self._update_station_filters_display()
        self._update_stats_display()
        
        if self.on_filter_change:
            self.on_filter_change()
        
        messagebox.showinfo("Cleared", "All per-station filters cleared.")
    
    def _update_station_filters_display(self):
        """Update the display showing current per-station filters"""
        self.station_filters_display.config(state='normal')
        self.station_filters_display.delete('1.0', tk.END)
        
        filters = self.data_manager.station_duration_filters
        if not filters:
            self.station_filters_display.insert('1.0', "No per-station filters set.")
        else:
            lines = []
            for station, vals in sorted(filters.items()):
                lines.append(f"{station[:20]}: {vals['min']:.0f}s - {vals['max']:.0f}s")
            self.station_filters_display.insert('1.0', "\n".join(lines))
        
        self.station_filters_display.config(state='disabled')
    
    def _apply_filters(self):
        """Apply the current filter settings"""
        if self.data_manager.df is None:
            messagebox.showwarning("No Data", "Please load a data file first.")
            return
        
        try:
            min_seconds = self._get_seconds(self.min_value.get(), self.min_unit.get())
            max_seconds = self._get_seconds(self.max_value.get(), self.max_unit.get())
            
            if min_seconds >= max_seconds:
                messagebox.showerror("Error", "Minimum must be less than maximum.")
                return
            
            enabled = self.filters_enabled.get()
            self.data_manager.set_duration_filters(min_seconds, max_seconds, enabled)
            
            # Apply advanced filters
            date_start = self.date_start.get().strip() if self.date_start.get().strip() else None
            date_end = self.date_end.get().strip() if self.date_end.get().strip() else None
            
            # Get selected shifts
            selected_shifts = []
            for idx in self.shift_listbox.curselection():
                shift_text = self.shift_listbox.get(idx)
                # Extract number from "Shift X"
                shift_num = int(shift_text.replace("Shift ", ""))
                selected_shifts.append(shift_num)
            
            # Get selected cell
            selected_cells = []
            if self.cell_var.get() != "All":
                selected_cells = [self.cell_var.get()]
            
            # Get selected operator
            selected_operators = []
            if self.operator_var.get() != "All":
                selected_operators = [self.operator_var.get()]
            
            # Get selected machine
            selected_machines = []
            if self.machine_var.get() != "All":
                selected_machines = [self.machine_var.get()]
            
            # Get minimum records filter
            try:
                min_records = int(self.min_records_var.get())
            except ValueError:
                min_records = 0
            self.data_manager.filter_min_records = min_records
            
            # Get excluded operators
            excluded_ops = [self.exclude_listbox.get(idx) for idx in self.exclude_listbox.curselection()]
            self.data_manager.excluded_operators = excluded_ops
            
            self.data_manager.set_advanced_filters(
                date_start=date_start,
                date_end=date_end,
                shifts=selected_shifts,
                cells=selected_cells,
                operators=selected_operators,
                machines=selected_machines
            )
            
            # Persist settings to file
            self.data_manager.save_settings()
            
            # Update stats display
            self._update_stats_display()
            
            # Notify other tabs to refresh
            if self.on_filter_change:
                self.on_filter_change()
            
            stats = self.data_manager.get_filter_stats()
            if enabled:
                messagebox.showinfo("Filters Applied", 
                    f"Filters applied successfully!\n\n"
                    f"• Kept: {stats['filtered']:,} records\n"
                    f"• Removed: {stats['removed']:,} records\n\n"
                    f"Settings saved for future sessions.")
            else:
                messagebox.showinfo("Filters Disabled", 
                    f"Filters disabled. Using all {stats['total']:,} records.")
                
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid filter values: {e}")
    
    def _update_stats_display(self):
        """Update the statistics display"""
        # Clear existing
        for widget in self.stats_display.winfo_children():
            widget.destroy()
        
        stats = self.data_manager.get_filter_stats()
        if stats is None:
            ttk.Label(self.stats_display, text="Load data to see filter statistics",
                     style='Subtitle.TLabel').pack(pady=50)
            return
        
        # Create compact stats display
        grid = ttk.Frame(self.stats_display)
        grid.pack(pady=10)
        
        # Total records
        total_frame = ttk.Frame(grid)
        total_frame.grid(row=0, column=0, padx=15, pady=5)
        ttk.Label(total_frame, text="📁 Total", style='Header.TLabel').pack()
        ttk.Label(total_frame, text=f"{stats['total']:,}", 
                  font=('Segoe UI', 20, 'bold')).pack()
        
        # Kept records
        kept_frame = ttk.Frame(grid)
        kept_frame.grid(row=0, column=1, padx=15, pady=5)
        ttk.Label(kept_frame, text="✅ Kept", style='Header.TLabel').pack()
        kept_pct = (stats['filtered'] / stats['total'] * 100) if stats['total'] > 0 else 0
        ttk.Label(kept_frame, text=f"{stats['filtered']:,}", 
                  font=('Segoe UI', 20, 'bold'), foreground='#22c55e').pack()
        ttk.Label(kept_frame, text=f"({kept_pct:.1f}%)", style='Subtitle.TLabel').pack()
        
        # Removed records
        removed_frame = ttk.Frame(grid)
        removed_frame.grid(row=0, column=2, padx=15, pady=5)
        ttk.Label(removed_frame, text="🚫 Removed", style='Header.TLabel').pack()
        removed_pct = (stats['removed'] / stats['total'] * 100) if stats['total'] > 0 else 0
        ttk.Label(removed_frame, text=f"{stats['removed']:,}", 
                  font=('Segoe UI', 20, 'bold'), foreground='#ef4444').pack()
        ttk.Label(removed_frame, text=f"({removed_pct:.1f}%)", style='Subtitle.TLabel').pack()
        
        # Filter range info - compact
        status = "✅ Enabled" if stats['enabled'] else "❌ Disabled"
        min_display = f"{stats['min_seconds']:.0f}s" if stats['min_seconds'] < 60 else f"{stats['min_seconds']/60:.1f}min"
        max_display = f"{stats['max_seconds']:.0f}s" if stats['max_seconds'] < 60 else f"{stats['max_seconds']/60:.1f}min"
        
        ttk.Label(self.stats_display, text=f"Filter: {min_display} to {max_display} | {status}",
                  font=('Segoe UI', 10)).pack(pady=5)
    
    def _clear_advanced_filters(self):
        """Clear all advanced filter selections"""
        self.date_start.set("")
        self.date_end.set("")
        self.shift_listbox.selection_clear(0, tk.END)
        self.cell_var.set("All")
        self.operator_var.set("All")
        self.machine_var.set("All")
        self.min_records_var.set("0")
        self.exclude_listbox.selection_clear(0, tk.END)
    
    def _select_all_exclude(self):
        """Select all operators in exclude list"""
        self.exclude_listbox.selection_set(0, tk.END)
    
    def _clear_all_exclude(self):
        """Clear all selections in exclude list"""
        self.exclude_listbox.selection_clear(0, tk.END)
    
    def refresh_data(self):
        """Refresh UI when data is loaded"""
        # Populate advanced filter options
        if self.data_manager.df is not None:
            # Populate shifts
            self.shift_listbox.delete(0, tk.END)
            for shift in self.data_manager.shifts:
                self.shift_listbox.insert(tk.END, f"Shift {shift}")
            
            # Populate cells
            cells_dict = self.data_manager.get_cells_and_stations()
            cells = ["All"] + sorted(cells_dict.keys())
            self.cell_combo['values'] = cells
            self.cell_var.set("All")
            
            # Populate cell duration filter combo (without "All")
            self.cell_duration_combo['values'] = sorted(cells_dict.keys())
            if cells_dict:
                self.cell_duration_combo.set(sorted(cells_dict.keys())[0])
                self._on_cell_duration_select()
            
            # Update per-cell filters display
            self._update_cell_filters_display()
            
            # Populate station duration filter combo
            self.station_duration_combo['values'] = sorted(self.data_manager.machines)
            if self.data_manager.machines:
                self.station_duration_combo.set(sorted(self.data_manager.machines)[0])
                self._on_station_duration_select()
            
            # Update per-station filters display
            self._update_station_filters_display()
            
            # Populate operators
            ops = ["All"] + self.data_manager.operators
            self.operator_combo['values'] = ops
            self.operator_var.set("All")
            
            # Populate exclude operators listbox
            self.exclude_listbox.delete(0, tk.END)
            for op in sorted(self.data_manager.operators):
                self.exclude_listbox.insert(tk.END, op)
            
            # Populate machines
            machines = ["All"] + self.data_manager.machines
            self.machine_combo['values'] = machines
            self.machine_var.set("All")
            
            # Set date range hints if dates available
            if 'ParsedDate' in self.data_manager.df.columns:
                dates = self.data_manager.df['ParsedDate'].dropna()
                if len(dates) > 0:
                    min_date = dates.min().strftime('%Y-%m-%d')
                    max_date = dates.max().strftime('%Y-%m-%d')
                    self.date_start.set(min_date)
                    self.date_end.set(max_date)
        
        # Apply default filters on data load
        if self.data_manager.df is not None:
            min_seconds = self._get_seconds(self.min_value.get(), self.min_unit.get())
            max_seconds = self._get_seconds(self.max_value.get(), self.max_unit.get())
            self.data_manager.set_duration_filters(min_seconds, max_seconds, self.filters_enabled.get())
        
        self._update_stats_display()


class DurationAnalysisTab(ttk.Frame):
    """Duration Analysis Tool Tab"""
    
    def __init__(self, parent, data_manager):
        super().__init__(parent)
        self.data_manager = data_manager
        self.analysis_summary = {}
        
        # Filter defaults
        self.min_duration = 3
        self.max_duration = 1800
        
        self._setup_ui()
    
    def _setup_ui(self):
        # Main container with padding
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(header_frame, text="📊 Duration Analysis", 
                  style='Title.TLabel').pack(side=tk.LEFT)
        ttk.Label(header_frame, text="Analyze operator performance against target times",
                  style='Subtitle.TLabel').pack(side=tk.LEFT, padx=(20, 0))
        
        # Content area - two columns
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Controls
        left_panel = ttk.LabelFrame(content_frame, text="⚙️ Settings", padding="15")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Data Filters info
        filter_frame = ttk.LabelFrame(left_panel, text="🔍 Data Filters", padding="10")
        filter_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(filter_frame, text="ℹ️ Duration filters are configured in the",
                  style='Subtitle.TLabel').pack(anchor=tk.W)
        ttk.Label(filter_frame, text="🔍 Data Filters tab",
                  font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W, pady=(2, 5))
        
        self.filter_summary = ttk.Label(filter_frame, text="No data loaded",
                                         style='Subtitle.TLabel', wraplength=260)
        self.filter_summary.pack(anchor=tk.W, pady=(5, 0))
        
        # Target times info (reference to Target Times tab)
        target_label_frame = ttk.LabelFrame(left_panel, text="⏱️ Target Times", padding="10")
        target_label_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(target_label_frame, text="ℹ️ Target times are configured in the",
                  style='Subtitle.TLabel').pack(anchor=tk.W)
        ttk.Label(target_label_frame, text="⏱️ Target Times tab",
                  font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W, pady=(2, 5))
        
        self.targets_summary = ttk.Label(target_label_frame, text="No data loaded",
                                          style='Subtitle.TLabel', wraplength=260)
        self.targets_summary.pack(anchor=tk.W, pady=(5, 0))
        
        # Action buttons
        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.analyze_btn = ttk.Button(btn_frame, text="🔍 Analyze Data", 
                                       command=self._analyze_data, style='Primary.TButton')
        self.analyze_btn.pack(fill=tk.X, pady=5)
        
        self.graph_btn = ttk.Button(btn_frame, text="📈 View Graph", 
                                     command=self._show_graph, state=tk.DISABLED)
        self.graph_btn.pack(fill=tk.X, pady=5)
        
        self.export_btn = ttk.Button(btn_frame, text="💾 Export Results", 
                                      command=self._export_results, state=tk.DISABLED)
        self.export_btn.pack(fill=tk.X, pady=5)
        
        # Right panel - Results with notebook tabs
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Notebook for different views
        self.results_notebook = ttk.Notebook(right_panel)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Overview Chart tab (DEFAULT - first tab)
        chart_frame = ttk.Frame(self.results_notebook, padding="10")
        self.results_notebook.add(chart_frame, text="  📊 Overview Chart  ")
        
        self.dur_fig = Figure(figsize=(8, 5), dpi=100)
        self.dur_canvas = FigureCanvasTkAgg(self.dur_fig, master=chart_frame)
        self.dur_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Duration Trends tab - with station selection
        dur_trend_frame = ttk.Frame(self.results_notebook, padding="10")
        self.results_notebook.add(dur_trend_frame, text="  📈 Duration Trends  ")
        
        # Left panel for station selection
        dur_st_select = ttk.LabelFrame(dur_trend_frame, text="Select Stations", padding="5")
        dur_st_select.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        dur_st_btn_frame = ttk.Frame(dur_st_select)
        dur_st_btn_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(dur_st_btn_frame, text="All", width=6,
                   command=lambda: self._select_all_dur_stations(True)).pack(side=tk.LEFT, padx=2)
        ttk.Button(dur_st_btn_frame, text="None", width=6,
                   command=lambda: self._select_all_dur_stations(False)).pack(side=tk.LEFT, padx=2)
        
        dur_st_list_frame = ttk.Frame(dur_st_select)
        dur_st_list_frame.pack(fill=tk.BOTH, expand=True)
        
        dur_st_canvas = tk.Canvas(dur_st_list_frame, width=150, height=300)
        dur_st_scrollbar = ttk.Scrollbar(dur_st_list_frame, orient="vertical", command=dur_st_canvas.yview)
        self.dur_st_checkbox_frame = ttk.Frame(dur_st_canvas)
        
        self.dur_st_checkbox_frame.bind("<Configure>",
            lambda e: dur_st_canvas.configure(scrollregion=dur_st_canvas.bbox("all")))
        dur_st_canvas.create_window((0, 0), window=self.dur_st_checkbox_frame, anchor="nw")
        dur_st_canvas.configure(yscrollcommand=dur_st_scrollbar.set)
        
        dur_st_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        dur_st_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.dur_st_trend_vars = {}
        
        # Chart area
        dur_trend_chart_frame = ttk.Frame(dur_trend_frame)
        dur_trend_chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.dur_trend_fig = Figure(figsize=(8, 5), dpi=100)
        self.dur_trend_canvas = FigureCanvasTkAgg(self.dur_trend_fig, master=dur_trend_chart_frame)
        self.dur_trend_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Operator Trends tab - with station selection for operator performance
        op_trend_frame = ttk.Frame(self.results_notebook, padding="10")
        self.results_notebook.add(op_trend_frame, text="  👥 Operator Trends  ")
        
        # Top panel for station selection
        op_station_frame = ttk.Frame(op_trend_frame)
        op_station_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(op_station_frame, text="Select Station:", font=('Segoe UI', 11, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        self.dur_op_station_var = tk.StringVar(value="")
        self.dur_op_station_combo = ttk.Combobox(op_station_frame, textvariable=self.dur_op_station_var,
                                                  values=[], state="readonly", width=35)
        self.dur_op_station_combo.pack(side=tk.LEFT, padx=5)
        self.dur_op_station_combo.bind('<<ComboboxSelected>>', lambda e: self._update_dur_operator_trend())
        
        ttk.Label(op_station_frame, text="(Shows operator duration trends for selected station)",
                  style='Subtitle.TLabel').pack(side=tk.LEFT, padx=10)
        
        # Left panel for operator selection
        dur_op_select = ttk.LabelFrame(op_trend_frame, text="Select Operators", padding="5")
        dur_op_select.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        dur_op_btn_frame = ttk.Frame(dur_op_select)
        dur_op_btn_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(dur_op_btn_frame, text="All", width=6,
                   command=lambda: self._select_all_dur_operators(True)).pack(side=tk.LEFT, padx=2)
        ttk.Button(dur_op_btn_frame, text="None", width=6,
                   command=lambda: self._select_all_dur_operators(False)).pack(side=tk.LEFT, padx=2)
        
        dur_op_list_frame = ttk.Frame(dur_op_select)
        dur_op_list_frame.pack(fill=tk.BOTH, expand=True)
        
        dur_op_canvas = tk.Canvas(dur_op_list_frame, width=150, height=250)
        dur_op_scrollbar = ttk.Scrollbar(dur_op_list_frame, orient="vertical", command=dur_op_canvas.yview)
        self.dur_op_checkbox_frame = ttk.Frame(dur_op_canvas)
        
        self.dur_op_checkbox_frame.bind("<Configure>",
            lambda e: dur_op_canvas.configure(scrollregion=dur_op_canvas.bbox("all")))
        dur_op_canvas.create_window((0, 0), window=self.dur_op_checkbox_frame, anchor="nw")
        dur_op_canvas.configure(yscrollcommand=dur_op_scrollbar.set)
        
        dur_op_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        dur_op_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.dur_op_trend_vars = {}
        
        # Chart area
        op_trend_chart_frame = ttk.Frame(op_trend_frame)
        op_trend_chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.op_dur_trend_fig = Figure(figsize=(8, 5), dpi=100)
        self.op_dur_trend_canvas = FigureCanvasTkAgg(self.op_dur_trend_fig, master=op_trend_chart_frame)
        self.op_dur_trend_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Results tab (LAST tab)
        text_frame = ttk.Frame(self.results_notebook, padding="10")
        self.results_notebook.add(text_frame, text="  📋 Results  ")
        
        self.results_text = tk.Text(text_frame, wrap=tk.WORD, font=('Consolas', 12),
                                    bg='#1e293b', fg='#e2e8f0', insertbackground='white')
        results_scroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scroll.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    def refresh_data(self):
        """Refresh UI when data is loaded"""
        # Update filter summary
        stats = self.data_manager.get_filter_stats()
        if stats:
            min_display = f"{stats['min_seconds']:.0f}s" if stats['min_seconds'] < 60 else f"{stats['min_seconds']/60:.1f}min"
            max_display = f"{stats['max_seconds']:.0f}s" if stats['max_seconds'] < 60 else f"{stats['max_seconds']/60:.1f}min"
            self.filter_summary.config(
                text=f"Range: {min_display} - {max_display}\n"
                     f"Using {stats['filtered']:,} of {stats['total']:,} records"
            )
        else:
            self.filter_summary.config(text="No data loaded")
        
        # Update summary of target times being used
        if self.data_manager.machines:
            num_targets = len(self.data_manager.target_times)
            num_machines = len(self.data_manager.machines)
            self.targets_summary.config(
                text=f"Using {num_targets}/{num_machines} station targets.\n"
                     f"Default: 30s for unset stations."
            )
        else:
            self.targets_summary.config(text="No data loaded")
    
    def _analyze_data(self):
        if self.data_manager.df is None:
            messagebox.showwarning("No Data", "Please load a data file first.")
            return
        
        try:
            # Use shared target times from data manager
            target_times = {}
            for machine in self.data_manager.machines:
                target_times[machine] = self.data_manager.get_target_time(machine)
            
            # Use centralized filtered data from Data Filters tab
            df = self.data_manager.get_filtered_df()
            
            # Perform analysis
            results = []
            self.analysis_summary = {}
            
            # Show filter info
            stats = self.data_manager.get_filter_stats()
            if stats and stats['removed'] > 0:
                results.append(f"ℹ️ Using filtered data: {stats['filtered']:,} records (filtered out {stats['removed']:,})\n")
            
            for machine in self.data_manager.machines:
                target_time = target_times[machine]
                machine_data = df[df['Machine Name'] == machine]
                
                if len(machine_data) == 0:
                    continue
                
                results.append(f"\n{'='*60}")
                results.append(f"MACHINE: {machine}")
                results.append(f"Target: {target_time}s | Records: {len(machine_data)}")
                results.append(f"{'='*60}\n")
                
                self.analysis_summary[machine] = {}
                
                for operator, op_data in machine_data.groupby('Operator Name'):
                    total_records = len(op_data)
                    avg_duration = op_data['Duration_Seconds'].mean()
                    meets_target = len(op_data[op_data['Duration_Seconds'] <= target_time])
                    meets_pct = (meets_target / total_records * 100) if total_records > 0 else 0
                    
                    status = "✅ ON TARGET" if avg_duration <= target_time else "⚠️ OVER TARGET"
                    
                    results.append(f"  {operator}")
                    results.append(f"    Records: {total_records} | Avg: {avg_duration:.1f}s | Target Met: {meets_pct:.1f}%")
                    results.append(f"    Status: {status}\n")
                    
                    self.analysis_summary[machine][operator] = {
                        'avg_duration': avg_duration,
                        'total_records': total_records,
                        'meets_pct': meets_pct,
                        'target_time': target_time
                    }
            
            # Display results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, "\n".join(results))
            
            self.graph_btn.config(state=tk.NORMAL)
            self.export_btn.config(state=tk.NORMAL)
            
            # Populate trend checkboxes (all unchecked initially)
            self._populate_dur_trend_checkboxes()
            
            # Update overview chart (trend charts will be empty until selections made)
            self._update_overview_chart()
            self._update_duration_trend_chart()
            self._update_operator_duration_trend_chart()
            
            messagebox.showinfo("Complete", "Analysis completed successfully!\n\n"
                                           "Use the checkboxes in trend tabs to select which to display.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
    
    def _update_overview_chart(self):
        """Update the overview bar chart showing avg duration by machine"""
        if not self.analysis_summary:
            return
        
        self.dur_fig.clear()
        ax = self.dur_fig.add_subplot(111)
        
        machines = []
        avg_durations = []
        target_times = []
        
        for machine, operators in self.analysis_summary.items():
            if operators:
                avg_dur = np.mean([op['avg_duration'] for op in operators.values()])
                target = list(operators.values())[0]['target_time']
                machines.append(machine)
                avg_durations.append(avg_dur)
                target_times.append(target)
        
        if not machines:
            return
        
        # Color based on whether meeting target
        colors = ['#10b981' if avg <= target else '#ef4444' 
                  for avg, target in zip(avg_durations, target_times)]
        
        x = np.arange(len(machines))
        bars = ax.bar(x, avg_durations, color=colors, alpha=0.8, edgecolor='black', label='Avg Duration')
        
        # Plot target times as markers
        ax.scatter(x, target_times, color='#2563eb', marker='_', s=200, linewidths=3, 
                   zorder=5, label='Target Time')
        
        # Add value labels
        for bar, val, target in zip(bars, avg_durations, target_times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}s', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels([m[:20] for m in machines], rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Machine', fontweight='bold')
        ax.set_ylabel('Duration (seconds)', fontweight='bold')
        ax.set_title('Average Duration by Machine', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        self.dur_fig.tight_layout()
        self.dur_canvas.draw()
    
    def _update_duration_trend_chart(self):
        """Update duration trends over time by machine - only selected stations"""
        self.dur_trend_fig.clear()
        ax = self.dur_trend_fig.add_subplot(111)
        
        # Get selected stations
        selected_stations = [st for st, var in self.dur_st_trend_vars.items() if var.get()]
        
        if not selected_stations:
            ax.text(0.5, 0.5, 'Select stations from the list\nto display their trends.',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title('Duration Trends Over Time', fontsize=14, fontweight='bold')
            self.dur_trend_fig.tight_layout()
            self.dur_trend_canvas.draw()
            return
        
        df = self.data_manager.get_filtered_df()
        if df is None or 'ParsedDate' not in df.columns:
            ax.text(0.5, 0.5, 'No date data available for trends.\nEnsure your data has a Date column.',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title('Duration Trends Over Time', fontsize=14, fontweight='bold')
            self.dur_trend_fig.tight_layout()
            self.dur_trend_canvas.draw()
            return
        
        df = df.dropna(subset=['ParsedDate'])
        if len(df) == 0:
            return
        
        df['DateOnly'] = df['ParsedDate'].dt.date
        
        colors = plt.cm.tab10.colors
        
        for i, machine in enumerate(selected_stations):
            m_data = df[df['Machine Name'] == machine]
            
            if len(m_data) == 0:
                continue
            
            dates = []
            avg_durs = []
            for date, date_group in m_data.groupby('DateOnly'):
                avg_dur = date_group['Duration_Seconds'].mean()
                dates.append(date)
                avg_durs.append(avg_dur)
            
            if len(dates) > 0:
                color = colors[i % len(colors)]
                cell = self.data_manager.extract_cell_number(machine)
                short_name = machine.replace(cell, '').strip() if cell else machine
                
                ax.plot(dates, avg_durs, marker='o', markersize=5, linewidth=1.5, 
                       label=short_name[:15], color=color, alpha=0.8)
                
                # Add trend line
                if len(dates) >= 2:
                    x_numeric = np.arange(len(dates))
                    z = np.polyfit(x_numeric, avg_durs, 1)
                    p = np.poly1d(z)
                    ax.plot(dates, p(x_numeric), linestyle='--', color=color, alpha=0.4, linewidth=1)
                
                # Add target time marker for this station
                target = self.data_manager.get_target_time(machine)
                ax.axhline(y=target, color=color, linestyle=':', linewidth=1.5, alpha=0.6)
        
        # Add legend note about dotted lines being targets
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_ylabel('Average Duration (seconds)', fontweight='bold')
        ax.set_title(f'Duration Trends ({len(selected_stations)} stations) - Dotted lines = Targets', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=7, ncol=2)
        ax.grid(axis='both', alpha=0.3)
        
        self.dur_trend_fig.autofmt_xdate(rotation=45)
        self.dur_trend_fig.tight_layout()
        self.dur_trend_canvas.draw()
    
    def _update_operator_duration_trend_chart(self):
        """Update duration trends over time by operator - filtered by selected station"""
        self.op_dur_trend_fig.clear()
        ax = self.op_dur_trend_fig.add_subplot(111)
        
        # Get selected station
        selected_station = self.dur_op_station_var.get() if hasattr(self, 'dur_op_station_var') else ""
        
        # Get selected operators
        selected_ops = [op for op, var in self.dur_op_trend_vars.items() if var.get()]
        
        if not selected_station:
            ax.text(0.5, 0.5, 'Select a station from the dropdown\nto display operator trends.',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title('Operator Duration Trends by Station', fontsize=14, fontweight='bold')
            self.op_dur_trend_fig.tight_layout()
            self.op_dur_trend_canvas.draw()
            return
        
        if not selected_ops:
            ax.text(0.5, 0.5, 'Select operators from the list\nto display their trends.',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title(f'Operator Duration Trends - {selected_station[:30]}', fontsize=14, fontweight='bold')
            self.op_dur_trend_fig.tight_layout()
            self.op_dur_trend_canvas.draw()
            return
        
        df = self.data_manager.get_filtered_df()
        if df is None or 'ParsedDate' not in df.columns:
            ax.text(0.5, 0.5, 'No date data available for trends.\nEnsure your data has a Date column.',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title('Operator Duration Trends by Station', fontsize=14, fontweight='bold')
            self.op_dur_trend_fig.tight_layout()
            self.op_dur_trend_canvas.draw()
            return
        
        # Filter by selected station
        df = df[df['Machine Name'] == selected_station]
        
        df = df.dropna(subset=['ParsedDate'])
        if len(df) == 0:
            ax.text(0.5, 0.5, f'No data for station: {selected_station[:30]}',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title(f'Operator Duration Trends - {selected_station[:30]}', fontsize=14, fontweight='bold')
            self.op_dur_trend_fig.tight_layout()
            self.op_dur_trend_canvas.draw()
            return
        
        df['DateOnly'] = df['ParsedDate'].dt.date
        
        # Get target time for this station
        target = self.data_manager.get_target_time(selected_station)
        
        colors = plt.cm.tab10.colors
        
        for i, operator in enumerate(selected_ops):
            op_data = df[df['Operator Name'] == operator]
            
            if len(op_data) == 0:
                continue
            
            dates = []
            avg_durs = []
            for date, date_group in op_data.groupby('DateOnly'):
                avg_dur = date_group['Duration_Seconds'].mean()
                dates.append(date)
                avg_durs.append(avg_dur)
            
            if len(dates) > 0:
                color = colors[i % len(colors)]
                
                ax.plot(dates, avg_durs, marker='o', markersize=5, linewidth=1.5, 
                       label=operator[:15], color=color, alpha=0.8)
                
                # Add trend line
                if len(dates) >= 2:
                    x_numeric = np.arange(len(dates))
                    z = np.polyfit(x_numeric, avg_durs, 1)
                    p = np.poly1d(z)
                    ax.plot(dates, p(x_numeric), linestyle='--', color=color, alpha=0.4, linewidth=1)
        
        # Add target time baseline
        ax.axhline(y=target, color='red', linestyle='--', linewidth=2, label=f'Target: {target:.0f}s')
        
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_ylabel('Average Duration (seconds)', fontweight='bold')
        
        # Shorten station name for title
        cell = self.data_manager.extract_cell_number(selected_station)
        short_name = selected_station.replace(cell, '').strip()[:25] if cell else selected_station[:25]
        ax.set_title(f'Operator Duration Trends - {short_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=7, ncol=2)
        ax.grid(axis='both', alpha=0.3)
        
        self.op_dur_trend_fig.autofmt_xdate(rotation=45)
        self.op_dur_trend_fig.tight_layout()
        self.op_dur_trend_canvas.draw()
    
    def _update_dur_operator_trend(self):
        """Wrapper for updating operator trend chart when station selection changes"""
        self._update_operator_duration_trend_chart()
    
    def _populate_dur_trend_checkboxes(self):
        """Populate checkboxes for station and operator selection in Duration Analysis"""
        # Clear existing checkboxes
        for widget in self.dur_st_checkbox_frame.winfo_children():
            widget.destroy()
        for widget in self.dur_op_checkbox_frame.winfo_children():
            widget.destroy()
        
        self.dur_st_trend_vars = {}
        self.dur_op_trend_vars = {}
        
        # Get stations
        stations = sorted(self.data_manager.machines)
        for station in stations:
            var = tk.BooleanVar(value=False)
            self.dur_st_trend_vars[station] = var
            cell = self.data_manager.extract_cell_number(station)
            display_name = station.replace(cell, '').strip()[:18] if cell else station[:18]
            cb = ttk.Checkbutton(self.dur_st_checkbox_frame, text=display_name, variable=var,
                                 command=self._update_duration_trend_chart)
            cb.pack(anchor=tk.W, pady=1)
        
        # Populate station combobox for operator trends
        if hasattr(self, 'dur_op_station_combo'):
            self.dur_op_station_combo['values'] = stations
            if stations:
                self.dur_op_station_var.set(stations[0])
        
        # Get operators
        for op in self.data_manager.operators:
            var = tk.BooleanVar(value=False)
            self.dur_op_trend_vars[op] = var
            cb = ttk.Checkbutton(self.dur_op_checkbox_frame, text=op[:20], variable=var,
                                 command=self._update_operator_duration_trend_chart)
            cb.pack(anchor=tk.W, pady=1)
    
    def _select_all_dur_stations(self, select=True):
        """Select or deselect all stations for duration trends"""
        for var in self.dur_st_trend_vars.values():
            var.set(select)
        self._update_duration_trend_chart()
    
    def _select_all_dur_operators(self, select=True):
        """Select or deselect all operators for duration trends"""
        for var in self.dur_op_trend_vars.values():
            var.set(select)
        self._update_operator_duration_trend_chart()
    
    def _show_graph(self):
        if not self.analysis_summary:
            return
        
        graph_window = tk.Toplevel(self)
        graph_window.title("Duration Analysis Graph")
        graph_window.geometry("1100x700")
        
        # Controls
        control_frame = ttk.Frame(graph_window, padding="10")
        control_frame.pack(fill=tk.X)
        
        ttk.Label(control_frame, text="Machine:").pack(side=tk.LEFT, padx=5)
        machine_var = tk.StringVar(value=list(self.analysis_summary.keys())[0])
        machine_combo = ttk.Combobox(control_frame, textvariable=machine_var,
                                      values=list(self.analysis_summary.keys()), state="readonly", width=25)
        machine_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control_frame, text="Metric:").pack(side=tk.LEFT, padx=5)
        metric_var = tk.StringVar(value="Average Duration")
        metric_combo = ttk.Combobox(control_frame, textvariable=metric_var,
                                     values=["Average Duration", "Target Met %", "Total Records"],
                                     state="readonly", width=18)
        metric_combo.pack(side=tk.LEFT, padx=5)
        
        # Graph
        graph_frame = ttk.Frame(graph_window, padding="10")
        graph_frame.pack(fill=tk.BOTH, expand=True)
        
        fig = Figure(figsize=(11, 5.5), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        def update_graph(*args):
            machine = machine_var.get()
            metric = metric_var.get()
            
            if machine not in self.analysis_summary:
                return
            
            fig.clear()
            ax = fig.add_subplot(111)
            
            operators = list(self.analysis_summary[machine].keys())
            if metric == "Average Duration":
                values = [self.analysis_summary[machine][op]['avg_duration'] for op in operators]
                target = self.analysis_summary[machine][operators[0]]['target_time']
                ax.axhline(y=target, color='red', linestyle='--', linewidth=2, label=f'Target: {target}s')
                ylabel = 'Duration (seconds)'
            elif metric == "Target Met %":
                values = [self.analysis_summary[machine][op]['meets_pct'] for op in operators]
                ylabel = 'Target Met (%)'
            else:
                values = [self.analysis_summary[machine][op]['total_records'] for op in operators]
                ylabel = 'Total Records'
            
            colors = ['#10b981' if metric == "Target Met %" and v >= 80 else '#2563eb' for v in values]
            bars = ax.bar(range(len(operators)), values, color=colors, alpha=0.8, edgecolor='black')
            
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                       f'{val:.1f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_xticks(range(len(operators)))
            ax.set_xticklabels(operators, rotation=45, ha='right')
            ax.set_xlabel('Operator', fontweight='bold')
            ax.set_ylabel(ylabel, fontweight='bold')
            ax.set_title(f'{machine} - {metric}', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            if metric == "Average Duration":
                ax.legend()
            
            fig.tight_layout()
            canvas.draw()
        
        ttk.Button(control_frame, text="Update", command=update_graph).pack(side=tk.LEFT, padx=20)
        machine_combo.bind("<<ComboboxSelected>>", update_graph)
        metric_combo.bind("<<ComboboxSelected>>", update_graph)
        
        update_graph()
    
    def _export_results(self):
        if not self.analysis_summary:
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")]
        )
        
        if file_path:
            export_data = []
            for machine, operators in self.analysis_summary.items():
                for operator, data in operators.items():
                    export_data.append({
                        'Machine': machine,
                        'Operator': operator,
                        'Avg Duration (s)': round(data['avg_duration'], 2),
                        'Target Time (s)': data['target_time'],
                        'Records': data['total_records'],
                        'Target Met %': round(data['meets_pct'], 1)
                    })
            
            df = pd.DataFrame(export_data)
            if file_path.endswith('.csv'):
                df.to_csv(file_path, index=False)
            else:
                df.to_excel(file_path, index=False)
            
            messagebox.showinfo("Success", f"Results exported to {file_path}")


class EfficiencyCalculatorTab(ttk.Frame):
    """Efficiency Calculator Tab"""
    
    def __init__(self, parent, data_manager):
        super().__init__(parent)
        self.data_manager = data_manager
        self.results_df = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(header_frame, text="⚡ Efficiency Calculator", 
                  style='Title.TLabel').pack(side=tk.LEFT)
        ttk.Label(header_frame, text="Calculate operator efficiency ratios",
                  style='Subtitle.TLabel').pack(side=tk.LEFT, padx=(20, 0))
        
        # Content - two columns
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Controls (simplified - no target times here)
        left_panel = ttk.LabelFrame(content_frame, text="⚙️ Controls", padding="15")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Info about target times
        info_frame = ttk.Frame(left_panel)
        info_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(info_frame, text="ℹ️ Target times are set in the", 
                  style='Subtitle.TLabel').pack(anchor=tk.W)
        ttk.Label(info_frame, text="⏱️ Target Times tab", 
                  font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W)
        
        ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Target times summary
        self.targets_summary = ttk.Label(left_panel, text="No data loaded", 
                                          style='Subtitle.TLabel', wraplength=280)
        self.targets_summary.pack(fill=tk.X, pady=(0, 15))
        
        # Buttons
        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill=tk.X, pady=(15, 0))
        
        self.calc_btn = ttk.Button(btn_frame, text="🔢 Calculate Efficiency",
                                    command=self._calculate, style='Primary.TButton')
        self.calc_btn.pack(fill=tk.X, pady=5)
        
        self.chart_btn = ttk.Button(btn_frame, text="📊 Show Chart",
                                     command=self._show_chart, state=tk.DISABLED)
        self.chart_btn.pack(fill=tk.X, pady=5)
        
        self.export_btn = ttk.Button(btn_frame, text="💾 Export Results",
                                      command=self._export_results, state=tk.DISABLED)
        self.export_btn.pack(fill=tk.X, pady=5)
        
        # Right panel - Results with notebook tabs
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Notebook for different views
        self.results_notebook = ttk.Notebook(right_panel)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Overview Chart tab (DEFAULT - first tab)
        chart_frame = ttk.Frame(self.results_notebook, padding="10")
        self.results_notebook.add(chart_frame, text="  📊 Overview Chart  ")
        
        self.eff_fig = Figure(figsize=(8, 5), dpi=100)
        self.eff_canvas = FigureCanvasTkAgg(self.eff_fig, master=chart_frame)
        self.eff_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Operator Trends tab - with selection panel
        op_trend_frame = ttk.Frame(self.results_notebook, padding="10")
        self.results_notebook.add(op_trend_frame, text="  📈 Operator Trends  ")
        
        # Left panel for operator selection
        op_select_frame = ttk.LabelFrame(op_trend_frame, text="Select Operators", padding="5")
        op_select_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Select all/none buttons
        op_btn_frame = ttk.Frame(op_select_frame)
        op_btn_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(op_btn_frame, text="All", width=6, 
                   command=lambda: self._select_all_operators(True)).pack(side=tk.LEFT, padx=2)
        ttk.Button(op_btn_frame, text="None", width=6,
                   command=lambda: self._select_all_operators(False)).pack(side=tk.LEFT, padx=2)
        
        # Scrollable checkbox list for operators
        op_list_frame = ttk.Frame(op_select_frame)
        op_list_frame.pack(fill=tk.BOTH, expand=True)
        
        op_canvas = tk.Canvas(op_list_frame, width=150, height=300)
        op_scrollbar = ttk.Scrollbar(op_list_frame, orient="vertical", command=op_canvas.yview)
        self.op_checkbox_frame = ttk.Frame(op_canvas)
        
        self.op_checkbox_frame.bind("<Configure>", 
            lambda e: op_canvas.configure(scrollregion=op_canvas.bbox("all")))
        op_canvas.create_window((0, 0), window=self.op_checkbox_frame, anchor="nw")
        op_canvas.configure(yscrollcommand=op_scrollbar.set)
        
        op_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        op_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.op_trend_vars = {}  # Will hold BooleanVars for each operator
        
        # Chart area for operators
        op_chart_frame = ttk.Frame(op_trend_frame)
        op_chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.eff_trend_fig = Figure(figsize=(8, 5), dpi=100)
        self.eff_trend_canvas = FigureCanvasTkAgg(self.eff_trend_fig, master=op_chart_frame)
        self.eff_trend_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Machine/Station Trends tab - with single station selection
        machine_trend_frame = ttk.Frame(self.results_notebook, padding="10")
        self.results_notebook.add(machine_trend_frame, text="  🔧 Station Trends  ")
        
        # Top panel for station selection (single station at a time)
        st_select_frame = ttk.Frame(machine_trend_frame)
        st_select_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(st_select_frame, text="Select Station:", font=('Segoe UI', 11, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        self.station_trend_var = tk.StringVar(value="")
        self.station_trend_combo = ttk.Combobox(st_select_frame, textvariable=self.station_trend_var,
                                                 values=[], state="readonly", width=40)
        self.station_trend_combo.pack(side=tk.LEFT, padx=5)
        self.station_trend_combo.bind('<<ComboboxSelected>>', lambda e: self._update_station_trend_chart())
        
        # Chart area for station
        st_chart_frame = ttk.Frame(machine_trend_frame)
        st_chart_frame.pack(fill=tk.BOTH, expand=True)
        
        self.machine_trend_fig = Figure(figsize=(8, 5), dpi=100)
        self.machine_trend_canvas = FigureCanvasTkAgg(self.machine_trend_fig, master=st_chart_frame)
        self.machine_trend_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Results tab (LAST tab)
        table_frame = ttk.Frame(self.results_notebook, padding="10")
        self.results_notebook.add(table_frame, text="  📋 Results  ")
        
        # Treeview
        columns = ('Operator', 'Machine', 'Efficiency', 'Records', 'Avg Duration', 'Target')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=18)
        
        self.tree.heading('Operator', text='Operator')
        self.tree.heading('Machine', text='Machine')
        self.tree.heading('Efficiency', text='Efficiency')
        self.tree.heading('Records', text='Records')
        self.tree.heading('Avg Duration', text='Avg Duration')
        self.tree.heading('Target', text='Target')
        
        self.tree.column('Operator', width=130)
        self.tree.column('Machine', width=130)
        self.tree.column('Efficiency', width=90)
        self.tree.column('Records', width=70)
        self.tree.column('Avg Duration', width=100)
        self.tree.column('Target', width=70)
        
        tree_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    def refresh_data(self):
        """Refresh UI when data is loaded"""
        # Update summary of target times being used
        if self.data_manager.machines:
            num_targets = len(self.data_manager.target_times)
            num_machines = len(self.data_manager.machines)
            self.targets_summary.config(
                text=f"Using target times for {num_targets}/{num_machines} stations.\n"
                     f"Default: 30s for stations without custom targets."
            )
        else:
            self.targets_summary.config(text="No data loaded")
    
    def _calculate(self):
        if self.data_manager.df is None:
            messagebox.showwarning("No Data", "Please load a data file first.")
            return
        
        try:
            # Use shared target times from data manager
            target_times = {}
            for machine in self.data_manager.machines:
                target_times[machine] = self.data_manager.get_target_time(machine)
            
            if not target_times:
                messagebox.showerror("Error", "No stations found. Please load data first.")
                return
            
            # Use centralized filtered data
            df = self.data_manager.get_filtered_df()
            
            results = []
            grouped = df.groupby(['Operator Name', 'Machine Name'])
            
            for (operator, machine), group in grouped:
                if machine in target_times:
                    # Use centralized calculation method
                    metrics = self.data_manager.get_operator_station_efficiency(operator, machine, df)
                    if metrics:
                        results.append({
                            'Operator Name': operator,
                            'Machine Name': machine,
                            'Efficiency': f"{metrics['efficiency']:.1f}%",
                            'Efficiency_Value': metrics['efficiency'],
                            'Records': metrics['records'],
                            'Avg Duration': f"{metrics['avg_duration']:.1f}s",
                            'Target': metrics['target']
                        })
            
            self.results_df = pd.DataFrame(results)
            
            if self.results_df.empty:
                messagebox.showwarning("No Results", "No efficiency data could be calculated.")
                return
            
            # Display in treeview
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            for _, row in self.results_df.iterrows():
                self.tree.insert('', 'end', values=(
                    row['Operator Name'], row['Machine Name'], row['Efficiency'],
                    row['Records'], row['Avg Duration'], row['Target']
                ))
            
            self.chart_btn.config(state=tk.NORMAL)
            self.export_btn.config(state=tk.NORMAL)
            
            # Populate trend checkboxes (all unchecked initially)
            self._populate_trend_checkboxes()
            
            # Update overview chart (trend charts will be empty until selections made)
            self._update_overview_chart()
            self._update_operator_trend_chart()
            self._update_station_trend_chart()
            
            messagebox.showinfo("Complete", f"Calculated efficiency for {len(self.results_df)} combinations.\n\n"
                                           f"Use the checkboxes in Operator/Station Trends tabs to select which to display.")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def _update_overview_chart(self):
        """Update the overview bar chart"""
        if self.results_df is None or self.results_df.empty:
            return
        
        self.eff_fig.clear()
        ax = self.eff_fig.add_subplot(111)
        
        # Group by operator and get average efficiency
        op_effs = self.results_df.groupby('Operator Name')['Efficiency_Value'].mean().sort_values(ascending=False)
        
        colors = []
        for e in op_effs.values:
            if e >= 100:
                colors.append('#10b981')
            elif e >= 85:
                colors.append('#f59e0b')
            else:
                colors.append('#ef4444')
        
        bars = ax.barh(range(len(op_effs)), op_effs.values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add 100% target line
        ax.axvline(x=100, color='#2563eb', linestyle='--', linewidth=2, label='Target (100%)')
        
        # Add value labels
        for bar, val in zip(bars, op_effs.values):
            ax.text(val + 2, bar.get_y() + bar.get_height()/2, f'{val:.0f}%', 
                   va='center', fontsize=8, fontweight='bold')
        
        ax.set_yticks(range(len(op_effs)))
        ax.set_yticklabels(op_effs.index, fontsize=8)
        ax.set_xlabel('Efficiency (%)', fontweight='bold')
        ax.set_title('Operator Efficiency Overview', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        
        self.eff_fig.tight_layout()
        self.eff_canvas.draw()
    
    def _update_operator_trend_chart(self):
        """Update operator efficiency trend over time - only selected operators"""
        self.eff_trend_fig.clear()
        ax = self.eff_trend_fig.add_subplot(111)
        
        # Get selected operators
        selected_ops = [op for op, var in self.op_trend_vars.items() if var.get()]
        
        if not selected_ops:
            ax.text(0.5, 0.5, 'Select operators from the list\nto display their trends.',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title('Operator Efficiency Trends Over Time', fontsize=14, fontweight='bold')
            self.eff_trend_fig.tight_layout()
            self.eff_trend_canvas.draw()
            return
        
        trends = self.data_manager.get_trend_data(group_by='operator')
        
        if not trends:
            ax.text(0.5, 0.5, 'No date data available for trends.\nEnsure your data has a Date column.',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title('Operator Efficiency Trends Over Time', fontsize=14, fontweight='bold')
            self.eff_trend_fig.tight_layout()
            self.eff_trend_canvas.draw()
            return
        
        colors = plt.cm.tab10.colors
        
        for i, operator in enumerate(selected_ops):
            if operator not in trends:
                continue
            data = trends[operator]
            if len(data['dates']) > 0:
                color = colors[i % len(colors)]
                dates = data['dates']
                effs = data['efficiencies']
                
                ax.plot(dates, effs, marker='o', markersize=5, linewidth=1.5, 
                       label=operator[:15], color=color, alpha=0.8)
                
                # Add trend line
                if len(dates) >= 2:
                    x_numeric = np.arange(len(dates))
                    z = np.polyfit(x_numeric, effs, 1)
                    p = np.poly1d(z)
                    ax.plot(dates, p(x_numeric), linestyle='--', color=color, alpha=0.4, linewidth=1)
        
        ax.axhline(y=100, color='#64748b', linestyle=':', linewidth=2, label='Target')
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_ylabel('Efficiency (%)', fontweight='bold')
        ax.set_title(f'Operator Efficiency Trends ({len(selected_ops)} selected)', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=7, ncol=2)
        ax.grid(axis='both', alpha=0.3)
        
        self.eff_trend_fig.autofmt_xdate(rotation=45)
        self.eff_trend_fig.tight_layout()
        self.eff_trend_canvas.draw()
    
    def _update_station_trend_chart(self):
        """Update station/machine efficiency trend over time - single station"""
        self.machine_trend_fig.clear()
        ax = self.machine_trend_fig.add_subplot(111)
        
        # Get selected station from combobox
        selected_station = self.station_trend_var.get()
        
        if not selected_station:
            ax.text(0.5, 0.5, 'Select a station from the dropdown\nto display its trend.',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title('Station Efficiency Trend Over Time', fontsize=14, fontweight='bold')
            self.machine_trend_fig.tight_layout()
            self.machine_trend_canvas.draw()
            return
        
        selected_stations = [selected_station]
        
        df = self.data_manager.get_filtered_df()
        if df is None or 'ParsedDate' not in df.columns:
            ax.text(0.5, 0.5, 'No date data available for trends.\nEnsure your data has a Date column.',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title('Station Efficiency Trends Over Time', fontsize=14, fontweight='bold')
            self.machine_trend_fig.tight_layout()
            self.machine_trend_canvas.draw()
            return
        
        df = df.dropna(subset=['ParsedDate'])
        if len(df) == 0:
            return
        
        df['DateOnly'] = df['ParsedDate'].dt.date
        
        colors = plt.cm.tab10.colors
        
        for i, machine in enumerate(selected_stations):
            target = self.data_manager.get_target_time(machine)
            m_data = df[df['Machine Name'] == machine]
            
            if len(m_data) == 0:
                continue
            
            dates = []
            effs = []
            for date, date_group in m_data.groupby('DateOnly'):
                valid_dur = date_group['Duration_Seconds'][date_group['Duration_Seconds'] > 0]
                if len(valid_dur) > 0:
                    # Use correct efficiency: target / avg_duration
                    avg_dur = valid_dur.mean()
                    eff = (target / avg_dur) * 100 if avg_dur > 0 else 0
                    dates.append(date)
                    effs.append(eff)
            
            if len(dates) > 0:
                color = colors[i % len(colors)]
                cell = self.data_manager.extract_cell_number(machine)
                short_name = machine.replace(cell, '').strip() if cell else machine
                
                ax.plot(dates, effs, marker='o', markersize=5, linewidth=1.5, 
                       label=short_name[:15], color=color, alpha=0.8)
                
                if len(dates) >= 2:
                    x_numeric = np.arange(len(dates))
                    z = np.polyfit(x_numeric, effs, 1)
                    p = np.poly1d(z)
                    ax.plot(dates, p(x_numeric), linestyle='--', color=color, alpha=0.4, linewidth=1)
        
        ax.axhline(y=100, color='#64748b', linestyle=':', linewidth=2, label='Target')
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_ylabel('Efficiency (%)', fontweight='bold')
        ax.set_title(f'Station Efficiency Trends ({len(selected_stations)} selected)', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=7, ncol=2)
        ax.grid(axis='both', alpha=0.3)
        
        self.machine_trend_fig.autofmt_xdate(rotation=45)
        self.machine_trend_fig.tight_layout()
        self.machine_trend_canvas.draw()
    
    def _populate_trend_checkboxes(self):
        """Populate checkboxes for operator selection and station combobox"""
        # Clear existing checkboxes
        for widget in self.op_checkbox_frame.winfo_children():
            widget.destroy()
        
        self.op_trend_vars = {}
        
        # Get operators from results
        if self.results_df is not None and not self.results_df.empty:
            operators = sorted(self.results_df['Operator Name'].unique())
            for op in operators:
                var = tk.BooleanVar(value=False)  # Initially unchecked
                self.op_trend_vars[op] = var
                cb = ttk.Checkbutton(self.op_checkbox_frame, text=op[:20], variable=var,
                                     command=self._update_operator_trend_chart)
                cb.pack(anchor=tk.W, pady=1)
        
        # Populate station combobox (single selection)
        stations = sorted(self.data_manager.machines)
        self.station_trend_combo['values'] = stations
        if stations:
            self.station_trend_var.set(stations[0])
            self._update_station_trend_chart()
    
    def _select_all_operators(self, select=True):
        """Select or deselect all operators"""
        for var in self.op_trend_vars.values():
            var.set(select)
        self._update_operator_trend_chart()
    
    def _show_chart(self):
        if self.results_df is None or self.results_df.empty:
            return
        
        chart_window = tk.Toplevel(self)
        chart_window.title("Efficiency Chart")
        chart_window.geometry("1100x700")
        
        # Controls
        control_frame = ttk.Frame(chart_window, padding="10")
        control_frame.pack(fill=tk.X)
        
        all_operators = sorted(self.results_df['Operator Name'].unique().tolist())
        
        ttk.Label(control_frame, text="Select Operators:", font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT)
        
        operator_vars = {}
        for op in all_operators[:8]:  # Show first 8 operators
            var = tk.BooleanVar(value=True)
            operator_vars[op] = var
            ttk.Checkbutton(control_frame, text=op, variable=var).pack(side=tk.LEFT, padx=5)
        
        # Graph
        graph_frame = ttk.Frame(chart_window, padding="10")
        graph_frame.pack(fill=tk.BOTH, expand=True)
        
        fig = Figure(figsize=(11, 5.5), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        def update_chart():
            selected = [op for op, var in operator_vars.items() if var.get()]
            if not selected:
                messagebox.showwarning("No Selection", "Select at least one operator")
                return
            
            filtered = self.results_df[self.results_df['Operator Name'].isin(selected)]
            machines = sorted(filtered['Machine Name'].unique())
            
            fig.clear()
            ax = fig.add_subplot(111)
            
            x = np.arange(len(machines))
            width = 0.8 / len(selected)
            colors = plt.cm.tab10(np.linspace(0, 1, len(selected)))
            
            for i, operator in enumerate(selected):
                op_data = filtered[filtered['Operator Name'] == operator]
                efficiencies = []
                for machine in machines:
                    m_data = op_data[op_data['Machine Name'] == machine]
                    efficiencies.append(m_data['Efficiency_Value'].values[0] if not m_data.empty else 0)
                
                pos = x + (i - len(selected)/2 + 0.5) * width
                bars = ax.bar(pos, efficiencies, width, label=operator, color=colors[i], 
                             edgecolor='black', alpha=0.8)
                
                for bar, val in zip(bars, efficiencies):
                    if val > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                               f'{val:.0f}%', ha='center', va='bottom', fontsize=8)
            
            ax.axhline(y=100, color='red', linestyle='--', linewidth=2, label='Target (100%)')
            ax.set_xticks(x)
            ax.set_xticklabels(machines, rotation=45, ha='right')
            ax.set_xlabel('Machine', fontweight='bold')
            ax.set_ylabel('Efficiency (%)', fontweight='bold')
            ax.set_title('Operator Efficiency by Machine', fontsize=14, fontweight='bold')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            
            fig.tight_layout()
            canvas.draw()
        
        ttk.Button(control_frame, text="Update Chart", command=update_chart).pack(side=tk.LEFT, padx=20)
        update_chart()
    
    def _export_results(self):
        if self.results_df is None or self.results_df.empty:
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")]
        )
        
        if file_path:
            export_df = self.results_df[['Operator Name', 'Machine Name', 'Efficiency', 
                                          'Records', 'Avg Duration', 'Target']]
            if file_path.endswith('.csv'):
                export_df.to_csv(file_path, index=False)
            else:
                export_df.to_excel(file_path, index=False)
            
            messagebox.showinfo("Success", f"Results exported to {file_path}")


class OperatorsTab(ttk.Frame):
    """Operators Browser Tab - View individual operator stats"""
    
    def __init__(self, parent, data_manager):
        super().__init__(parent)
        self.data_manager = data_manager
        self.current_view = "list"  # "list" or "detail"
        self.selected_operator = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        # Main container
        self.main_frame = ttk.Frame(self, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header with back button (hidden initially)
        self.header_frame = ttk.Frame(self.main_frame)
        self.header_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.back_btn = ttk.Button(self.header_frame, text="← Back to Operators", 
                                    command=self._show_list_view, style='Modern.TButton')
        
        self.title_label = ttk.Label(self.header_frame, text="👥 Operators", style='Title.TLabel')
        self.title_label.pack(side=tk.LEFT)
        
        self.subtitle_label = ttk.Label(self.header_frame, text="Click an operator to view detailed stats",
                                         style='Subtitle.TLabel')
        self.subtitle_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # Content frame (will switch between list and detail views)
        self.content_frame = ttk.Frame(self.main_frame)
        self.content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initialize list view
        self._create_list_view()
    
    def _create_list_view(self):
        """Create the operator list/grid view"""
        # Clear content
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Search/filter bar
        filter_frame = ttk.Frame(self.content_frame)
        filter_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(filter_frame, text="🔍 Search:", style='Header.TLabel').pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self._filter_operators)
        search_entry = ttk.Entry(filter_frame, textvariable=self.search_var, width=30)
        search_entry.pack(side=tk.LEFT, padx=(10, 20))
        
        self.count_label = ttk.Label(filter_frame, text="0 operators", style='Subtitle.TLabel')
        self.count_label.pack(side=tk.LEFT)
        
        # Scrollable frame for operator cards
        canvas_frame = ttk.Frame(self.content_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.list_canvas = tk.Canvas(canvas_frame, highlightthickness=0, bg='#f8fafc')
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.list_canvas.yview)
        self.operators_frame = ttk.Frame(self.list_canvas)
        
        self.operators_frame.bind("<Configure>",
            lambda e: self.list_canvas.configure(scrollregion=self.list_canvas.bbox("all")))
        
        self.canvas_window = self.list_canvas.create_window((0, 0), window=self.operators_frame, anchor="nw")
        self.list_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Bind canvas resize to update frame width
        self.list_canvas.bind('<Configure>', self._on_canvas_configure)
        
        # Bind mouse wheel only when over canvas
        self.list_canvas.bind("<Enter>", lambda e: self.list_canvas.bind_all("<MouseWheel>", self._on_mousewheel))
        self.list_canvas.bind("<Leave>", lambda e: self.list_canvas.unbind_all("<MouseWheel>"))
        
        self.list_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate operators
        self._populate_operators()
    
    def _on_canvas_configure(self, event):
        """Update the operators frame width when canvas resizes"""
        self.list_canvas.itemconfig(self.canvas_window, width=event.width)
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        if self.current_view == "list":
            self.list_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def _populate_operators(self, filter_text=""):
        """Populate the operators grid"""
        # Clear existing
        for widget in self.operators_frame.winfo_children():
            widget.destroy()
        
        if self.data_manager.df is None:
            ttk.Label(self.operators_frame, text="No data loaded. Please load a file first.",
                     style='Subtitle.TLabel').pack(pady=50)
            return
        
        operators = self.data_manager.operators
        
        # Filter if search text provided
        if filter_text:
            operators = [op for op in operators if filter_text.lower() in op.lower()]
        
        self.count_label.config(text=f"{len(operators)} operators")
        
        if not operators:
            ttk.Label(self.operators_frame, text="No operators found.",
                     style='Subtitle.TLabel').pack(pady=50)
            return
        
        # Create grid of operator cards (3 columns)
        columns = 3
        for i, operator in enumerate(operators):
            row = i // columns
            col = i % columns
            
            self._create_operator_card(operator, row, col)
    
    def _create_operator_card(self, operator, row, col):
        """Create a clickable operator card"""
        # Card frame
        card = tk.Frame(self.operators_frame, bg='white', relief='solid', bd=1,
                       cursor='hand2')
        card.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')
        
        # Configure grid weights for even distribution
        self.operators_frame.columnconfigure(col, weight=1)
        
        # Get operator stats preview (use filtered data)
        df = self.data_manager.get_filtered_df()
        op_data = df[df['Operator Name'] == operator] if df is not None else pd.DataFrame()
        record_count = len(op_data)
        machine_count = op_data['Machine Name'].nunique()
        avg_duration = op_data['Duration_Seconds'].mean() if 'Duration_Seconds' in op_data.columns else 0
        
        # Card content
        content_frame = tk.Frame(card, bg='white', padx=15, pady=12)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Operator name
        name_label = tk.Label(content_frame, text=f"👤 {operator}", 
                             font=('Segoe UI', 12, 'bold'), bg='white', fg='#1e293b',
                             anchor='w')
        name_label.pack(fill=tk.X)
        
        # Stats row
        stats_frame = tk.Frame(content_frame, bg='white')
        stats_frame.pack(fill=tk.X, pady=(8, 0))
        
        tk.Label(stats_frame, text=f"📊 {record_count} records", 
                font=('Segoe UI', 9), bg='white', fg='#64748b').pack(side=tk.LEFT)
        tk.Label(stats_frame, text=f"🔧 {machine_count} machines", 
                font=('Segoe UI', 9), bg='white', fg='#64748b').pack(side=tk.LEFT, padx=(15, 0))
        
        # Avg duration
        tk.Label(content_frame, text=f"⏱️ Avg: {avg_duration:.1f}s", 
                font=('Segoe UI', 9), bg='white', fg='#64748b',
                anchor='w').pack(fill=tk.X, pady=(5, 0))
        
        # Click to view text
        tk.Label(content_frame, text="Click to view details →", 
                font=('Segoe UI', 8, 'italic'), bg='white', fg='#94a3b8',
                anchor='w').pack(fill=tk.X, pady=(8, 0))
        
        # Bind click events to all widgets in card
        for widget in [card, content_frame, name_label, stats_frame] + list(content_frame.winfo_children()) + list(stats_frame.winfo_children()):
            widget.bind('<Button-1>', lambda e, op=operator: self._show_operator_detail(op))
            widget.bind('<Enter>', lambda e, c=card: c.configure(bg='#f0f9ff'))
            widget.bind('<Leave>', lambda e, c=card: c.configure(bg='white'))
    
    def _filter_operators(self, *args):
        """Filter operators based on search text"""
        self._populate_operators(self.search_var.get())
    
    def _show_list_view(self):
        """Switch back to operator list view"""
        self.current_view = "list"
        self.selected_operator = None
        
        # Update header
        self.back_btn.pack_forget()
        self.title_label.config(text="👥 Operators")
        self.subtitle_label.config(text="Click an operator to view detailed stats")
        self.subtitle_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # Recreate list view
        self._create_list_view()
    
    def _show_operator_detail(self, operator):
        """Show detailed stats for selected operator"""
        self.current_view = "detail"
        self.selected_operator = operator
        
        # Update header
        self.title_label.config(text=f"👤 {operator}")
        self.subtitle_label.config(text="Performance Overview")
        self.back_btn.pack(side=tk.LEFT, padx=(0, 20))
        self.title_label.pack_forget()
        self.subtitle_label.pack_forget()
        self.title_label.pack(side=tk.LEFT)
        self.subtitle_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # Clear content and create detail view
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        self._create_detail_view(operator)
    
    def _create_detail_view(self, operator):
        """Create the operator detail view"""
        # Get operator data (use filtered data)
        df = self.data_manager.get_filtered_df()
        op_data = df[df['Operator Name'] == operator].copy() if df is not None else pd.DataFrame()
        
        if op_data.empty:
            ttk.Label(self.content_frame, text="No data available for this operator.",
                     style='Subtitle.TLabel').pack(pady=50)
            return
        
        # Summary stats card at top
        summary_frame = ttk.LabelFrame(self.content_frame, text="📈 Summary Statistics", padding="15")
        summary_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Calculate summary stats
        total_records = len(op_data)
        total_machines = op_data['Machine Name'].nunique()
        total_duration = op_data['Duration_Seconds'].sum()
        avg_duration = op_data['Duration_Seconds'].mean()
        min_duration = op_data['Duration_Seconds'].min()
        max_duration = op_data['Duration_Seconds'].max()
        
        # Summary grid
        stats_grid = ttk.Frame(summary_frame)
        stats_grid.pack(fill=tk.X)
        
        stats = [
            ("📊 Total Records", f"{total_records:,}"),
            ("🔧 Machines Worked", f"{total_machines}"),
            ("⏱️ Total Time", f"{total_duration/3600:.1f} hrs"),
            ("📉 Avg Duration", f"{avg_duration:.1f}s"),
            ("⬇️ Min Duration", f"{min_duration:.1f}s"),
            ("⬆️ Max Duration", f"{max_duration:.1f}s"),
        ]
        
        for i, (label, value) in enumerate(stats):
            frame = ttk.Frame(stats_grid)
            frame.grid(row=0, column=i, padx=20, pady=5)
            ttk.Label(frame, text=label, style='Subtitle.TLabel').pack()
            ttk.Label(frame, text=value, font=('Segoe UI', 16, 'bold')).pack()
        
        # Main content area with charts and table
        content_paned = ttk.PanedWindow(self.content_frame, orient=tk.HORIZONTAL)
        content_paned.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Machine stats table
        table_frame = ttk.LabelFrame(content_paned, text="📋 Performance by Machine", padding="10")
        content_paned.add(table_frame, weight=1)
        
        # Create treeview for machine stats
        columns = ('Machine', 'Records', 'Avg Duration', 'Min', 'Max', 'Efficiency')
        tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=12)
        
        tree.heading('Machine', text='Machine')
        tree.heading('Records', text='Records')
        tree.heading('Avg Duration', text='Avg (s)')
        tree.heading('Min', text='Min (s)')
        tree.heading('Max', text='Max (s)')
        tree.heading('Efficiency', text='Efficiency')
        
        tree.column('Machine', width=140)
        tree.column('Records', width=70)
        tree.column('Avg Duration', width=80)
        tree.column('Min', width=70)
        tree.column('Max', width=70)
        tree.column('Efficiency', width=80)
        
        tree_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=tree_scroll.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Calculate per-machine stats
        machine_stats = []
        
        for machine in sorted(op_data['Machine Name'].unique()):
            m_data = op_data[op_data['Machine Name'] == machine]
            records = len(m_data)
            avg_dur = m_data['Duration_Seconds'].mean()
            min_dur = m_data['Duration_Seconds'].min()
            max_dur = m_data['Duration_Seconds'].max()
            
            # Get target time from shared data manager
            target_time = self.data_manager.get_target_time(machine)
            
            # Calculate efficiency (target / actual)
            valid_durations = m_data['Duration_Seconds'][m_data['Duration_Seconds'] > 0]
            if len(valid_durations) > 0:
                efficiency = (target_time / valid_durations).mean() * 100
            else:
                efficiency = 0
            
            machine_stats.append({
                'machine': machine,
                'records': records,
                'avg': avg_dur,
                'min': min_dur,
                'max': max_dur,
                'efficiency': efficiency
            })
            
            tree.insert('', 'end', values=(
                machine, records, f"{avg_dur:.1f}", f"{min_dur:.1f}", 
                f"{max_dur:.1f}", f"{efficiency:.1f}%"
            ))
        
        # Right side - Charts
        chart_frame = ttk.LabelFrame(content_paned, text="📊 Visual Analytics", padding="10")
        content_paned.add(chart_frame, weight=1)
        
        # Create matplotlib figure with two subplots
        fig = Figure(figsize=(6, 8), dpi=100)
        
        # Duration chart
        ax1 = fig.add_subplot(211)
        machines = [s['machine'] for s in machine_stats]
        avg_durations = [s['avg'] for s in machine_stats]
        
        # Get target times for each machine
        targets = [self.data_manager.get_target_time(m) for m in machines]
        
        colors = ['#10b981' if d <= t else '#ef4444' for d, t in zip(avg_durations, targets)]
        bars1 = ax1.barh(range(len(machines)), avg_durations, color=colors, alpha=0.8, edgecolor='black')
        
        # Draw individual target markers for each machine
        for i, target in enumerate(targets):
            ax1.plot(target, i, marker='|', markersize=20, markeredgewidth=3, 
                    color='#2563eb', zorder=5)
        
        # Add a legend entry for targets
        ax1.plot([], [], marker='|', markersize=10, markeredgewidth=3, color='#2563eb', 
                linestyle='None', label='Target')
        
        ax1.set_yticks(range(len(machines)))
        ax1.set_yticklabels(machines, fontsize=10)
        ax1.set_xlabel('Avg Duration (seconds)', fontsize=11)
        ax1.set_title(f'Duration by Machine (with per-machine targets)', fontweight='bold', fontsize=12)
        ax1.legend(loc='lower right', fontsize=9)
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels showing duration and target
        for i, (bar, val, target) in enumerate(zip(bars1, avg_durations, targets)):
            label = f'{val:.1f}s (T:{target}s)'
            ax1.text(val + 1, bar.get_y() + bar.get_height()/2, label, 
                    va='center', fontsize=9)
        
        # Efficiency chart
        ax2 = fig.add_subplot(212)
        efficiencies = [s['efficiency'] for s in machine_stats]
        
        colors2 = ['#10b981' if e >= 100 else '#f59e0b' if e >= 80 else '#ef4444' for e in efficiencies]
        bars2 = ax2.barh(range(len(machines)), efficiencies, color=colors2, alpha=0.8, edgecolor='black')
        ax2.axvline(x=100, color='#2563eb', linestyle='--', linewidth=2, label='Target: 100%')
        ax2.set_yticks(range(len(machines)))
        ax2.set_yticklabels(machines, fontsize=10)
        ax2.set_xlabel('Efficiency (%)', fontsize=11)
        ax2.set_title(f'Efficiency by Machine', fontweight='bold', fontsize=12)
        ax2.legend(loc='lower right', fontsize=9)
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars2, efficiencies):
            ax2.text(val + 2, bar.get_y() + bar.get_height()/2, f'{val:.0f}%', 
                    va='center', fontsize=9)
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def refresh_data(self):
        """Refresh UI when data is loaded"""
        if self.current_view == "list":
            self._populate_operators()
        elif self.selected_operator:
            self._show_operator_detail(self.selected_operator)


class CellEfficiencyTab(ttk.Frame):
    """Cell Efficiency Tab - Overall efficiency per manufacturing cell"""
    
    def __init__(self, parent, data_manager):
        super().__init__(parent)
        self.data_manager = data_manager
        self.cell_data = None  # Cell-level aggregated data
        self.station_data = None  # Station-level data
        
        self._setup_ui()
    
    def _setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(header_frame, text="🏭 Cell Efficiency", style='Title.TLabel').pack(side=tk.LEFT)
        ttk.Label(header_frame, text="Overall efficiency metrics per manufacturing cell",
                  style='Subtitle.TLabel').pack(side=tk.LEFT, padx=(20, 0))
        
        # Two-column layout
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Settings and explanation
        left_panel = ttk.Frame(content_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        
        # Explanation card
        explain_frame = ttk.LabelFrame(left_panel, text="📖 How Cell Efficiency is Calculated", padding="15")
        explain_frame.pack(fill=tk.X, pady=(0, 15))
        
        explanation = tk.Text(explain_frame, wrap=tk.WORD, height=14, width=40,
                             font=('Segoe UI', 9), bg='#f8fafc', relief='flat')
        explanation.pack(fill=tk.X)
        
        explain_text = """Cell Efficiency measures how well a manufacturing cell performs against target cycle times.

🏭 Structure:
• Cell = Group of stations (e.g., Cell 221)
• Station = Individual machine (e.g., 221 W01 Press)
• Stations are grouped by their cell # prefix

📊 Formula:
Efficiency = (Target Time ÷ Actual Time) × 100%

💡 Interpretation:
• 100% = Exactly meeting target
• >100% = Faster than target (good)
• <100% = Slower than target

📝 Cell efficiency averages ALL stations within that cell to give overall cell performance."""
        
        explanation.insert('1.0', explain_text)
        explanation.config(state='disabled')
        
        # Target times info
        target_frame = ttk.LabelFrame(left_panel, text="⏱️ Target Times", padding="10")
        target_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(target_frame, text="ℹ️ Target times are configured in the",
                  style='Subtitle.TLabel').pack(anchor=tk.W)
        ttk.Label(target_frame, text="⏱️ Target Times tab",
                  font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W, pady=(2, 5))
        
        self.targets_summary = ttk.Label(target_frame, text="No data loaded",
                                          style='Subtitle.TLabel', wraplength=260)
        self.targets_summary.pack(anchor=tk.W, pady=(5, 0))
        
        # Calculate button
        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.calc_btn = ttk.Button(btn_frame, text="📊 Calculate Cell Efficiency",
                                    command=self._calculate_efficiency, style='Primary.TButton')
        self.calc_btn.pack(fill=tk.X, pady=5)
        
        self.export_btn = ttk.Button(btn_frame, text="💾 Export Results",
                                      command=self._export_results, state=tk.DISABLED)
        self.export_btn.pack(fill=tk.X, pady=5)
        
        # Right panel - Results
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Overall efficiency headline card
        self.overall_frame = ttk.LabelFrame(right_panel, text="🎯 Overall Factory Cell Efficiency", padding="15")
        self.overall_frame.pack(fill=tk.X, pady=(0, 10))
        
        overall_content = ttk.Frame(self.overall_frame)
        overall_content.pack(fill=tk.X)
        
        # Big efficiency number (placeholder)
        self.overall_efficiency_label = ttk.Label(overall_content, text="--.--%", 
                                                   font=('Segoe UI', 36, 'bold'))
        self.overall_efficiency_label.pack(side=tk.LEFT, padx=(20, 30))
        
        # Stats beside it
        self.overall_stats_frame = ttk.Frame(overall_content)
        self.overall_stats_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.overall_desc_label = ttk.Label(self.overall_stats_frame, 
                                            text="Load data and calculate to see overall efficiency",
                                            style='Subtitle.TLabel')
        self.overall_desc_label.pack(anchor=tk.W)
        
        # Results table
        table_frame = ttk.LabelFrame(right_panel, text="📋 Cell & Station Efficiency Results", padding="10")
        table_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Treeview with hierarchy support
        columns = ('Efficiency', 'Avg Duration', 'Records', 'Operators', 'Status')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='tree headings', height=8)
        
        self.tree.heading('#0', text='Cell / Station')
        self.tree.heading('Efficiency', text='Efficiency')
        self.tree.heading('Avg Duration', text='Avg Duration')
        self.tree.heading('Records', text='Records')
        self.tree.heading('Operators', text='Operators')
        self.tree.heading('Status', text='Status')
        
        self.tree.column('#0', width=180)
        self.tree.column('Efficiency', width=90)
        self.tree.column('Avg Duration', width=100)
        self.tree.column('Records', width=70)
        self.tree.column('Operators', width=80)
        self.tree.column('Status', width=100)
        
        tree_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Chart notebook with tabs for Overview and Trends
        chart_notebook = ttk.Notebook(right_panel)
        chart_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Overview tab (bar chart)
        overview_frame = ttk.Frame(chart_notebook, padding="5")
        chart_notebook.add(overview_frame, text="  📊 Overview  ")
        
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=overview_frame)
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Cell Trends tab
        cell_trend_frame = ttk.Frame(chart_notebook, padding="5")
        chart_notebook.add(cell_trend_frame, text="  📈 Cell Trends  ")
        
        self.trend_fig = Figure(figsize=(8, 4), dpi=100)
        self.trend_canvas = FigureCanvasTkAgg(self.trend_fig, master=cell_trend_frame)
        self.trend_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Summary stats frame
        self.summary_frame = ttk.Frame(right_panel)
        self.summary_frame.pack(fill=tk.X, pady=(10, 0))
    
    def refresh_data(self):
        """Refresh UI when data is loaded"""
        # Update summary of target times being used
        if self.data_manager.machines:
            cells = self.data_manager.get_cells_and_stations()
            num_cells = len(cells)
            num_stations = len(self.data_manager.machines)
            num_targets = len(self.data_manager.target_times)
            self.targets_summary.config(
                text=f"{num_cells} cells, {num_stations} stations\n"
                     f"Targets set: {num_targets}/{num_stations}"
            )
        else:
            self.targets_summary.config(text="No data loaded")
    
    def _calculate_efficiency(self):
        """Calculate cell efficiency - aggregating stations into cells"""
        if self.data_manager.df is None:
            messagebox.showwarning("No Data", "Please load a data file first.")
            return
        
        try:
            # Use shared target times from data manager
            target_times = {}
            for station in self.data_manager.machines:
                target_times[station] = self.data_manager.get_target_time(station)
            
            # Use centralized filtered data
            df = self.data_manager.get_filtered_df()
            
            # Get cells and stations
            cells = self.data_manager.get_cells_and_stations()
            
            # Calculate per-station metrics first
            self.station_data = {}
            
            for station in self.data_manager.machines:
                # Use centralized calculation method
                metrics = self.data_manager.get_station_efficiency(station, df)
                if not metrics:
                    continue
                
                cell_num = self.data_manager.extract_cell_number(station)
                station_short = station.replace(cell_num, '').strip() if cell_num else station
                
                if metrics['efficiency'] >= 100:
                    status = "✅ On Target"
                elif metrics['efficiency'] >= 85:
                    status = "⚠️ Near"
                else:
                    status = "❌ Below"
                
                self.station_data[station] = {
                    'cell': cell_num,
                    'station': station,
                    'station_short': station_short,
                    'efficiency': metrics['efficiency'],
                    'avg_duration': metrics['avg_duration'],
                    'records': metrics['records'],
                    'operators': metrics['operators'],
                    'status': status
                }
            
            # Calculate cell-level aggregates
            self.cell_data = []
            
            for cell_num in sorted(cells.keys()):
                cell_stations = [self.station_data[s] for s in cells[cell_num] if s in self.station_data]
                
                if not cell_stations:
                    continue
                
                # Average efficiency across all stations in the cell
                cell_efficiency = sum(s['efficiency'] for s in cell_stations) / len(cell_stations)
                cell_avg_duration = sum(s['avg_duration'] for s in cell_stations) / len(cell_stations)
                cell_records = sum(s['records'] for s in cell_stations)
                cell_operators = len(set(op for s in cell_stations for op in 
                                        df[df['Machine Name'] == s['station']]['Operator Name'].unique()))
                
                if cell_efficiency >= 100:
                    status = "✅ On Target"
                elif cell_efficiency >= 85:
                    status = "⚠️ Near Target"
                else:
                    status = "❌ Below Target"
                
                self.cell_data.append({
                    'cell': cell_num,
                    'efficiency': cell_efficiency,
                    'avg_duration': cell_avg_duration,
                    'records': cell_records,
                    'operators': cell_operators,
                    'stations': cell_stations,
                    'status': status
                })
            
            # Sort cells by efficiency (descending)
            self.cell_data.sort(key=lambda x: x['efficiency'], reverse=True)
            
            # Update treeview with hierarchical data
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            for cell in self.cell_data:
                # Insert cell row
                cell_id = self.tree.insert('', 'end', text=f"🏭 Cell {cell['cell']}", 
                    values=(
                        f"{cell['efficiency']:.1f}%",
                        f"{cell['avg_duration']:.1f}s",
                        cell['records'],
                        cell['operators'],
                        cell['status']
                    ), open=True)
                
                # Insert station rows under the cell
                for station in sorted(cell['stations'], key=lambda x: x['efficiency'], reverse=True):
                    self.tree.insert(cell_id, 'end', text=f"  ├─ {station['station_short']}",
                        values=(
                            f"{station['efficiency']:.1f}%",
                            f"{station['avg_duration']:.1f}s",
                            station['records'],
                            station['operators'],
                            station['status']
                        ))
            
            # Update chart
            self._update_chart()
            
            # Update summary
            self._update_summary()
            
            self.export_btn.config(state=tk.NORMAL)
            
            num_stations = sum(len(c['stations']) for c in self.cell_data)
            messagebox.showinfo("Complete", 
                f"Calculated efficiency for {len(self.cell_data)} cells ({num_stations} stations).")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", str(e))
    
    def _update_chart(self):
        """Update the efficiency chart - showing cell-level data"""
        if not self.cell_data:
            return
        
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        cells = [f"Cell {c['cell']}" for c in self.cell_data]
        efficiencies = [c['efficiency'] for c in self.cell_data]
        
        # Color based on efficiency
        colors = []
        for e in efficiencies:
            if e >= 100:
                colors.append('#10b981')  # Green
            elif e >= 85:
                colors.append('#f59e0b')  # Yellow/Orange
            else:
                colors.append('#ef4444')  # Red
        
        # Create bar chart
        bars = ax.bar(range(len(cells)), efficiencies, color=colors, alpha=0.8, edgecolor='black')
        
        # Add 100% target line
        ax.axhline(y=100, color='#2563eb', linestyle='--', linewidth=2, label='Target (100%)')
        
        # Add value labels
        for bar, val in zip(bars, efficiencies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{val:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels(cells, rotation=45, ha='right', fontsize=9)
        ax.set_xlabel('Cell', fontweight='bold')
        ax.set_ylabel('Efficiency (%)', fontweight='bold')
        ax.set_title('Overall Cell Efficiency', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        # Set y-axis to start at 0
        ax.set_ylim(bottom=0, top=max(efficiencies) * 1.15 if efficiencies else 120)
        
        self.fig.tight_layout()
        self.chart_canvas.draw()
        
        # Update trend charts
        self._update_cell_trend_chart()
    
    def _update_cell_trend_chart(self):
        """Update the cell efficiency trend chart over time"""
        self.trend_fig.clear()
        ax = self.trend_fig.add_subplot(111)
        
        trends = self.data_manager.get_trend_data(group_by='cell')
        
        if not trends:
            ax.text(0.5, 0.5, 'No date data available for trends.\nEnsure your data has a Date column.',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title('Cell Efficiency Trends Over Time', fontsize=14, fontweight='bold')
            self.trend_fig.tight_layout()
            self.trend_canvas.draw()
            return
        
        # Color palette for different cells
        colors = ['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16']
        
        for i, (cell_num, data) in enumerate(sorted(trends.items())):
            if len(data['dates']) > 0:
                color = colors[i % len(colors)]
                dates = data['dates']
                effs = data['efficiencies']
                
                # Plot line with markers
                ax.plot(dates, effs, marker='o', markersize=6, linewidth=2, 
                       label=f'Cell {cell_num}', color=color, alpha=0.8)
                
                # Add trend line (linear regression)
                if len(dates) >= 2:
                    x_numeric = np.arange(len(dates))
                    z = np.polyfit(x_numeric, effs, 1)
                    p = np.poly1d(z)
                    ax.plot(dates, p(x_numeric), linestyle='--', color=color, alpha=0.5, linewidth=1)
        
        # Add 100% target line
        ax.axhline(y=100, color='#64748b', linestyle=':', linewidth=2, label='Target (100%)')
        
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_ylabel('Efficiency (%)', fontweight='bold')
        ax.set_title('Cell Efficiency Trends Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(axis='both', alpha=0.3)
        
        # Rotate x-axis labels
        self.trend_fig.autofmt_xdate(rotation=45)
        
        self.trend_fig.tight_layout()
        self.trend_canvas.draw()
    
    def _update_summary(self):
        """Update summary statistics"""
        # Clear previous summary
        for widget in self.summary_frame.winfo_children():
            widget.destroy()
        
        if not self.cell_data:
            self.overall_efficiency_label.config(text="--.--%")
            self.overall_desc_label.config(text="Load data and calculate to see overall efficiency")
            # Clear additional stats
            for widget in self.overall_stats_frame.winfo_children():
                if widget != self.overall_desc_label:
                    widget.destroy()
            return
        
        # Calculate overall stats
        efficiencies = [c['efficiency'] for c in self.cell_data]
        avg_efficiency = sum(efficiencies) / len(efficiencies)
        best_cell = max(self.cell_data, key=lambda x: x['efficiency'])
        worst_cell = min(self.cell_data, key=lambda x: x['efficiency'])
        on_target = sum(1 for c in self.cell_data if c['efficiency'] >= 100)
        total_records = sum(c['records'] for c in self.cell_data)
        
        # Update the headline overall efficiency
        self.overall_efficiency_label.config(text=f"{avg_efficiency:.1f}%")
        
        # Update the stats beside the big number
        for widget in self.overall_stats_frame.winfo_children():
            widget.destroy()
        
        # Status indicator
        if avg_efficiency >= 100:
            status_text = "✅ Factory is ON TARGET"
            status_color = "#10b981"
        elif avg_efficiency >= 85:
            status_text = "⚠️ Factory is NEAR TARGET"
            status_color = "#f59e0b"
        else:
            status_text = "❌ Factory is BELOW TARGET"
            status_color = "#ef4444"
        
        num_stations = sum(len(c['stations']) for c in self.cell_data)
        
        ttk.Label(self.overall_stats_frame, text=status_text, 
                  font=('Segoe UI', 12, 'bold')).pack(anchor=tk.W)
        ttk.Label(self.overall_stats_frame, 
                  text=f"Average across {len(self.cell_data)} cells ({num_stations} stations) • {total_records:,} total records",
                  style='Subtitle.TLabel').pack(anchor=tk.W, pady=(2, 0))
        ttk.Label(self.overall_stats_frame,
                  text=f"🏆 Best: Cell {best_cell['cell']} ({best_cell['efficiency']:.0f}%)  •  "
                       f"📉 Lowest: Cell {worst_cell['cell']} ({worst_cell['efficiency']:.0f}%)",
                  style='Subtitle.TLabel').pack(anchor=tk.W, pady=(2, 0))
        
        # Summary stats row at bottom
        stats = [
            ("✅ On Target", f"{on_target} / {len(self.cell_data)} cells"),
            ("🏆 Best Cell", f"Cell {best_cell['cell']}"),
            ("📉 Needs Improvement", f"Cell {worst_cell['cell']}"),
        ]
        
        for i, (label, value) in enumerate(stats):
            frame = ttk.Frame(self.summary_frame)
            frame.pack(side=tk.LEFT, padx=20)
            ttk.Label(frame, text=label, style='Subtitle.TLabel').pack()
            lbl = ttk.Label(frame, text=value, font=('Segoe UI', 11, 'bold'))
            lbl.pack()
    
    def _export_results(self):
        """Export cell efficiency results with cell and station breakdown"""
        if not self.cell_data:
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")]
        )
        
        if file_path:
            export_data = []
            
            for cell in self.cell_data:
                # Add cell-level row
                export_data.append({
                    'Type': 'Cell',
                    'Cell': cell['cell'],
                    'Station': f"Cell {cell['cell']} (Total)",
                    'Efficiency (%)': round(cell['efficiency'], 1),
                    'Avg Duration (s)': round(cell['avg_duration'], 1),
                    'Total Records': cell['records'],
                    'Operators': cell['operators'],
                    'Status': cell['status']
                })
                
                # Add station-level rows
                for station in cell['stations']:
                    export_data.append({
                        'Type': 'Station',
                        'Cell': cell['cell'],
                        'Station': station['station_short'],
                        'Efficiency (%)': round(station['efficiency'], 1),
                        'Avg Duration (s)': round(station['avg_duration'], 1),
                        'Total Records': station['records'],
                        'Operators': station['operators'],
                        'Status': station['status']
                    })
            
            df = pd.DataFrame(export_data)
            if file_path.endswith('.csv'):
                df.to_csv(file_path, index=False)
            else:
                df.to_excel(file_path, index=False)
            
            messagebox.showinfo("Success", f"Results exported to {file_path}")


class AnalyticsTab(ttk.Frame):
    """Advanced Analytics Tab - Shift analysis, productivity, outliers"""
    
    def __init__(self, parent, data_manager):
        super().__init__(parent)
        self.data_manager = data_manager
        self._setup_ui()
    
    def _setup_ui(self):
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(header_frame, text="📈 Advanced Analytics", style='Title.TLabel').pack(side=tk.LEFT)
        ttk.Label(header_frame, text="Manufacturing insights, productivity metrics, and outlier detection",
                  style='Subtitle.TLabel').pack(side=tk.LEFT, padx=(20, 0))
        
        # Notebook for different analytics views
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Productivity Metrics Tab
        self._setup_productivity_tab()
        
        # Outlier Detection Tab
        self._setup_outliers_tab()
        
        # Operator Rankings Tab
        self._setup_rankings_tab()
    
    def _setup_productivity_tab(self):
        """Setup productivity metrics tab"""
        prod_frame = ttk.Frame(self.notebook, padding="15")
        self.notebook.add(prod_frame, text="  📦 Productivity  ")
        
        # Top section - Last Station Configuration
        config_frame = ttk.LabelFrame(prod_frame, text="🏭 Last Station per Cell (for quantity calculation)", padding="10")
        config_frame.pack(fill=tk.X, pady=(0, 15))
        
        config_info = ttk.Label(config_frame, 
            text="Define the last station in each cell. Quantity/parts per hour is calculated using only records from the last station.",
            style='Subtitle.TLabel', wraplength=700)
        config_info.pack(fill=tk.X, pady=(0, 10))
        
        # Cell and station selection
        select_row = ttk.Frame(config_frame)
        select_row.pack(fill=tk.X, pady=5)
        
        ttk.Label(select_row, text="Cell:", width=8).pack(side=tk.LEFT)
        self.last_station_cell_var = tk.StringVar(value="")
        self.last_station_cell_combo = ttk.Combobox(select_row, textvariable=self.last_station_cell_var,
                                                     values=[], state="readonly", width=15)
        self.last_station_cell_combo.pack(side=tk.LEFT, padx=5)
        self.last_station_cell_combo.bind('<<ComboboxSelected>>', self._on_last_station_cell_select)
        
        ttk.Label(select_row, text="Last Station:", width=12).pack(side=tk.LEFT, padx=(15, 0))
        self.last_station_var = tk.StringVar(value="")
        self.last_station_combo = ttk.Combobox(select_row, textvariable=self.last_station_var,
                                                values=[], state="readonly", width=30)
        self.last_station_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(select_row, text="Set Last Station", 
                   command=self._set_last_station).pack(side=tk.LEFT, padx=10)
        ttk.Button(select_row, text="Clear", 
                   command=self._clear_last_station).pack(side=tk.LEFT, padx=2)
        
        # Display current settings
        self.last_station_display = tk.Text(config_frame, height=3, width=80,
                                             font=('Consolas', 10), bg='#f1f5f9', state='disabled')
        self.last_station_display.pack(fill=tk.X, pady=(10, 0))
        
        # Controls
        ctrl_frame = ttk.Frame(prod_frame)
        ctrl_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Button(ctrl_frame, text="📊 Calculate Metrics", 
                   command=self._calculate_productivity, style='Primary.TButton').pack(side=tk.LEFT, padx=5)
        
        # Big numbers dashboard
        self.metrics_frame = ttk.Frame(prod_frame)
        self.metrics_frame.pack(fill=tk.BOTH, expand=True)
        
        # Placeholder
        ttk.Label(self.metrics_frame, text="Click 'Calculate Metrics' to view productivity data",
                  style='Subtitle.TLabel').pack(pady=50)
    
    def _setup_outliers_tab(self):
        """Setup outlier detection tab"""
        outlier_frame = ttk.Frame(self.notebook, padding="15")
        self.notebook.add(outlier_frame, text="  ⚠️ Outliers  ")
        
        # Controls
        ctrl_frame = ttk.Frame(outlier_frame)
        ctrl_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(ctrl_frame, text="Z-Score Threshold:").pack(side=tk.LEFT, padx=5)
        self.zscore_var = tk.StringVar(value="2.0")
        ttk.Entry(ctrl_frame, textvariable=self.zscore_var, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(ctrl_frame, text="(Higher = fewer outliers)", style='Subtitle.TLabel').pack(side=tk.LEFT, padx=5)
        
        ttk.Button(ctrl_frame, text="🔍 Detect Outliers", 
                   command=self._detect_outliers, style='Primary.TButton').pack(side=tk.LEFT, padx=15)
        
        # Results table
        columns = ('Timestamp', 'Operator', 'Machine', 'Duration', 'Z-Score', 'Type')
        self.outlier_tree = ttk.Treeview(outlier_frame, columns=columns, show='headings', height=15)
        
        self.outlier_tree.heading('Timestamp', text='Timestamp')
        self.outlier_tree.heading('Operator', text='Operator')
        self.outlier_tree.heading('Machine', text='Machine')
        self.outlier_tree.heading('Duration', text='Duration (s)')
        self.outlier_tree.heading('Z-Score', text='Z-Score')
        self.outlier_tree.heading('Type', text='Type')
        
        self.outlier_tree.column('Timestamp', width=150)
        self.outlier_tree.column('Operator', width=120)
        self.outlier_tree.column('Machine', width=150)
        self.outlier_tree.column('Duration', width=100)
        self.outlier_tree.column('Z-Score', width=80)
        self.outlier_tree.column('Type', width=100)
        
        scrollbar = ttk.Scrollbar(outlier_frame, orient="vertical", command=self.outlier_tree.yview)
        self.outlier_tree.configure(yscrollcommand=scrollbar.set)
        
        self.outlier_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _setup_rankings_tab(self):
        """Setup operator rankings tab"""
        rank_frame = ttk.Frame(self.notebook, padding="15")
        self.notebook.add(rank_frame, text="  🏆 Rankings  ")
        
        # Controls
        ctrl_frame = ttk.Frame(rank_frame)
        ctrl_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(ctrl_frame, text="Sort By:").pack(side=tk.LEFT, padx=5)
        self.sort_var = tk.StringVar(value="efficiency")
        sort_combo = ttk.Combobox(ctrl_frame, textvariable=self.sort_var,
                                   values=["efficiency", "records", "quantity", "consistency"],
                                   state="readonly", width=15)
        sort_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(ctrl_frame, text="📊 Update Rankings", 
                   command=self._update_rankings, style='Primary.TButton').pack(side=tk.LEFT, padx=15)
        
        # Rankings table
        columns = ('Rank', 'Operator', 'Efficiency', 'Records', 'Consistency', 'Quantity')
        self.rank_tree = ttk.Treeview(rank_frame, columns=columns, show='headings', height=15)
        
        self.rank_tree.heading('Rank', text='#')
        self.rank_tree.heading('Operator', text='Operator')
        self.rank_tree.heading('Efficiency', text='Efficiency')
        self.rank_tree.heading('Records', text='Records')
        self.rank_tree.heading('Consistency', text='Consistency')
        self.rank_tree.heading('Quantity', text='Quantity')
        
        self.rank_tree.column('Rank', width=50)
        self.rank_tree.column('Operator', width=150)
        self.rank_tree.column('Efficiency', width=100)
        self.rank_tree.column('Records', width=80)
        self.rank_tree.column('Consistency', width=100)
        self.rank_tree.column('Quantity', width=80)
        
        scrollbar = ttk.Scrollbar(rank_frame, orient="vertical", command=self.rank_tree.yview)
        self.rank_tree.configure(yscrollcommand=scrollbar.set)
        
        self.rank_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _on_last_station_cell_select(self, event=None):
        """When a cell is selected, populate stations for that cell"""
        cell = self.last_station_cell_var.get()
        if cell:
            cells_dict = self.data_manager.get_cells_and_stations()
            if cell in cells_dict:
                stations = sorted(cells_dict[cell])
                self.last_station_combo['values'] = stations
                # Show current setting if exists
                if cell in self.data_manager.last_stations:
                    self.last_station_var.set(self.data_manager.last_stations[cell])
                elif stations:
                    self.last_station_var.set(stations[0])
    
    def _set_last_station(self):
        """Set the last station for the selected cell"""
        cell = self.last_station_cell_var.get()
        station = self.last_station_var.get()
        
        if not cell:
            messagebox.showwarning("No Cell", "Please select a cell first.")
            return
        if not station:
            messagebox.showwarning("No Station", "Please select a station.")
            return
        
        self.data_manager.last_stations[cell] = station
        self.data_manager.save_settings()
        self._update_last_station_display()
        messagebox.showinfo("Saved", f"Last station for Cell {cell} set to:\n{station}")
    
    def _clear_last_station(self):
        """Clear the last station setting for the selected cell"""
        cell = self.last_station_cell_var.get()
        if not cell:
            messagebox.showwarning("No Cell", "Please select a cell first.")
            return
        
        if cell in self.data_manager.last_stations:
            del self.data_manager.last_stations[cell]
            self.data_manager.save_settings()
        
        self._update_last_station_display()
        messagebox.showinfo("Cleared", f"Last station setting cleared for Cell {cell}.")
    
    def _update_last_station_display(self):
        """Update the display showing current last station settings"""
        self.last_station_display.config(state='normal')
        self.last_station_display.delete('1.0', tk.END)
        
        if not self.data_manager.last_stations:
            self.last_station_display.insert('1.0', "No last stations defined. All stations will be used for quantity calculation.")
        else:
            lines = []
            for cell, station in sorted(self.data_manager.last_stations.items()):
                lines.append(f"Cell {cell}: {station}")
            self.last_station_display.insert('1.0', "\n".join(lines))
        
        self.last_station_display.config(state='disabled')
    
    def _populate_last_station_combos(self):
        """Populate the cell combobox for last station selection"""
        cells_dict = self.data_manager.get_cells_and_stations()
        cells = sorted(cells_dict.keys())
        self.last_station_cell_combo['values'] = cells
        if cells:
            self.last_station_cell_var.set(cells[0])
            self._on_last_station_cell_select()
        self._update_last_station_display()
    
    def _calculate_productivity(self):
        """Calculate and display productivity metrics"""
        if self.data_manager.df is None:
            messagebox.showwarning("No Data", "Please load a data file first.")
            return
        
        metrics = self.data_manager.get_productivity_metrics()
        
        if not metrics:
            return
        
        # Clear existing widgets
        for widget in self.metrics_frame.winfo_children():
            widget.destroy()
        
        # Create metrics dashboard
        grid = ttk.Frame(self.metrics_frame)
        grid.pack(expand=True, pady=20)
        
        # Row 1 - Main metrics
        metrics_data = [
            ("📁 Total Records", f"{metrics['total_records']:,}", "#3b82f6"),
            ("👥 Operators", f"{metrics['total_operators']}", "#8b5cf6"),
            ("🔧 Machines", f"{metrics['total_machines']}", "#06b6d4"),
        ]
        
        for i, (label, value, color) in enumerate(metrics_data):
            frame = ttk.Frame(grid)
            frame.grid(row=0, column=i, padx=30, pady=15)
            ttk.Label(frame, text=label, font=('Segoe UI', 11)).pack()
            ttk.Label(frame, text=value, font=('Segoe UI', 36, 'bold'), foreground=color).pack()
        
        # Row 2 - Duration metrics
        duration_data = [
            ("⏱️ Avg Duration", f"{metrics['avg_duration']:.1f}s", "#f59e0b"),
            ("📊 Median Duration", f"{metrics['median_duration']:.1f}s", "#10b981"),
            ("📈 Std Dev", f"{metrics['std_duration']:.1f}s", "#ef4444"),
        ]
        
        for i, (label, value, color) in enumerate(duration_data):
            frame = ttk.Frame(grid)
            frame.grid(row=1, column=i, padx=30, pady=15)
            ttk.Label(frame, text=label, font=('Segoe UI', 11)).pack()
            ttk.Label(frame, text=value, font=('Segoe UI', 28, 'bold'), foreground=color).pack()
        
        # Row 3 - Quantity metrics (if available)
        if 'total_quantity' in metrics:
            qty_data = [
                ("📦 Total Quantity", f"{metrics['total_quantity']:,}", "#22c55e"),
                ("⚡ Parts/Hour", f"{metrics['parts_per_hour']:.1f}", "#f97316"),
            ]
            
            for i, (label, value, color) in enumerate(qty_data):
                frame = ttk.Frame(grid)
                frame.grid(row=2, column=i, padx=30, pady=15)
                ttk.Label(frame, text=label, font=('Segoe UI', 11)).pack()
                ttk.Label(frame, text=value, font=('Segoe UI', 28, 'bold'), foreground=color).pack()
    
    def _detect_outliers(self):
        """Detect statistical outliers"""
        if self.data_manager.df is None:
            messagebox.showwarning("No Data", "Please load a data file first.")
            return
        
        try:
            threshold = float(self.zscore_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid threshold value")
            return
        
        outliers = self.data_manager.get_outliers(threshold=threshold)
        
        if outliers is None or len(outliers) == 0:
            messagebox.showinfo("No Outliers", f"No outliers found with z-score > {threshold}")
            return
        
        # Clear and populate table
        for item in self.outlier_tree.get_children():
            self.outlier_tree.delete(item)
        
        for _, row in outliers.iterrows():
            # Get timestamp if available
            timestamp = ""
            if 'ParsedDate' in row.index:
                timestamp = str(row['ParsedDate'])[:19] if pd.notna(row['ParsedDate']) else ""
            elif 'Date' in row.index:
                timestamp = str(row['Date'])[:19] if pd.notna(row['Date']) else ""
            
            self.outlier_tree.insert('', 'end', values=(
                timestamp,
                row['Operator Name'],
                row['Machine Name'],
                f"{row['Duration_Seconds']:.1f}",
                f"{row['z_score']:.2f}",
                row['outlier_type']
            ))
        
        messagebox.showinfo("Outliers Found", 
                           f"Found {len(outliers)} outliers with z-score > {threshold}")
    
    def _update_rankings(self):
        """Update operator rankings"""
        if self.data_manager.df is None:
            messagebox.showwarning("No Data", "Please load a data file first.")
            return
        
        sort_by = self.sort_var.get()
        rankings = self.data_manager.get_operator_rankings(sort_by=sort_by)
        
        if not rankings:
            return
        
        # Clear and populate table
        for item in self.rank_tree.get_children():
            self.rank_tree.delete(item)
        
        for i, r in enumerate(rankings, 1):
            # Add medal emoji for top 3
            rank_display = {1: "🥇 1", 2: "🥈 2", 3: "🥉 3"}.get(i, str(i))
            
            self.rank_tree.insert('', 'end', values=(
                rank_display,
                r['operator'],
                f"{r['efficiency']:.1f}%",
                f"{r['records']:,}",
                f"{r['consistency']:.1f}%",
                f"{r['quantity']:,}" if r['quantity'] > 0 else "N/A"
            ))
    
    def refresh_data(self):
        """Refresh when data is loaded"""
        # Populate last station configuration
        if hasattr(self, 'last_station_cell_combo'):
            self._populate_last_station_combos()


class FactoryAnalyticsDashboard:
    """Main Dashboard Application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Factory Analytics Dashboard")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Configure modern style
        ModernStyle.configure_styles()
        
        # Shared data manager
        self.data_manager = DataManager()
        
        self._setup_ui()
        
        # Auto-load default file if exists
        self._try_load_default()
    
    def _setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="0")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header bar
        header = ttk.Frame(main_frame, padding="15")
        header.pack(fill=tk.X)
        
        ttk.Label(header, text="🏭 Factory Analytics Dashboard", 
                  style='Title.TLabel').pack(side=tk.LEFT)
        
        # File controls
        file_frame = ttk.Frame(header)
        file_frame.pack(side=tk.RIGHT)
        
        self.file_label = ttk.Label(file_frame, text="No file loaded", 
                                     style='Subtitle.TLabel')
        self.file_label.pack(side=tk.LEFT, padx=(0, 15))
        
        ttk.Button(file_frame, text="📂 Load File", command=self._browse_file,
                   style='Modern.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="📊 Use Sample Data", command=self._load_sample,
                   style='Modern.TButton').pack(side=tk.LEFT, padx=5)
        
        # Separator
        ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        
        # Notebook (tabs)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.duration_tab = DurationAnalysisTab(self.notebook, self.data_manager)
        self.efficiency_tab = EfficiencyCalculatorTab(self.notebook, self.data_manager)
        self.operators_tab = OperatorsTab(self.notebook, self.data_manager)
        self.cell_efficiency_tab = CellEfficiencyTab(self.notebook, self.data_manager)
        self.target_times_tab = TargetTimesTab(self.notebook, self.data_manager)
        self.data_filters_tab = DataFiltersTab(self.notebook, self.data_manager, 
                                                on_filter_change=self._on_filter_change)
        self.analytics_tab = AnalyticsTab(self.notebook, self.data_manager)
        
        self.notebook.add(self.target_times_tab, text="  ⏱️ Target Times  ")
        self.notebook.add(self.data_filters_tab, text="  🔍 Data Filters  ")
        self.notebook.add(self.duration_tab, text="  📊 Duration Analysis  ")
        self.notebook.add(self.efficiency_tab, text="  ⚡ Efficiency Calculator  ")
        self.notebook.add(self.operators_tab, text="  👥 Operators  ")
        self.notebook.add(self.cell_efficiency_tab, text="  🏭 Cell Efficiency  ")
        self.notebook.add(self.analytics_tab, text="  📈 Analytics  ")
        
        # Status bar
        self.status_bar = ttk.Label(main_frame, text="Ready - Load a data file to begin", 
                                     style='Subtitle.TLabel', padding="10")
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
    
    def _browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self._load_file(file_path)
    
    def _load_sample(self):
        """Try to load sample data files"""
        sample_files = [
            "9.2-9.6 Operator Data 221 .xlsx",
            "Operator Duration per Station/9.2-9.6 Operator Data 221  1.xlsx",
            "Operator Efficiency Calculator/9.2-9.6 Operator Data 221 .xlsx"
        ]
        
        for sample in sample_files:
            if os.path.exists(sample):
                self._load_file(sample)
                return
        
        messagebox.showinfo("No Sample Data", "No sample data files found in the current directory.")
    
    def _try_load_default(self):
        """Try to auto-load a default file"""
        default_files = [
            "9.2-9.6 Operator Data 221 .xlsx",
            "Operator Duration per Station/9.2-9.6 Operator Data 221  1.xlsx"
        ]
        
        for default in default_files:
            if os.path.exists(default):
                self._load_file(default)
                return
    
    def _load_file(self, file_path: str):
        try:
            self.data_manager.load_data(file_path)
            
            # Update UI
            self.file_label.config(text=f"📁 {os.path.basename(file_path)}")
            self.status_bar.config(text=f"Loaded {len(self.data_manager.df)} records | "
                                        f"{len(self.data_manager.machines)} machines | "
                                        f"{len(self.data_manager.operators)} operators")
            
            # Refresh tabs - target times and filters first so other tabs can use the values
            self.target_times_tab.refresh_data()
            self.data_filters_tab.refresh_data()
            self.duration_tab.refresh_data()
            self.efficiency_tab.refresh_data()
            self.operators_tab.refresh_data()
            self.cell_efficiency_tab.refresh_data()
            self.analytics_tab.refresh_data()
            
            messagebox.showinfo("Success", f"Loaded {len(self.data_manager.machines)} machines successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def _on_filter_change(self):
        """Callback when filters are changed - refresh analysis tabs"""
        # Update status bar with filter info
        stats = self.data_manager.get_filter_stats()
        if stats:
            self.status_bar.config(
                text=f"Loaded {stats['total']} records | "
                     f"Filtered: {stats['filtered']} kept, {stats['removed']} removed | "
                     f"{len(self.data_manager.machines)} machines"
            )


def main():
    root = tk.Tk()
    app = FactoryAnalyticsDashboard(root)
    root.mainloop()


if __name__ == "__main__":
    main()
