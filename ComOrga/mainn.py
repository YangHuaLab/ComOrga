import os
import sys
import traceback
import pandas as pd
import numpy as np
import shutil

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout,
                             QVBoxLayout, QPushButton, QFileDialog, QListWidget, QListWidgetItem,
                             QLabel, QGroupBox, QFormLayout, QLineEdit, QComboBox,
                             QFrame, QSplitter, QStackedWidget, QTabWidget, QAbstractItemView,
                             QTextEdit, QScrollArea, QSpinBox, QDoubleSpinBox, QMessageBox,
                             QTableWidget, QTableWidgetItem, QHeaderView, QColorDialog, QCheckBox,
                             QDialog, QDialogButtonBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns

from concattif import run_concat_tif
from multiworker_cli import run_cellprofiler_by_cli
from extract_params import extract_target_parameters
from tsne_merged import run_merged_tsne_by_r
from pca_analysis import run_pca_comprehensive


# ==========================================
# Data Aggregator (For Step 4 & 5)
# ==========================================
def do_aggregation(input_dir, out_dir, group_config, is_rows_are_samples, selected_feats, log):
    log(">>> 🔄 Starting Data Aggregation across multiple CSVs...")
    df_list = []
    adjusted_config = []

    for g in group_config:
        prefix = g['prefix'].strip()
        if not prefix:
            continue

        target_file = None
        for f in os.listdir(input_dir):
            if f.lower().endswith('.csv') and f.startswith(prefix + "_"):
                target_file = os.path.join(input_dir, f)
                break

        if not target_file:
            log(f"    ⚠️ Warning: No CSV found starting with '{prefix}_'. Skipping group '{g['name']}'.")
            continue

        log(f"    📦 Reading {os.path.basename(target_file)} for Group '{g['name']}'...")
        try:
            df = pd.read_csv(target_file, encoding='utf-8-sig')
            actual_count = 0

            if is_rows_are_samples:
                if selected_feats:
                    valid_cols = [c for c in selected_feats if c in df.columns]
                else:
                    valid_cols = df.columns
                df = df[valid_cols]

                if g['count'] > 0:
                    df = df.head(g['count'])
                actual_count = df.shape[0]
            else:
                if selected_feats:
                    df = df[df.iloc[:, 0].astype(str).isin(selected_feats)]
                if g['count'] > 0:
                    df = df.iloc[:, :g['count'] + 1]
                actual_count = df.shape[1] - 1

            if actual_count > 0:
                df_list.append(df)
                adjusted_config.append({
                    'name': g['name'],
                    'count': actual_count,
                    'color': g['color']
                })
                log(f"      -> Extracted {actual_count} sample(s).")
            else:
                log(f"      -> ⚠️ No valid data extracted.")
        except Exception as e:
            log(f"    ❌ Error processing {os.path.basename(target_file)}: {str(e)}")

    if not df_list:
        log("❌ Aggregation failed: No data extracted from any mapped groups.")
        return None, None

    log(">>> 🧬 Merging aggregated datasets into a Master Table...")
    try:
        if is_rows_are_samples:
            master_df = pd.concat(df_list, axis=0, ignore_index=True)
        else:
            df_list_indexed = [d.set_index(d.columns[0]) for d in df_list]
            master_df = pd.concat(df_list_indexed, axis=1).reset_index()
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        temp_csv_path = os.path.join(out_dir, "temp_aggregated_master.csv")
        master_df.to_csv(temp_csv_path, index=False, encoding='utf-8-sig')
        log(f">>> ✨ Aggregation complete! Master dataset created.")
        return temp_csv_path, adjusted_config
    except Exception as e:
        log(f"❌ Error merging datasets: {str(e)}")
        return None, None


class WorkerThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, task_func, *args, **kwargs):
        super().__init__()
        self.task_func = task_func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        self.kwargs['status_callback'] = self.log_signal.emit
        try:
            result = self.task_func(*self.args, **self.kwargs)
            if isinstance(result, tuple):
                success, msg = result[0], result[1]
            else:
                success = result
                msg = "Execution Completed Successfully!" if success else "Task failed or was aborted."
            self.finished_signal.emit(success, msg)
        except Exception as e:
            err_msg = traceback.format_exc()
            self.finished_signal.emit(False, f"Critical exception:\n{err_msg}")


class EnvSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Global Environment Settings")
        self.resize(650, 220)
        layout = QFormLayout(self)

        self.cp_path = QLineEdit(parent.env_cp_path)
        self.pipeline_path = QLineEdit(parent.env_pipeline_path)
        self.rscript_path = QLineEdit(parent.env_rscript_path)
        self.r_file_path = QLineEdit(parent.env_r_file)

        layout.addRow("CellProfiler.exe Path:", self.create_path_picker(self.cp_path, "CellProfiler.exe"))
        layout.addRow("CP Pipeline (.cppipe):", self.create_path_picker(self.pipeline_path, "pipeline_3D-1115.cppipe"))
        layout.addRow("Rscript.exe Path:", self.create_path_picker(self.rscript_path, "Rscript.exe"))
        layout.addRow("tsne_merged.R Path:", self.create_path_picker(self.r_file_path, "tsne_merged.R"))

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addRow(btn_box)

    def create_path_picker(self, line_edit, target_name):
        box = QHBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(line_edit)

        btn_file = QPushButton("📄")
        btn_file.setFixedWidth(30)
        btn_file.clicked.connect(lambda: self.browse_file(line_edit))
        box.addWidget(btn_file)

        btn_scan = QPushButton("🔍")
        btn_scan.setFixedWidth(30)
        btn_scan.clicked.connect(lambda: self.scan_folder(line_edit, target_name))
        box.addWidget(btn_scan)

        w = QWidget()
        w.setLayout(box)
        return w

    def browse_file(self, line_edit):
        path, _ = QFileDialog.getOpenFileName(self, "Select Executable/File", "", "All Files (*)")
        if path: line_edit.setText(path)

    def scan_folder(self, line_edit, target_name):
        folder = QFileDialog.getExistingDirectory(self, f"Select folder to auto-scan for '{target_name}'")
        if folder:
            found_path = None
            for root, dirs, files in os.walk(folder):
                if target_name in files:
                    found_path = os.path.join(root, target_name)
                    break

            if found_path:
                line_edit.setText(os.path.abspath(found_path))
                QMessageBox.information(self, "Found", f"Successfully found:\n{found_path}")
            else:
                QMessageBox.warning(self, "Not Found", f"Could not find '{target_name}'.")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-functional Omics Data Processing System V7.0 (Advanced Extraction)")
        self.resize(1480, 880)

        self.current_step = 0
        self.raw_folder_path = ""
        self.worker = None

        self.env_cp_path = "cellprofiler"
        self.env_pipeline_path = "pipeline_3D-1115.cppipe"
        self.env_rscript_path = "Rscript"
        self.env_r_file = "tsne_merged.R"

        self.s3_all_params = []  # Used for advanced Step 3 feature filtering

        self.GROUP_CONFIG = [
            {'prefix': '', 'name': 'C', 'count': 2000, 'color': '#0000FF'},
            {'prefix': '', 'name': 'NC', 'count': 2000, 'color': '#8B00FF'},
            {'prefix': '', 'name': 'SIRNA', 'count': 2000, 'color': '#00A6FF'},
            {'prefix': '', 'name': 'M1', 'count': 2000, 'color': '#FF0000'},
            {'prefix': '', 'name': 'M1+NC', 'count': 2000, 'color': '#FF7F00'},
            {'prefix': '', 'name': 'M1+SIRNA', 'count': 2000, 'color': '#ffc0cb'},
            {'prefix': '', 'name': 'M2', 'count': 2000, 'color': '#999999'},
            {'prefix': '', 'name': 'M2+NC', 'count': 2000, 'color': '#000000'},
            {'prefix': '', 'name': 'M2+SIRNA', 'count': 2000, 'color': '#D9D9D9'},
        ]

        self.init_ui()
        self.apply_style()
        self.on_step_selected(0)
        self.auto_detect_embedded_paths()

    def get_app_path(self):
        if getattr(sys, 'frozen', False):
            return getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
        else:
            return os.path.dirname(os.path.abspath(__file__))

    def get_embedded_tool_path(self, tool_subpath):
        app_root = self.get_app_path()
        exe_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else app_root
        for root in [app_root, exe_dir, os.path.dirname(exe_dir), os.path.dirname(os.path.dirname(exe_dir))]:
            candidate = os.path.join(root, tool_subpath)
            if os.path.exists(candidate): return os.path.abspath(candidate)
        return os.path.abspath(os.path.join(exe_dir, tool_subpath))

    def auto_detect_embedded_paths(self):
        self.log_text_edit.append("=================== System Path Diagnostics ===================")
        embedded_cp = self.get_embedded_tool_path("CellProfiler/CellProfiler.exe")
        if os.path.exists(embedded_cp):
            self.env_cp_path = embedded_cp
            self.log_text_edit.append("💡 Auto-detected CellProfiler.")

        local_pipeline = self.get_embedded_tool_path("pipeline_3D-1115.cppipe")
        if os.path.exists(local_pipeline):
            self.env_pipeline_path = os.path.abspath(local_pipeline)
            self.log_text_edit.append("💡 Auto-detected pipeline_3D-1115.cppipe.")

        embedded_rscript = self.get_embedded_tool_path("R-4.5.0/bin/Rscript.exe")
        if os.path.exists(embedded_rscript):
            self.env_rscript_path = embedded_rscript
            self.log_text_edit.append("💡 Auto-detected Rscript.")

        local_r_file = self.get_embedded_tool_path("tsne_merged.R")
        if os.path.exists(local_r_file):
            self.env_r_file = os.path.abspath(local_r_file)
            self.log_text_edit.append("💡 Auto-detected tsne_merged.R.")

        self.log_text_edit.append("===============================================================\n")

    def open_env_settings(self):
        dialog = EnvSettingsDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            self.env_cp_path = dialog.cp_path.text()
            self.env_pipeline_path = dialog.pipeline_path.text()
            self.env_rscript_path = dialog.rscript_path.text()
            self.env_r_file = dialog.r_file_path.text()
            self.log_text_edit.append("💡 Environment settings updated manually.")

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # ==================== Left Sidebar ====================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.folder_group = QGroupBox("1. Environment & Logs")
        folder_layout = QVBoxLayout(self.folder_group)

        self.btn_env_settings = QPushButton("⚙️ Environment Path Settings")
        self.btn_env_settings.clicked.connect(self.open_env_settings)
        folder_layout.addWidget(self.btn_env_settings)

        global_group_layout = QHBoxLayout()
        lbl_global_group = QLabel("Global Group By:")
        lbl_global_group.setStyleSheet("font-weight: bold; color: #2C3E50;")
        self.global_group_by = QComboBox()
        self.global_group_by.addItems(["Letter (e.g., 'B' from 'B02')", "Number (e.g., '02' from 'B02')"])
        global_group_layout.addWidget(lbl_global_group)
        global_group_layout.addWidget(self.global_group_by)
        folder_layout.addLayout(global_group_layout)

        self.tab_widget = QTabWidget()
        self.file_list_widget = QListWidget()
        self.tab_widget.addTab(self.file_list_widget, "📁 Raw File List")
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setStyleSheet("background-color: #1E1E1E; color: #00FF00; font-family: 'Consolas';")
        self.tab_widget.addTab(self.log_text_edit, "💬 Console Logs")

        folder_layout.addWidget(self.tab_widget)

        self.feature_group = QGroupBox("2. Pipeline Navigation")
        feature_layout = QVBoxLayout(self.feature_group)

        self.btn_concattif = QPushButton("1. concattif (2D to 3D)")
        self.btn_multiworker = QPushButton("2. CellProfiler Analysis")
        self.btn_extract = QPushButton("3. Extract & Group Parameters")
        self.btn_tsne_merged = QPushButton("4. Aggregated t-SNE Reduction")
        self.btn_pca_comprehensive = QPushButton("5. Aggregated PCA Scoring")

        self.step_buttons = [self.btn_concattif, self.btn_multiworker, self.btn_extract, self.btn_tsne_merged,
                             self.btn_pca_comprehensive]

        for i, btn in enumerate(self.step_buttons):
            btn.clicked.connect(lambda checked, idx=i: self.on_step_selected(idx))
            feature_layout.addWidget(btn)

        left_layout.addWidget(self.folder_group, stretch=6)
        left_layout.addWidget(self.feature_group, stretch=5)

        # ==================== Middle Panel ====================
        self.config_box = QGroupBox("Parameter Configuration")
        config_box_layout = QVBoxLayout(self.config_box)
        self.lbl_config_title = QLabel("⚡ Configuring: Step 1")
        self.lbl_config_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #3498DB; margin-bottom: 5px;")
        config_box_layout.addWidget(self.lbl_config_title)

        self.config_stack = QStackedWidget()
        config_box_layout.addWidget(self.config_stack, stretch=4)
        self.init_step_panels()

        self.group_config_box = QGroupBox("Mapping & Aggregation Configuration")
        group_vbox = QVBoxLayout(self.group_config_box)
        group_vbox.setContentsMargins(5, 5, 5, 5)

        self.lbl_detected_prefixes = QLabel("Detected Prefixes in Folder: None")
        self.lbl_detected_prefixes.setStyleSheet("color: #7F8C8D; font-style: italic;")
        self.lbl_detected_prefixes.setWordWrap(True)
        group_vbox.addWidget(self.lbl_detected_prefixes)

        self.btn_add_group = QPushButton("➕ Add Mapping Row")
        self.btn_add_group.setStyleSheet("background-color: #9B59B6; padding: 5px;")
        self.btn_add_group.clicked.connect(lambda: self.add_group_row("", "New", 2000, "#3498DB"))
        group_vbox.addWidget(self.btn_add_group)

        self.group_scroll = QScrollArea()
        self.group_scroll.setWidgetResizable(True)
        self.group_inner_widget = QWidget()
        self.group_layout = QVBoxLayout(self.group_inner_widget)
        self.group_layout.setContentsMargins(0, 0, 0, 0)
        self.group_layout.setAlignment(Qt.AlignTop)
        self.group_widgets = []

        for g in self.GROUP_CONFIG:
            self.add_group_row(g['prefix'], g['name'], g['count'], g['color'])

        self.group_scroll.setWidget(self.group_inner_widget)
        group_vbox.addWidget(self.group_scroll)
        config_box_layout.addWidget(self.group_config_box, stretch=3)
        self.group_config_box.hide()

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        config_box_layout.addWidget(divider)

        self.btn_run = QPushButton("⚡ Run Current Task")
        self.btn_run.setObjectName("runButton")
        self.btn_run.setEnabled(True)
        self.btn_run.clicked.connect(self.run_task)
        config_box_layout.addWidget(self.btn_run)

        # ==================== Right Panel ====================
        self.visual_box = QGroupBox("Dashboard & Preview")
        visual_layout = QVBoxLayout(self.visual_box)
        self.right_tabs = QTabWidget()

        self.tab_viz = QWidget()
        viz_layout = QVBoxLayout(self.tab_viz)
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        viz_layout.addWidget(self.canvas, stretch=6)

        self.lbl_preview_file = QLabel("Previewing Parameter Distributions")
        self.lbl_preview_file.setStyleSheet("color: #7F8C8D; font-size: 10px;")
        viz_layout.addWidget(self.lbl_preview_file)

        self.tab_preview = QWidget()
        preview_layout = QVBoxLayout(self.tab_preview)
        self.preview_table = QTableWidget()
        self.preview_table.setStyleSheet("font-size: 11px;")
        preview_layout.addWidget(self.preview_table)
        self.right_tabs.addTab(self.tab_viz, "📊 Plot & Stats")
        self.right_tabs.addTab(self.tab_preview, "📄 CSV Data Preview")

        visual_layout.addWidget(self.right_tabs)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(self.config_box)
        splitter.addWidget(self.visual_box)
        splitter.setSizes([250, 480, 550])
        main_layout.addWidget(splitter)

    # --- Step Panels ---
    def init_step_panels(self):
        def create_path_picker(line_edit, target_type='both'):
            box = QHBoxLayout()
            box.setContentsMargins(0, 0, 0, 0)
            box.addWidget(line_edit)
            if target_type in ['both', 'file']:
                btn_file = QPushButton("📄")
                btn_file.setFixedWidth(30)
                btn_file.clicked.connect(lambda: self.browse_path(line_edit, is_file=True))
                box.addWidget(btn_file)
            if target_type in ['both', 'folder']:
                btn_folder = QPushButton("📁")
                btn_folder.setFixedWidth(30)
                btn_folder.clicked.connect(lambda: self.browse_path(line_edit, is_file=False))
                box.addWidget(btn_folder)
            widget = QWidget()
            widget.setLayout(box)
            return widget

        # Step 1: concattif
        self.panel_1 = QWidget()
        layout_1 = QFormLayout(self.panel_1)

        self.btn_select_folder = QPushButton("📁 Select Raw 2D Folder / Auto-Link Paths")
        self.btn_select_folder.clicked.connect(self.select_folder)
        self.btn_select_folder.setStyleSheet("background-color: #2ECC71; color: white;")

        self.lbl_folder_path = QLabel("No folder selected")
        self.lbl_folder_path.setWordWrap(True)
        self.lbl_folder_path.setStyleSheet("color: #7F8C8D; font-size: 11px; margin-bottom: 10px;")

        layout_1.addRow(self.btn_select_folder)
        layout_1.addRow("Current Raw Path:", self.lbl_folder_path)

        self.s1_out_dir = QLineEdit()
        layout_1.addRow("3D TIF Output Dir:", create_path_picker(self.s1_out_dir, 'folder'))
        self.config_stack.addWidget(self.panel_1)

        # Step 2: CellProfiler
        self.panel_2 = QWidget()
        layout_2 = QFormLayout(self.panel_2)
        self.s2_in_dir = QLineEdit()
        self.s2_overlay_dir = QLineEdit()
        self.s2_csv_dir = QLineEdit()

        layout_2.addRow("3D TIF Input Dir:", create_path_picker(self.s2_in_dir, 'folder'))
        layout_2.addRow("Overlay Output Dir:", create_path_picker(self.s2_overlay_dir, 'folder'))
        layout_2.addRow("CSV Data Output Dir:", create_path_picker(self.s2_csv_dir, 'folder'))
        self.config_stack.addWidget(self.panel_2)

        # ===============================================
        # Step 3: Extract Parameters (🌟 Advanced Mode)
        # ===============================================
        self.panel_3 = QWidget()
        layout_3 = QFormLayout(self.panel_3)
        self.s3_in_csv_dir = QLineEdit()
        self.s3_in_csv_dir.textChanged.connect(self.on_s3_in_csv_dir_changed)
        self.s3_out_cell_dir = QLineEdit()
        self.s3_out_image_dir = QLineEdit()

        layout_3.addRow("Input CP CSV Dir:", create_path_picker(self.s3_in_csv_dir, 'folder'))
        layout_3.addRow("Output Cell CSV Dir:", create_path_picker(self.s3_out_cell_dir, 'folder'))
        layout_3.addRow("Output Image CSV Dir:", create_path_picker(self.s3_out_image_dir, 'folder'))

        # Extract Mode Selection
        self.s3_extract_mode = QComboBox()
        self.s3_extract_mode.addItem("Preset Parameters (13 Core Metrics)", "preset")
        self.s3_extract_mode.addItem("All Parameters (Keep Everything)", "all")
        self.s3_extract_mode.addItem("Custom Selection", "custom")
        self.s3_extract_mode.currentIndexChanged.connect(self.toggle_s3_custom_ui)
        layout_3.addRow("Extract Mode:", self.s3_extract_mode)

        # Custom Selection Widget
        self.s3_custom_widget = QWidget()
        s3_custom_layout = QVBoxLayout(self.s3_custom_widget)
        s3_custom_layout.setContentsMargins(0, 5, 0, 5)

        self.s3_custom_mode = QComboBox()
        self.s3_custom_mode.addItem("Include Selected", "include")
        self.s3_custom_mode.addItem("Exclude Selected", "exclude")

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Action for Selected:"))
        mode_layout.addWidget(self.s3_custom_mode)
        s3_custom_layout.addLayout(mode_layout)

        # Filters and Search
        filter_layout = QHBoxLayout()
        self.s3_filter1 = QComboBox()
        self.s3_filter1.currentTextChanged.connect(self.update_s3_filter2)
        self.s3_filter2 = QComboBox()
        self.s3_filter2.currentTextChanged.connect(self.update_s3_avail_list)

        filter_layout.addWidget(QLabel("Filter 1:"))
        filter_layout.addWidget(self.s3_filter1, stretch=1)
        filter_layout.addWidget(QLabel("Filter 2:"))
        filter_layout.addWidget(self.s3_filter2, stretch=1)
        s3_custom_layout.addLayout(filter_layout)

        self.s3_search = QLineEdit()
        self.s3_search.setPlaceholderText("Search parameters...")
        self.s3_search.textChanged.connect(self.update_s3_avail_list)
        s3_custom_layout.addWidget(self.s3_search)

        # Lists and Buttons
        lists_layout = QHBoxLayout()

        avail_layout = QVBoxLayout()
        avail_layout.addWidget(QLabel("Available Parameters:"))
        self.s3_avail_list = QListWidget()
        self.s3_avail_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        avail_layout.addWidget(self.s3_avail_list)

        btn_layout = QVBoxLayout()
        self.btn_s3_add = QPushButton(">> Add")
        self.btn_s3_add.clicked.connect(self.s3_add_params)
        self.btn_s3_remove = QPushButton("<< Remove")
        self.btn_s3_remove.clicked.connect(self.s3_remove_params)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_s3_add)
        btn_layout.addWidget(self.btn_s3_remove)
        btn_layout.addStretch()

        sel_layout = QVBoxLayout()
        sel_layout.addWidget(QLabel("Selected Parameters:"))
        self.s3_selected_list = QListWidget()
        self.s3_selected_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        sel_layout.addWidget(self.s3_selected_list)

        lists_layout.addLayout(avail_layout, stretch=2)
        lists_layout.addLayout(btn_layout)
        lists_layout.addLayout(sel_layout, stretch=2)

        s3_custom_layout.addLayout(lists_layout)
        layout_3.addRow(self.s3_custom_widget)
        self.s3_custom_widget.hide()  # Hidden by default

        self.config_stack.addWidget(self.panel_3)

        # Step 4: t-SNE (Aggregated)
        self.panel_4 = QWidget()
        layout_4 = QFormLayout(self.panel_4)
        self.s4_csv_path = QLineEdit()
        self.s4_csv_path.textChanged.connect(self.on_csv_path_changed)
        self.s4_out_dir = QLineEdit()

        self.s4_orientation = QComboBox()
        self.s4_orientation.addItems(["Rows are Samples (Standard)", "Columns are Samples (Transpose)"])
        self.s4_orientation.currentIndexChanged.connect(
            lambda: self.load_parameters_to_selectors(self.s4_csv_path.text()))

        self.s4_perplexity = QSpinBox()
        self.s4_perplexity.setRange(2, 100)
        self.s4_perplexity.setValue(30)
        self.s4_point_size = QDoubleSpinBox()
        self.s4_point_size.setRange(0.1, 20.0)
        self.s4_point_size.setSingleStep(0.5)
        self.s4_point_size.setValue(1.5)
        self.s4_point_alpha = QDoubleSpinBox()
        self.s4_point_alpha.setRange(0.1, 1.0)
        self.s4_point_alpha.setSingleStep(0.1)
        self.s4_point_alpha.setValue(0.6)

        self.s4_param_list = QListWidget()
        self.s4_param_list.setFixedHeight(120)
        self.s4_param_list.itemSelectionChanged.connect(self.update_histogram_preview)

        layout_4.addRow("Input Folder (Extracted):", create_path_picker(self.s4_csv_path, 'folder'))
        layout_4.addRow("TIFF Output Dir:", create_path_picker(self.s4_out_dir, 'folder'))
        layout_4.addRow("Data Orientation:", self.s4_orientation)
        layout_4.addRow("Perplexity:", self.s4_perplexity)
        layout_4.addRow("Point Size:", self.s4_point_size)
        layout_4.addRow("Point Transparency:", self.s4_point_alpha)
        layout_4.addRow("Select Params\n(Tick=Use, Click=Preview):", self.s4_param_list)
        self.config_stack.addWidget(self.panel_4)

        # Step 5: PCA Comprehensive (Aggregated)
        self.panel_5 = QWidget()
        layout_5 = QFormLayout(self.panel_5)
        self.s5_csv_path = QLineEdit()
        self.s5_csv_path.textChanged.connect(self.on_csv_path_changed)
        self.s5_out_dir = QLineEdit()

        self.s5_orientation = QComboBox()
        self.s5_orientation.addItems(["Rows are Samples (Standard)", "Columns are Samples (Transpose)"])
        self.s5_orientation.currentIndexChanged.connect(
            lambda: self.load_parameters_to_selectors(self.s5_csv_path.text()))

        self.s5_point_size = QDoubleSpinBox()
        self.s5_point_size.setRange(0.1, 20.0)
        self.s5_point_size.setSingleStep(0.5)
        self.s5_point_size.setValue(3.0)

        self.s5_point_alpha = QDoubleSpinBox()
        self.s5_point_alpha.setRange(0.01, 1.0)
        self.s5_point_alpha.setSingleStep(0.05)
        self.s5_point_alpha.setValue(0.05)

        self.s5_param_list = QListWidget()
        self.s5_param_list.setFixedHeight(150)
        self.s5_param_list.itemSelectionChanged.connect(self.update_histogram_preview)

        layout_5.addRow("Input Folder (Extracted):", create_path_picker(self.s5_csv_path, 'folder'))
        layout_5.addRow("Analysis Output Dir:", create_path_picker(self.s5_out_dir, 'folder'))
        layout_5.addRow("Data Orientation:", self.s5_orientation)
        layout_5.addRow("Scatter Point Size:", self.s5_point_size)
        layout_5.addRow("Point Transparency:", self.s5_point_alpha)
        layout_5.addRow("Select Params\n(Tick=Use, Click=Preview):", self.s5_param_list)
        self.config_stack.addWidget(self.panel_5)

    def browse_path(self, line_edit, is_file):
        if is_file:
            path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        else:
            path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if path: line_edit.setText(path)

    # ==========================
    # Step 3 Advanced Parameter Selection Logic
    # ==========================
    def toggle_s3_custom_ui(self):
        if self.s3_extract_mode.currentData() == 'custom':
            self.s3_custom_widget.show()
        else:
            self.s3_custom_widget.hide()

    def on_s3_in_csv_dir_changed(self, path):
        if not os.path.isdir(path): return
        cell_csv = os.path.join(path, "Control_Cell_Watershed.csv")
        img_csv = os.path.join(path, "Control_Image.csv")

        cols = set()
        for f in [cell_csv, img_csv]:
            if os.path.exists(f):
                try:
                    df = pd.read_csv(f, nrows=0, encoding='utf-8-sig')
                    cols.update([str(c).strip() for c in df.columns])
                except Exception:
                    pass
        if cols:
            self.s3_all_params = sorted(list(cols))
            self.update_s3_filter1()
            self.log_text_edit.append(f"💡 Detected {len(self.s3_all_params)} unique parameters for extraction.")

    def update_s3_filter1(self):
        prefixes = set()
        for p in self.s3_all_params:
            parts = p.split('_')
            if len(parts) > 0 and parts[0]:
                prefixes.add(parts[0])

        self.s3_filter1.blockSignals(True)
        self.s3_filter1.clear()
        self.s3_filter1.addItem("All", "")
        for pf in sorted(prefixes):
            self.s3_filter1.addItem(pf, pf)
        self.s3_filter1.blockSignals(False)
        self.update_s3_filter2()

    def update_s3_filter2(self):
        f1 = self.s3_filter1.currentData()
        prefixes = set()
        for p in self.s3_all_params:
            parts = p.split('_')
            if len(parts) > 1 and parts[1]:
                if not f1 or parts[0] == f1:
                    prefixes.add(parts[1])

        self.s3_filter2.blockSignals(True)
        self.s3_filter2.clear()
        self.s3_filter2.addItem("All", "")
        for pf in sorted(prefixes):
            self.s3_filter2.addItem(pf, pf)
        self.s3_filter2.blockSignals(False)
        self.update_s3_avail_list()

    def update_s3_avail_list(self):
        f1 = self.s3_filter1.currentData()
        f2 = self.s3_filter2.currentData()
        search_txt = self.s3_search.text().lower()

        self.s3_avail_list.clear()
        for p in self.s3_all_params:
            parts = p.split('_')
            p1 = parts[0] if len(parts) > 0 else ""
            p2 = parts[1] if len(parts) > 1 else ""

            if f1 and p1 != f1: continue
            if f2 and p2 != f2: continue
            if search_txt and search_txt not in p.lower(): continue

            self.s3_avail_list.addItem(p)

    def s3_add_params(self):
        selected = self.s3_avail_list.selectedItems()
        for item in selected:
            text = item.text()
            exists = False
            for i in range(self.s3_selected_list.count()):
                if self.s3_selected_list.item(i).text() == text:
                    exists = True
                    break
            if not exists:
                self.s3_selected_list.addItem(text)

    def s3_remove_params(self):
        selected = self.s3_selected_list.selectedItems()
        for item in selected:
            self.s3_selected_list.takeItem(self.s3_selected_list.row(item))

    def get_selected_features(self, list_widget):
        features = []
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            if item.checkState() == Qt.Checked:
                features.append(item.text())
        return features

    # --- UI Group Elements ---
    def add_group_row(self, prefix, name, count, color):
        row_widget = QWidget()
        layout = QHBoxLayout(row_widget)
        layout.setContentsMargins(0, 2, 0, 2)

        chk = QCheckBox()
        chk.setChecked(True)

        prefix_edit = QLineEdit(prefix)
        prefix_edit.setPlaceholderText("Prefix (e.g. 02)")
        prefix_edit.setToolTip("File Prefix to scan in folder")

        name_edit = QLineEdit(name)
        count_spin = QSpinBox()
        count_spin.setRange(0, 1000000)
        count_spin.setValue(count)
        count_spin.setToolTip("Lines to read (0 = Read All)")

        color_btn = QPushButton()
        color_btn.setFixedSize(20, 20)
        color_btn.setStyleSheet(f"background-color: {color}; border-radius: 4px; border: 1px solid #7F8C8D;")
        color_btn.color_val = color

        def change_color():
            c = QColorDialog.getColor(QColor(color_btn.color_val))
            if c.isValid():
                color_btn.color_val = c.name()
                color_btn.setStyleSheet(f"background-color: {c.name()}; border-radius: 4px; border: 1px solid #7F8C8D;")

        color_btn.clicked.connect(change_color)

        del_btn = QPushButton("🗑️")
        del_btn.setFixedSize(25, 25)
        del_btn.setStyleSheet("background-color: transparent; border: none; font-size: 14px;")
        del_btn.clicked.connect(lambda: self.remove_group_row(row_widget))

        layout.addWidget(chk)
        layout.addWidget(prefix_edit, stretch=2)
        layout.addWidget(name_edit, stretch=3)
        layout.addWidget(count_spin, stretch=2)
        layout.addWidget(color_btn, stretch=1)
        layout.addWidget(del_btn, stretch=1)

        self.group_layout.addWidget(row_widget)
        self.group_widgets.append({
            'widget': row_widget, 'chk': chk, 'prefix': prefix_edit, 'name': name_edit, 'count': count_spin,
            'color_btn': color_btn
        })

    def remove_group_row(self, widget):
        for item in self.group_widgets:
            if item['widget'] == widget:
                self.group_widgets.remove(item)
                widget.deleteLater()
                break

    def get_current_group_config(self):
        config = []
        for item in self.group_widgets:
            if item['chk'].isChecked():
                config.append({
                    'prefix': item['prefix'].text().strip(),
                    'name': item['name'].text().strip(),
                    'count': item['count'].value(),
                    'color': item['color_btn'].color_val
                })
        return config

    # --- Logic ---
    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Raw 2D Folder")
        if folder_path:
            self.raw_folder_path = os.path.abspath(folder_path)
            self.lbl_folder_path.setText(f"{self.raw_folder_path}")
            self.file_list_widget.clear()

            try:
                files = os.listdir(self.raw_folder_path)
                tif_files = [f for f in files if f.lower().endswith(('.tif', '.tiff'))]
                if tif_files: self.file_list_widget.addItems(tif_files)
            except Exception:
                pass

            parent_dir = os.path.dirname(self.raw_folder_path)
            base_name = os.path.basename(self.raw_folder_path)
            for suffix in ["_stacked", "_overlay", "_csv", "_extracted_cell", "_PCA_output", "_tSNE_output"]:
                if base_name.endswith(suffix):
                    base_name = base_name[:-len(suffix)]
                    break

            self.s1_out_dir.setText(os.path.join(parent_dir, f"{base_name}_stacked"))
            self.s2_in_dir.setText(os.path.join(parent_dir, f"{base_name}_stacked"))
            self.s2_overlay_dir.setText(os.path.join(parent_dir, f"{base_name}_overlay"))
            self.s2_csv_dir.setText(os.path.join(parent_dir, f"{base_name}_csv"))

            self.s3_in_csv_dir.setText(os.path.join(parent_dir, f"{base_name}_csv"))
            self.s3_out_cell_dir.setText(os.path.join(parent_dir, f"{base_name}_extracted_cell"))
            self.s3_out_image_dir.setText(os.path.join(parent_dir, f"{base_name}_extracted_image"))

            self.s4_csv_path.setText(os.path.join(parent_dir, f"{base_name}_extracted_cell"))
            self.s4_out_dir.setText(os.path.join(parent_dir, f"{base_name}_tSNE_output"))

            self.s5_csv_path.setText(os.path.join(parent_dir, f"{base_name}_extracted_cell"))
            self.s5_out_dir.setText(os.path.join(parent_dir, f"{base_name}_PCA_output"))

    def on_csv_path_changed(self, path):
        if not os.path.exists(path): return

        target_csv = None
        if os.path.isfile(path) and path.lower().endswith('.csv'):
            target_csv = path
        elif os.path.isdir(path):
            for f in os.listdir(path):
                if f.lower().endswith('.csv'):
                    target_csv = os.path.join(path, f)
                    break
            self.scan_folder_for_prefixes(path)

        if target_csv:
            self.load_csv_preview(target_csv)
            self.load_parameters_to_selectors(target_csv)

    def scan_folder_for_prefixes(self, folder_path):
        if not os.path.isdir(folder_path): return
        prefixes = set()
        for f in os.listdir(folder_path):
            if f.lower().endswith('.csv') and '_' in f:
                pref = f.split('_')[0]
                prefixes.add(pref)

        if prefixes:
            self.lbl_detected_prefixes.setText(f"Detected Prefixes in Folder: {', '.join(sorted(prefixes))}")
            self.lbl_detected_prefixes.setStyleSheet("color: #27AE60; font-weight: bold;")
        else:
            self.lbl_detected_prefixes.setText("Detected Prefixes in Folder: None")
            self.lbl_detected_prefixes.setStyleSheet("color: #E74C3C; font-weight: bold;")

    def load_csv_preview(self, path):
        try:
            df = pd.read_csv(path, nrows=30, encoding='utf-8-sig')
            self.preview_table.setRowCount(df.shape[0])
            self.preview_table.setColumnCount(df.shape[1])
            self.preview_table.setHorizontalHeaderLabels(df.columns.astype(str))
            for row in range(df.shape[0]):
                for col in range(df.shape[1]):
                    self.preview_table.setItem(row, col, QTableWidgetItem(str(df.iloc[row, col])))
            self.log_text_edit.append(f"💡 Preview loaded from: {os.path.basename(path)}")
        except Exception:
            pass

    def load_parameters_to_selectors(self, path_or_dir):
        if not path_or_dir or not os.path.exists(path_or_dir): return

        csv_path = path_or_dir if os.path.isfile(path_or_dir) else next(
            (os.path.join(path_or_dir, f) for f in os.listdir(path_or_dir) if f.endswith('.csv')), None)
        if not csv_path: return

        try:
            if self.current_step == 3:
                is_rows_are_samples = (self.s4_orientation.currentIndex() == 0)
            elif self.current_step == 4:
                is_rows_are_samples = (self.s5_orientation.currentIndex() == 0)
            else:
                is_rows_are_samples = True

            if is_rows_are_samples:
                df_temp = pd.read_csv(csv_path, nrows=1)
                features = [col for col in df_temp.columns if col not in ['Metadata_Well', 'Type', 'Group']]
            else:
                df_temp = pd.read_csv(csv_path, usecols=[0])
                features = df_temp.iloc[:, 0].astype(str).tolist()

            target_list = self.s4_param_list if self.current_step == 3 else self.s5_param_list
            target_list.clear()
            for f in features:
                item = QListWidgetItem(f)
                item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                item.setCheckState(Qt.Checked)
                target_list.addItem(item)

            if target_list.count() > 0: target_list.setCurrentRow(0)
            self.update_histogram_preview()
        except Exception as e:
            self.log_text_edit.append(f"⚠️ Error parsing features: {e}")

    def update_histogram_preview(self):
        list_widget = self.s4_param_list if self.current_step == 3 else self.s5_param_list
        selected_items = list_widget.selectedItems()
        if not selected_items: return

        param_name = selected_items[0].text()
        path = self.s4_csv_path.text() if self.current_step == 3 else self.s5_csv_path.text()
        if not os.path.exists(path): return

        target_csv = path if os.path.isfile(path) else next(
            (os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')), None)
        if not target_csv: return

        try:
            df = pd.read_csv(target_csv)
            is_rows_are_samples = (self.s4_orientation.currentIndex() == 0) if self.current_step == 3 else (
                    self.s5_orientation.currentIndex() == 0)

            if is_rows_are_samples:
                if param_name in df.columns:
                    data = pd.to_numeric(df[param_name], errors='coerce').dropna().values
                else:
                    return
            else:
                row_data = df[df.iloc[:, 0].astype(str) == param_name]
                if not row_data.empty:
                    data = pd.to_numeric(row_data.iloc[0, 1:], errors='coerce').dropna().values
                else:
                    return

            if data is None or len(data) == 0: return

            mean_val, std_val = np.mean(data), np.std(data)

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.hist(data, bins=30, color="#3498DB", edgecolor="#2980B9", alpha=0.8, density=True)
            ax.axvline(mean_val, color='#E74C3C', linestyle='solid', linewidth=1.5, label=f"Mean: {mean_val:.4f}")
            ax.axvline(mean_val - std_val, color='#9B59B6', linestyle='dashed',
                       label=f"Mean-Std: {(mean_val - std_val):.4f}")
            ax.axvline(mean_val + std_val, color='#9B59B6', linestyle='dashed',
                       label=f"Mean+Std: {(mean_val + std_val):.4f}")
            ax.set_title(f"Distribution of {param_name}", fontsize=10, fontweight='bold')
            ax.legend(frameon=False, fontsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            self.canvas.draw()
            self.lbl_preview_file.setText(f"Previewing: {os.path.basename(target_csv)}")
        except Exception:
            pass

    def on_step_selected(self, index):
        self.current_step = index
        self.config_stack.setCurrentIndex(index)
        step_names = [
            "Step 1: concattif",
            "Step 2: CellProfiler",
            "Step 3: Extract & Group Parameters",
            "Step 4: Aggregated t-SNE",
            "Step 5: Aggregated PCA"
        ]
        self.lbl_config_title.setText(f"⚡ Configuring: {step_names[index]}")

        for i, btn in enumerate(self.step_buttons):
            btn.setStyleSheet("background-color: #2ECC71; font-weight: bold;" if i == index else "")

        if index >= 3:
            self.group_config_box.show()
        else:
            self.group_config_box.hide()

        target_path = None
        if index == 3:
            target_path = self.s4_csv_path.text()
        elif index == 4:
            target_path = self.s5_csv_path.text()

        if target_path:
            self.load_parameters_to_selectors(target_path)
            self.scan_folder_for_prefixes(target_path)

    def display_output_image_on_canvas(self):
        try:
            if self.current_step == 3:
                out_dir = self.s4_out_dir.text()
                img_path = os.path.join(out_dir, "Aggregated_tSNE_Result.tiff")
                if img_path and os.path.exists(img_path):
                    import matplotlib.image as mpimg
                    img = mpimg.imread(img_path)
                    self.figure.clear()
                    ax = self.figure.add_subplot(111)
                    ax.imshow(img)
                    ax.axis('off')
                    self.canvas.draw()

            elif self.current_step == 4:
                out_dir = self.s5_out_dir.text()
                img_path = os.path.join(out_dir, "Aggregated_PCA_Boxplot.png")

                if img_path and os.path.exists(img_path):
                    import matplotlib.image as mpimg
                    img = mpimg.imread(img_path)
                    self.figure.clear()
                    ax = self.figure.add_subplot(111)
                    ax.imshow(img)
                    ax.axis('off')
                    self.canvas.draw()
        except Exception as e:
            pass

    # ==================================
    # 后台线程分发任务
    # ==================================
    def run_task(self):
        try:
            self.tab_widget.setCurrentIndex(1)
            self.log_text_edit.clear()
            self.btn_run.setEnabled(False)

            if self.current_step == 0:
                if not self.raw_folder_path:
                    raise Exception("Please select the Raw 2D Folder first!")
                self.worker = WorkerThread(run_concat_tif, self.raw_folder_path, self.s1_out_dir.text())

            elif self.current_step == 1:
                self.worker = WorkerThread(
                    run_cellprofiler_by_cli,
                    self.env_cp_path, self.s2_in_dir.text(), self.s2_overlay_dir.text(),
                    self.s2_csv_dir.text(), self.env_pipeline_path or None
                )

            elif self.current_step == 2:
                extract_mode = self.s3_extract_mode.currentData()
                custom_mode = self.s3_custom_mode.currentData()

                custom_params = []
                if extract_mode == 'custom':
                    for i in range(self.s3_selected_list.count()):
                        custom_params.append(self.s3_selected_list.item(i).text())
                    if not custom_params:
                        raise Exception("Custom Selection is empty! Please add parameters or change Extract Mode.")

                self.worker = WorkerThread(
                    extract_target_parameters,
                    input_csv_dir=self.s3_in_csv_dir.text(),
                    out_cell_dir=self.s3_out_cell_dir.text(),
                    out_image_dir=self.s3_out_image_dir.text(),
                    group_by=self.global_group_by.currentText(),
                    extract_mode=extract_mode,
                    custom_mode=custom_mode,
                    custom_params=custom_params
                )

            elif self.current_step == 3:
                config = self.get_current_group_config()
                selected_feats = self.get_selected_features(self.s4_param_list)
                if not config or not selected_feats:
                    raise Exception("Please CHECK at least one Group and TICK at least one Parameter!")

                def tsne_wrapper(status_callback=None):
                    temp_csv, adj_config = do_aggregation(
                        self.s4_csv_path.text(), self.s4_out_dir.text(),
                        config, (self.s4_orientation.currentIndex() == 0),
                        selected_feats, status_callback
                    )
                    if not temp_csv: return False, "Data aggregation failed."

                    success = run_merged_tsne_by_r(
                        r_script_path=self.env_r_file, input_path=temp_csv,
                        output_dir=self.s4_out_dir.text(), group_config=adj_config,
                        is_rows_are_samples=(self.s4_orientation.currentIndex() == 0),
                        perplexity=self.s4_perplexity.value(), point_size=self.s4_point_size.value(),
                        point_alpha=self.s4_point_alpha.value(), selected_features=selected_feats,
                        rscript_exec=self.env_rscript_path, status_callback=status_callback
                    )
                    if success:
                        out_tiff = os.path.join(self.s4_out_dir.text(), "temp_aggregated_master_tsne.tiff")
                        final_tiff = os.path.join(self.s4_out_dir.text(), "Aggregated_tSNE_Result.tiff")
                        if os.path.exists(out_tiff): shutil.move(out_tiff, final_tiff)
                    if os.path.exists(temp_csv): os.remove(temp_csv)
                    return success, "Aggregated t-SNE completed."

                self.worker = WorkerThread(tsne_wrapper)

            elif self.current_step == 4:
                config = self.get_current_group_config()
                selected_feats = self.get_selected_features(self.s5_param_list)
                if not config or not selected_feats:
                    raise Exception("Please CHECK at least one Group and TICK at least one Parameter!")

                def pca_wrapper(status_callback=None):
                    temp_csv, adj_config = do_aggregation(
                        self.s5_csv_path.text(), self.s5_out_dir.text(),
                        config, (self.s5_orientation.currentIndex() == 0),
                        selected_feats, status_callback
                    )
                    if not temp_csv: return False, "Data aggregation failed."

                    success = run_pca_comprehensive(
                        input_path=temp_csv, output_dir=self.s5_out_dir.text(),
                        group_config=adj_config,
                        is_rows_are_samples=(self.s5_orientation.currentIndex() == 0),
                        scatter_size=self.s5_point_size.value(), alpha=self.s5_point_alpha.value(),
                        selected_features=selected_feats, status_callback=status_callback
                    )
                    if success:
                        out_score = os.path.join(self.s5_out_dir.text(),
                                                 "temp_aggregated_master_comprehensive_score.csv")
                        final_score = os.path.join(self.s5_out_dir.text(), "Aggregated_PCA_Score.csv")
                        if os.path.exists(out_score): shutil.move(out_score, final_score)
                        out_svg = os.path.join(self.s5_out_dir.text(),
                                               "temp_aggregated_master_comprehensive_score_boxplot.svg")
                        final_svg = os.path.join(self.s5_out_dir.text(), "Aggregated_PCA_Boxplot.svg")
                        if os.path.exists(out_svg): shutil.move(out_svg, final_svg)
                        out_png = os.path.join(self.s5_out_dir.text(),
                                               "temp_aggregated_master_comprehensive_score_boxplot.png")
                        final_png = os.path.join(self.s5_out_dir.text(), "Aggregated_PCA_Boxplot.png")
                        if os.path.exists(out_png): shutil.move(out_png, final_png)
                    if os.path.exists(temp_csv): os.remove(temp_csv)
                    return success, "Aggregated PCA completed."

                self.worker = WorkerThread(pca_wrapper)

            self.worker.log_signal.connect(self.log_text_edit.append)
            self.worker.finished_signal.connect(self.task_finished)
            self.worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Validation Error", str(e))
            self.btn_run.setEnabled(True)

    def task_finished(self, success, message):
        self.btn_run.setEnabled(True)
        if success:
            QMessageBox.information(self, "Information", "Task Completed!")
            self.display_output_image_on_canvas()
        else:
            QMessageBox.critical(self, "Error", f"Execution failed:\n{message}")

    def apply_style(self):
        qss = """
            QMainWindow { background-color: #F4F6F9; }
            QGroupBox {
                background-color: #FFFFFF; border: 1px solid #DCDFE6;
                border-radius: 8px; margin-top: 15px; font-weight: bold; color: #2C3E50;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 15px; background-color: #FFFFFF; }
            QPushButton {
                background-color: #3498DB; color: white; border: none;
                border-radius: 5px; padding: 7px 12px; font-weight: bold;
            }
            QPushButton:hover { background-color: #2980B9; }
            QPushButton#runButton { background-color: #2ECC71; font-size: 14px; padding: 10px; }
            QPushButton#runButton:hover { background-color: #27AE60; }
            QPushButton:disabled { background-color: #C0C4CC; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                border: 1px solid #DCDFE6; border-radius: 4px; padding: 4px;
            }
            QListWidget {
                border: 1px solid #DCDFE6; border-radius: 4px;
                background-color: #FDFDFD; padding: 2px;
            }
            QListWidget::item { padding: 4px; border-bottom: 1px solid #F0F0F0;}
            QListWidget::item:selected { background-color: #E1F0FF; color: #000; }
            QTableWidget { gridline-color: #DCDFE6; border: 1px solid #DCDFE6; }
            QHeaderView::section { background-color: #F2F6FC; font-weight: bold; }
        """
        self.setStyleSheet(qss)


if __name__ == "__main__":
    import matplotlib

    matplotlib.use('Agg')
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())