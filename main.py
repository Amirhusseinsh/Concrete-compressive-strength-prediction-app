import sys
import numpy as np
import joblib
from catboost import CatBoostRegressor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QLabel, QLineEdit, QPushButton, QGroupBox, QFormLayout,
    QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QDoubleValidator

# ================== Load Model and Scaler ==================
MODEL_PATH = "best_model_CatBoost_fs8.joblib"
SCALER_PATH = "scaler_fs8.joblib"

FEATURE_NAMES = [
    "Age",
    "Water",
    "Cement",
    "Fine_Aggregates",
    "Coarse_aaggregate",
    "Superplasticizer",
    "Max_grain_size",
    "Volume_of_paste"
]

FEATURE_LABELS = [
    "Age",
    "Water",
    "Cement",
    "Fine Aggregates",
    "Coarse Aggregate",
    "Superplasticizer",
    "Max Grain Size",
    "Volume of Paste"
]

class StrengthPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Compressive Strength Prediction")
        self.setFixedSize(520, 720)
        self.setStyleSheet(self.get_stylesheet())

        try:
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model or scaler:\n{str(e)}")
            sys.exit(1)

        self.init_ui()

    def get_stylesheet(self):
        return """
            QMainWindow {
                background-color: #f0f4f8;
            }
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: #1a365d;
                border: 2px solid #cbd5e0;
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
            }
            QLabel {
                font-size: 13px;
                color: #2d3748;
            }
            QLineEdit {
                padding: 8px 12px;
                border: 1.5px solid #a0aec0;
                border-radius: 6px;
                font-size: 13px;
                background-color: #ffffff;
                color: black;
            }
            QLineEdit:focus {
                border: 2px solid #3182ce;
            }
            QPushButton {
                background-color: #2b6cb0;
                color: white;
                font-size: 15px;
                font-weight: bold;
                padding: 12px;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background-color: #2c5282;
            }
            QPushButton:pressed {
                background-color: #1a365d;
            }
            #resultBox {
                background-color: #c6f6d5;
                border: 2px solid #68d391;
                border-radius: 10px;
                padding: 15px;
            }
            #resultTitle {
                font-size: 14px;
                color: #276749;
                font-weight: bold;
            }
            #resultValue {
                font-size: 28px;
                font-weight: bold;
                color: #22543d;
            }
        """

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Title
        title = QLabel("Compressive Strength Prediction")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #1a365d; margin-bottom: 5px;")
        main_layout.addWidget(title)

        subtitle = QLabel("CatBoost Model | 8 Input Parameters")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #718096; font-size: 12px;")
        main_layout.addWidget(subtitle)

        # Input Group
        input_group = QGroupBox("Input Parameters")
        form_layout = QFormLayout()
        form_layout.setSpacing(12)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.inputs = []
        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)

        for label_text in FEATURE_LABELS:
            line_edit = QLineEdit()
            line_edit.setValidator(validator)
            line_edit.setPlaceholderText("Enter value...")
            line_edit.setStyleSheet("color: black;")
            form_layout.addRow(QLabel(label_text + " :"), line_edit)
            self.inputs.append(line_edit)

        input_group.setLayout(form_layout)
        main_layout.addWidget(input_group)

        # Predict Button
        self.predict_btn = QPushButton("Predict Compressive Strength")
        self.predict_btn.clicked.connect(self.predict)
        main_layout.addWidget(self.predict_btn)

        # ========== Result Area ==========
        result_container = QWidget()
        result_container.setObjectName("resultBox")
        result_layout = QVBoxLayout(result_container)
        result_layout.setContentsMargins(10, 10, 10, 10)
        result_layout.setSpacing(5)

        self.result_title = QLabel("Predicted Compressive Strength")
        self.result_title.setObjectName("resultTitle")
        self.result_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.result_value = QLabel("--.-- MPa")
        self.result_value.setObjectName("resultValue")
        self.result_value.setAlignment(Qt.AlignmentFlag.AlignCenter)

        result_layout.addWidget(self.result_title)
        result_layout.addWidget(self.result_value)

        main_layout.addWidget(result_container)

        # Note
        note = QLabel("Note: Values should be within the range of the training data.")
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        note.setStyleSheet("color: #a0aec0; font-size: 11px; margin-top: 5px;")
        main_layout.addWidget(note)

    def predict(self):
        try:
            values = []
            for i, line_edit in enumerate(self.inputs):
                text = line_edit.text().strip()
                if not text:
                    QMessageBox.warning(self, "Missing Input",
                        f"Please enter a value for «{FEATURE_LABELS[i]}».")
                    return
                values.append(float(text.replace(",", ".")))

            print("=== Debug ===")
            print("Values entered:", values)        
            print("Feature order:", FEATURE_NAMES)

            X = np.array(values).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            print("Scaled values:", X_scaled)         

            pred = self.model.predict(X_scaled)[0]
            print("Prediction:", pred)

            self.result_value.setText(f"{pred:.2f} MPa")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed:\n{str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StrengthPredictorApp()
    window.show()
    sys.exit(app.exec())