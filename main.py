import os
import sys
import torch
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QTextEdit, QPushButton, QLabel, QLineEdit,
                             QListWidget, QListWidgetItem, QToolBar, QFileDialog,
                             QProgressBar, QFrame, QMessageBox, QDockWidget, QMainWindow,
                             QAction, QMenuBar)
from PyQt5.QtGui import QPixmap
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import time
import pickle

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def load_image(image_file, input_size=448):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    pixel_values = transform(image).unsqueeze(0)
    return pixel_values

class LLMWorker(QThread):
    result_ready = pyqtSignal(str, str)
    progress_update = pyqtSignal(str)

    def __init__(self, model=None, tokenizer=None, system_prompt="", question="", max_tokens=256):
        super().__init__()
        self.image_path = None
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.question = question
        self.max_tokens = max_tokens

    def set_image(self, image_path, system_prompt, question):
        self.image_path = image_path
        self.system_prompt = system_prompt
        self.question = question

    def run(self):
        if not self.image_path or not self.model or not self.tokenizer:
            return
        try:
            self.progress_update.emit(f"Processing {self.image_path}...")
            pixel_values = load_image(self.image_path).to(torch.float16)
            task = self.system_prompt or (
                "Your task is to give a detailed description for image generation.")
            question = f"<image>\n{task}\n{self.question}"
            generation_config = {'max_new_tokens': self.max_tokens, 'do_sample': False}
            retries = 3
            while retries > 0:
                try:
                    response, history = self.model.chat(self.tokenizer, pixel_values, question, generation_config, return_history=True)
                    self.result_ready.emit(self.image_path, response)
                    self.progress_update.emit(f"Successfully processed {self.image_path}")
                    return
                except Exception as e:
                    retries -= 1
                    self.progress_update.emit(f"Error: {e}. Retries left: {retries}")
                    time.sleep(1)
            self.result_ready.emit(self.image_path, "Error: Could not process image after retries.")
        except Exception as e:
            self.result_ready.emit(self.image_path, f"Error: {str(e)}")
            self.progress_update.emit(f"Error: {str(e)}")

class ImageListItemWidget(QWidget):
    def __init__(self, image_path, text_edit_callback, image_select_callback):
        super().__init__()
        self.image_path = image_path
        self.text_edit_callback = text_edit_callback
        self.image_select_callback = image_select_callback

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        self.image_label = QLabel(self)
        pixmap = QPixmap(image_path).scaled(128, 128, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)
        layout.addWidget(self.image_label)

        self.text_edit = QTextEdit(self)
        self.text_edit.setFixedHeight(100)
        self.text_edit.textChanged.connect(
            lambda: self.text_edit_callback(self.image_path, self.text_edit.toPlainText()))
        layout.addWidget(self.text_edit)

        self.setLayout(layout)

    def set_text(self, text):
        self.text_edit.setPlainText(text)

class VisualLLMPlayground(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.worker = LLMWorker()
        self.image_queue = []
        self.project_data = {}  # Store the project metadata
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Visual LLM Playground")
        self.setGeometry(100, 100, 1200, 800)

        # Central widget setup
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # Left Dock for controls and image list
        dock = QDockWidget("Controls", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea)

        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        # System Prompt
        self.system_prompt_edit = QTextEdit(self)
        self.system_prompt_edit.setPlaceholderText("System Prompt")
        self.system_prompt_edit.setPlainText(
            "Your task is to give a detailed description for image generation."
        )

        # Question input
        self.question_edit = QTextEdit(self)
        self.question_edit.setPlaceholderText("Question")
        self.question_edit.setPlainText("Describe this image in detail:")

        # Buttons
        load_model_btn = QPushButton("Load Model", self)
        load_model_btn.clicked.connect(self.load_model)

        process_all_btn = QPushButton("Process All Images", self)
        process_all_btn.clicked.connect(self.process_all_images)

        process_single_btn = QPushButton("Process Selected Image", self)
        process_single_btn.clicked.connect(self.process_selected_image)

        # Progress bar
        self.progress_bar = QProgressBar(self)

        # Add controls to layout
        control_layout.addWidget(QLabel("System Prompt:"))
        control_layout.addWidget(self.system_prompt_edit)
        control_layout.addWidget(QLabel("Question:"))
        control_layout.addWidget(self.question_edit)
        control_layout.addWidget(load_model_btn)
        control_layout.addWidget(process_single_btn)
        control_layout.addWidget(process_all_btn)
        control_layout.addWidget(self.progress_bar)

        # Set the widget for the dock
        dock.setWidget(control_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        # Image list and preview in the central widget
        self.image_list = QListWidget(self)
        self.image_list.itemClicked.connect(self.display_image_info)
        layout.addWidget(self.image_list)

        # Central Widget for image display and result
        self.image_display = QLabel(self)
        self.image_display.setFixedSize(400, 400)

        self.result_text = QTextEdit(self)

        central_display_layout = QVBoxLayout()
        central_display_layout.addWidget(self.image_display)
        central_display_layout.addWidget(self.result_text)

        layout.addLayout(central_display_layout)

        self.worker.result_ready.connect(self.on_llm_result)
        self.worker.progress_update.connect(self.update_status)

        # Create menu bar and actions
        self.create_menu()

    def create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')

        import_images_action = QAction('Import Image(s)', self)
        import_images_action.triggered.connect(self.import_images)

        import_folder_action = QAction('Import Folder(s)', self)
        import_folder_action.triggered.connect(self.import_folders)

        export_images_action = QAction('Export Images', self)
        export_images_action.triggered.connect(self.export_images)

        save_project_action = QAction('Save Project', self)
        save_project_action.triggered.connect(self.save_project)

        load_project_action = QAction('Load Project', self)
        load_project_action.triggered.connect(self.load_project)

        file_menu.addAction(import_images_action)
        file_menu.addAction(import_folder_action)
        file_menu.addAction(export_images_action)
        file_menu.addSeparator()
        file_menu.addAction(save_project_action)
        file_menu.addAction(load_project_action)

    def load_model(self):
        model_name = 'OpenGVLab/InternVL2-8B'
        quant_config = BitsAndBytesConfig(load_in_8bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, quantization_config=quant_config, trust_remote_code=True).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.worker.model = self.model
        self.worker.tokenizer = self.tokenizer
        self.update_status("Model loaded successfully")

    def import_images(self):
        image_files, _ = QFileDialog.getOpenFileNames(self, "Import Images", "", "Images (*.png *.xpm *.jpg *.jpeg)")
        for image_file in image_files:
            self.add_image_to_list(image_file)

    def import_folders(self):
        folder = QFileDialog.getExistingDirectory(self, "Import Folder")
        if folder:
            for image_file in os.listdir(folder):
                if image_file.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(folder, image_file)
                    self.add_image_to_list(image_path)

    def add_image_to_list(self, image_path):
        if image_path not in self.project_data:
            self.project_data[image_path] = {'caption': ""}
            item = QListWidgetItem(self.image_list)
            custom_widget = ImageListItemWidget(image_path, self.on_text_changed, self.display_image_info)
            item.setSizeHint(custom_widget.sizeHint())
            self.image_list.setItemWidget(item, custom_widget)

    def on_text_changed(self, image_path, new_text):
        self.project_data[image_path]['caption'] = new_text

    def display_image_info(self, item):
        widget = self.image_list.itemWidget(item)
        pixmap = QPixmap(widget.image_path).scaled(400, 400, Qt.KeepAspectRatio)
        self.image_display.setPixmap(pixmap)

        self.result_text.setPlainText(self.project_data[widget.image_path]['caption'])

    def process_all_images(self):
        self.image_queue = list(self.project_data.keys())
        self.progress_bar.setMaximum(len(self.image_queue))
        self.process_next_image()

    def process_selected_image(self):
        selected_item = self.image_list.currentItem()
        if selected_item:
            widget = self.image_list.itemWidget(selected_item)
            self.worker.set_image(widget.image_path, self.system_prompt_edit.toPlainText(), self.question_edit.toPlainText())
            self.worker.start()

    def process_next_image(self):
        if not self.image_queue:
            self.update_status("Processing complete!")
            return

        next_image = self.image_queue.pop(0)
        self.worker.set_image(next_image, self.system_prompt_edit.toPlainText(), self.question_edit.toPlainText())
        self.worker.start()

    def on_llm_result(self, image_path, result):
        self.project_data[image_path]['caption'] = result
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            widget = self.image_list.itemWidget(item)
            if widget.image_path == image_path:
                widget.set_text(result)
                break
        self.progress_bar.setValue(self.progress_bar.value() + 1)
        self.process_next_image()

    def update_status(self, message):
        self.statusBar().showMessage(message)

    def export_images(self):
        folder = QFileDialog.getExistingDirectory(self, "Export Images")
        if folder:
            for image_path, data in self.project_data.items():
                image_name = os.path.basename(image_path)
                export_path = os.path.join(folder, image_name)
                with open(os.path.splitext(export_path)[0] + ".txt", "w") as f:
                    f.write(data['caption'])
                if not os.path.exists(export_path):
                    with open(export_path, 'wb') as img_out:
                        with open(image_path, 'rb') as img_in:
                            img_out.write(img_in.read())

    def save_project(self):
        save_file, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "Project Files (*.proj)")
        if save_file:
            with open(save_file, "wb") as f:
                pickle.dump(self.project_data, f)
            self.update_status("Project saved successfully.")

    def load_project(self):
        load_file, _ = QFileDialog.getOpenFileName(self, "Load Project", "", "Project Files (*.proj)")
        if load_file:
            with open(load_file, "rb") as f:
                self.project_data = pickle.load(f)
            self.update_status("Project loaded successfully.")
            self.image_list.clear()
            for image_path in self.project_data:
                self.add_image_to_list(image_path)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VisualLLMPlayground()
    window.show()
    sys.exit(app.exec_())
