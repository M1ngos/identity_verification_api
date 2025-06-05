import base64
from PIL import Image
import cv2
import io
import numpy as np
import traceback
import tempfile
import logging
import os

# Criar diretório de logs se não existir
log_dir = "C:\\captacao\\logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configurar logger
logger = logging.getLogger("imageHelperLogger")
logger.setLevel(logging.DEBUG)  # Para permitir INFO e ERROR

# Criar handler para ERROR
error_handler = logging.FileHandler(os.path.join(log_dir, "imageHelper_ERROR_log.txt"))
error_handler.setLevel(logging.ERROR)
error_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
error_handler.setFormatter(error_formatter)

# Criar handler para INFO
info_handler = logging.FileHandler(os.path.join(log_dir, "imageHelper_INFO_log.txt"))
info_handler.setLevel(logging.INFO)
info_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
info_handler.setFormatter(info_formatter)

# Adicionar handlers ao logger
logger.addHandler(error_handler)
logger.addHandler(info_handler)

def base64_to_image(base64_str: str) -> np.ndarray:
    """Convert base64 string to OpenCV image."""
    try:
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))
        cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        logger.debug(f"Image converted successfully: shape={cv_image.shape}")
        print(f"Image converted successfully: shape={cv_image.shape}")
        return cv_image
    except Exception as e:
        logger.error(f"Error converting base64 to image: {str(e)}", exc_info=True)
        print(f"Error converting base64 to image: {str(e)}")

def save_temp_image(image) -> str:
    """Salva uma imagem temporariamente e retorna o caminho do arquivo."""
    try:
        logger.info("Salvando temporariamente imagem recebida")

        # Se a imagem for um array NumPy, converte para PIL.Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)  
        image.save(temp_file, format="JPEG")
        temp_file.close()  # Fechar para garantir acesso ao caminho

        logger.info("Ficheiro temporário da imagem criado com sucesso!")
        print(f"CAMINHO: {temp_file.name}")
        return temp_file.name  

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Erro na criação do ficheiro temporário da imagem:\n{error_trace}")
        return {"Erro": str(e), "StackTrace": error_trace}