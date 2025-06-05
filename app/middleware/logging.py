import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#TODO: Organize

# Criar diretório de logs se não existir
# log_dir = "C:\\captacao\\logs"
# os.makedirs(log_dir, exist_ok=True)  # Garante que a pasta exista
#
# # Configurar logger
# logger = logging.getLogger("imageProcessorLogger")
# logger.setLevel(logging.DEBUG)  # Para permitir INFO e ERROR
#
# # Criar handler para ERROR
# error_handler = logging.FileHandler(os.path.join(log_dir, "imageProcessor_ERROR_log.txt"))
# error_handler.setLevel(logging.ERROR)
# error_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
# error_handler.setFormatter(error_formatter)
#
# # Criar handler para INFO
# info_handler = logging.FileHandler(os.path.join(log_dir, "imageProcessor_INFO_log.txt"))
# info_handler.setLevel(logging.INFO)
# info_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
# info_handler.setFormatter(info_formatter)
#
# # Adicionar handlers ao logger
# logger.addHandler(error_handler)
# logger.addHandler(info_handler)

#TODO: Organize