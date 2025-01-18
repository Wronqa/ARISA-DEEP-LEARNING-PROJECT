import logging

def setup_logger(log_file="app.log"):
    # Tworzenie loggera
    logger = logging.getLogger("MedMnist")
    logger.setLevel(logging.INFO)

    # Tworzenie formatera
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Tworzenie FileHandler do zapisywania logów do pliku
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Tworzenie StreamHandler do wyświetlania logów w konsoli
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Dodanie handlerów do loggera
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

