# Bazowy obraz TensorFlow
FROM tensorflow/tensorflow

# Ustawienie katalogu roboczego
WORKDIR /app

# Skopiowanie pliku requirements.txt
COPY requirements.txt .

# Instalacja zależności
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# Skopiowanie pozostałych plików aplikacji
COPY . .

# Domyślne polecenie kontenera
CMD ["sleep", "infinity"]