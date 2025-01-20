optymized_params = {
    "learning_rate_range": (1e-6, 1e-1),  # Rozszerzony zakres do testowania ekstremalnych wartości
    "batch_size_options": [64, 128, 256, 512, 1024],  # Dodano więcej opcji batch size
    "dropout_rate_range": (0.0, 0.7),  # Poszerzony zakres, w tym brak dropout
    "filters_0_options": [16, 32, 64],  # Dodano mniejsze wartości dla eksperymentów z prostszymi modelami
    "filters_1_options": [8, 16, 32, 64],  # Dodano mniejsze wartości dla eksperymentów z prostszymi modelami
    "filters_2_options": [32, 64, 128, 256, 512],  # Dodano większe wartości dla bardziej złożonych modeli
    "filters_3_options": [128, 256, 512],  # Dodano dla modeli z trzecią warstwą Conv2D
    "filters_4_options": [128, 256, 512, 1024],  # Dodano dla modeli z czwartą warstwą Conv2D
    "dense_units_options": [32, 64, 128, 256, 512],  # Poszerzony zakres dla gęstych warstw
    "activation_functions": ["relu", "tanh", "elu", "swish"],  # Opcje różnych funkcji aktywacji
    "optimizer_options": ["adam", "sgd", "rmsprop", "adamax"],  # Różne optymalizatory do testowania
    "epochs_range": (50, 100, 200),  # Zakres liczby epok do trenowania
    "early_stopping_patience": [5, 10],  # Dłuższa cierpliwość dla wczesnego zatrzymywania
     "l2_reg_range": (1e-6, 1e-2),  # Zakres regularizacji L2 dla optymalizacji
     "dropout_dense": (0.2, 0.7)  # Zakres dropout dla warstw gęstych
}