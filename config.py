# Настройки эксперимента
VARIANT = 1              # номер варианта (1..24) — выберите свой
LAGS = 5                 # сколько предыдущих шагов подаём как признаки (лаги)
TEST_SIZE = 0.2
VAL_SIZE = 0.2
RANDOM_STATE = 42
STEP = 1                 # шаг прогноза для одношаговой модели
MULTI_HORIZON = 7        # для многошагового прогноза

DATASET_PATH = "data/Lab3_Dataset.xlsx"
RESULTS_PATH = "outputs/results_Comparison.xlsx"
FIG_DIR = "outputs/figures"