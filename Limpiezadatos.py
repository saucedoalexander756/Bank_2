import pandas as pd

# ====== Cargar CSV completo ======
df = pd.read_csv("bank_sample.csv", sep=';')

# ====== Tomar 30k filas aleatorias ======
df_sample = df.sample(n=30000, random_state=42)

# ====== Guardar temporalmente para entrenar ======
df_sample.to_csv("bank_sample30.csv", sep=';', index=False)

print("✅ CSV reducido a 30k filas listo para entrenamiento!")
