from transformers import pipeline

# Crear el pipeline de clasificación
clasificador = pipeline(
    "text-classification",
    model="polodealvarado/distilbert-review_classification",
    tokenizer="polodealvarado/distilbert-review_classification",
    top_k=1,  # Solo la clase más probable
)

# Texto de entrada
texto = "Este producto superó mis expectativas, lo recomiendo totalmente."

# Realizar predicción
output = clasificador(texto)

# Extraer la clase predicha (por ejemplo, 'LABEL_0', 'LABEL_1', ...)
etiqueta = output[0][0]["label"]
indice = int(etiqueta.replace("LABEL_", ""))  # 'LABEL_0' → 0
estrellas_predichas = indice + 1

print(f"Predicción: {estrellas_predichas} estrellas")
