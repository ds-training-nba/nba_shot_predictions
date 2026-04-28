from app.modeling import evaluate_model, MODEL_ID_RANDOM_FOREST

cm, cr = evaluate_model(MODEL_ID_RANDOM_FOREST)
print(cm, cr)