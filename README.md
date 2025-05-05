# Automatic-Battery-Gen-AI-Based-Models-
Automatic Battery Gen AI Based Models 

Separar los modelos en un archivo aparte permitirá que los puedas actualizar fácilmente sin tocar la estructura principal del código.

Aquí te muestro cómo puedes hacerlo:

### **Paso 1: Crear un archivo externo**
Llamémoslo `modelos.py`, donde almacenaremos la base de datos de modelos.

```python
# modelos.py
MODELOS_HUGGINGFACE = {
    ("image", "text"): [
        "openai/clip-text", "facebook/detr-resnet-50", "microsoft/beit-base-patch16-224",
        "google/vit-large-patch16-224", "deepmind/image-caption-transformer"
    ],
    ("text", "image"): [
        "runwayml/stable-diffusion-v1-5", "openai/dall-e", "google/imagen",
        "stabilityai/stable-diffusion-2", "huggingface/diffusion-xl"
    ],
    ("image", "3d"): [
        "nvidia/image-to-3d", "researchlab/depth-estimation", "openai/image-3d-transformer",
        "ai3d/depth-to-3d", "microsoft/image-voxelization"
    ],
    ("text", "3d"): [
        "ai3d/text-to-3d-shape", "deepmind/text2mesh", "researchlab/text-3d-generator",
        "openai/text-to-3d-object", "nvidia/text-to-3d-model"
    ],
}
```

### **Paso 2: Modificar el código principal para importar modelos**
Ahora, en tu código principal, simplemente importamos los modelos desde `modelos.py`.

```python
from transformers import pipeline
import random
from modelos import MODELOS_HUGGINGFACE  # Importar modelos desde el archivo externo

class ModeloBatch:
    def __init__(self, input_type, output_type, num_models, responses_per_model):
        self.input_type = input_type
        self.output_type = output_type
        self.num_models = num_models
        self.responses_per_model = responses_per_model
        self.available_models = self._search_models()
        self.selected_models = self._select_models()

    def _search_models(self):
        return MODELOS_HUGGINGFACE.get((self.input_type, self.output_type), [])

    def _select_models(self):
        if not self.available_models:
            raise ValueError("No se encontraron modelos adecuados para la conversión.")

        selected = random.sample(self.available_models, min(self.num_models, len(self.available_models)))
        return selected

    def process(self, input_data):
        results = {}
        for model_name in self.selected_models:
            model = pipeline(self.input_type + "-to-" + self.output_type, model=model_name)
            responses = [model(input_data) for _ in range(self.responses_per_model)]
            results[model_name] = responses
        
        print(f"Modelos utilizados: {', '.join(self.selected_models)}")
        return results

# Uso de la librería con modelos separados en un archivo
batch_processor = ModeloBatch("image", "3d", 3, 2)
resultados = batch_processor.process("ruta/a/imagen.jpg")
print(resultados)
```

### **Beneficios de este enfoque**
✅ **Facilidad de actualización**: Puedes añadir más modelos en `modelos.py` sin tocar el código principal.  
✅ **Mantenimiento simple**: Mantienes una estructura clara y organizada.  
✅ **Escalabilidad**: Si en el futuro quieres ampliar la base de datos, simplemente modificas `modelos.py`.  



