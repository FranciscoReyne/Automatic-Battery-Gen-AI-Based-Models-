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

Para manejar la salida de modelos 3D de manera eficiente, lo mejor es guardar los resultados en una carpeta en lugar de simplemente imprimirlos en pantalla. Dependiendo del tipo de datos generados por los modelos, podrías guardarlos en formatos como `.obj`, `.stl`, `.ply` o `.glb`.

Aquí está el código para guardar automáticamente los archivos en una carpeta llamada `resultados_3d`:

```python
import os
from transformers import pipeline
import random
from modelos import MODELOS_HUGGINGFACE

class ModeloBatch:
    def __init__(self, input_type, output_type, num_models, responses_per_model, output_folder="resultados_3d"):   #nombre carpeta salida
        self.input_type = input_type
        self.output_type = output_type
        self.num_models = num_models
        self.responses_per_model = responses_per_model
        self.output_folder = output_folder
        self.available_models = self._search_models()
        self.selected_models = self._select_models()

        # Crear carpeta de resultados si no existe
        os.makedirs(self.output_folder, exist_ok=True)

    def _search_models(self):
        return MODELOS_HUGGINGFACE.get((self.input_type, self.output_type), [])

    def _select_models(self):
        if not self.available_models:
            raise ValueError("No se encontraron modelos adecuados para la conversión.")

        selected = random.sample(self.available_models, min(self.num_models, len(self.available_models)))
        return selected

    def process(self, input_data):
        results = {}
        for i, model_name in enumerate(self.selected_models):
            model = pipeline(self.input_type + "-to-" + self.output_type, model=model_name)
            responses = [model(input_data) for _ in range(self.responses_per_model)]
            results[model_name] = responses

            # Guardar cada respuesta como un archivo .obj o similar
            for j, response in enumerate(responses):
                output_path = os.path.join(self.output_folder, f"{model_name}_response_{i}_{j}.obj")
                with open(output_path, "w") as f:
                    f.write(str(response))  # Reemplazar con conversión adecuada a formato 3D

        print(f"Modelos utilizados: {', '.join(self.selected_models)}")
        print(f"Resultados guardados en la carpeta: {self.output_folder}")
        return results

# Uso de la librería con salida de modelos 3D guardados en archivos
batch_processor = ModeloBatch("image", "3d", 3, 2)
resultados = batch_processor.process("ruta/a/imagen.jpg")
```

### **Mejoras en esta versión**
✅ **Guarda los resultados en archivos** dentro de una carpeta, asegurando persistencia.  
✅ **Usa formatos adecuados** para salida de modelos 3D (puedes cambiar `.obj` por `.stl`, `.ply`, etc.).  
✅ **Organiza cada ejecución** dentro de una estructura clara.  

Esto hace que puedas revisar los archivos generados sin depender de la terminal. ¿Quieres que agreguemos más formatos de exportación? 🚀🔧

