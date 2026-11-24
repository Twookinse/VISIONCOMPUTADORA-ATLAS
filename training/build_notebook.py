"""
Genera training/entrenar_modelo_nex.ipynb (notebook de Google Colab).

El notebook incluye una copia de scripts/merge_dataset_classes.py y de
config/class_mapping.json para funcionar solo, sin clonar el repositorio.
Si se modifica alguno de esos archivos, volver a generar el notebook:

    python training/build_notebook.py

(tests/test_class_mapping.py verifica que el notebook este al dia)
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "training" / "entrenar_modelo_nex.ipynb"
MERGE_SCRIPT = ROOT / "scripts" / "merge_dataset_classes.py"
MAPPING = ROOT / "config" / "class_mapping.json"

# Linea base: modelo jarras_nex_v1.pt (38 clases) evaluado sobre el conjunto de
# validacion con las clases fusionadas (ver el PR / historial de la conversacion)
BASELINE = {"mAP50": 0.772, "mAP50-95": 0.633, "Jarra mAP50": 0.929, "Jarra mAP50-95": 0.744}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def build():
    cells = [
        md("""
# Entrenamiento del modelo Nex (v2, clases fusionadas)

Este notebook:
1. Descarga el dataset de Roboflow (`nex-cocina-oficial` v1, formato YOLOv11).
2. **Fusiona las 38 clases en 21** (`Jarra1..Jarra5` -> `Jarra`, `Persona1..6` -> `Persona`, etc.) sin volver a etiquetar.
3. Entrena YOLO11 y lo compara con el modelo actual (`jarras_nex_v1.pt`).
4. Descarga el modelo nuevo como `jarras_nex_v2.pt`.

**Antes de empezar:** menu *Entorno de ejecucion -> Cambiar tipo de entorno de ejecucion -> GPU T4*.

> Generado por `training/build_notebook.py` del repositorio VISIONCOMPUTADORA-ATLAS. No editar a mano.
"""),
        code("""
# Misma version de Ultralytics que la PC del proyecto (requirements.txt)
!pip install -q ultralytics==8.4.176 roboflow
import torch
print("GPU disponible:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
"""),
        md("""
## 1. Descargar el dataset

**Opcion A (recomendada): directo desde Roboflow.** Guarda tu clave de API en los *Secretos* de Colab
(icono de llave a la izquierda) con el nombre `ROBOFLOW_API_KEY` y activa el acceso para este notebook.
Asi la clave **no queda escrita en el notebook**.

**Opcion B:** cambia `USAR_ROBOFLOW = False` y sube el zip exportado de Roboflow.
"""),
        code("""
USAR_ROBOFLOW = True

if USAR_ROBOFLOW:
    from google.colab import userdata
    from roboflow import Roboflow
    rf = Roboflow(api_key=userdata.get("ROBOFLOW_API_KEY"))
    dataset = rf.workspace("alans-workspace-2f18i").project("nex-cocina-oficial").version(1).download("yolov11")
    ORIGEN = dataset.location
else:
    import zipfile
    from google.colab import files
    subido = files.upload()
    ORIGEN = "/content/dataset_roboflow"
    zipfile.ZipFile(next(iter(subido))).extractall(ORIGEN)

print("Dataset en:", ORIGEN)
"""),
        md("""
## 2. Fusionar las clases (38 -> 21)

Las dos celdas siguientes crean los archivos del repositorio (`config/class_mapping.json` y
`scripts/merge_dataset_classes.py`). Si quieres cambiar como se agrupan las clases, edita la seccion
`"fusion"` del JSON.
"""),
        code("%%writefile class_mapping.json\n" + MAPPING.read_text(encoding="utf-8")),
        code("%%writefile merge_dataset_classes.py\n" + MERGE_SCRIPT.read_text(encoding="utf-8")),
        code("""
!rm -rf /content/nex_v2
!python merge_dataset_classes.py --src "{ORIGEN}" --dst /content/nex_v2 --mapping class_mapping.json
"""),
        md("""
## 3. Entrenar

- `PESOS_INICIALES = "yolo11n.pt"` entrena desde el modelo base de Ultralytics (igual que v1).
- Si subes `jarras_nex_v1.pt` a Colab puedes usarlo como punto de partida (`"/content/jarras_nex_v1.pt"`):
  reutiliza lo aprendido de la cocina y suele converger mas rapido.
- Con GPU T4 tarda aproximadamente 15-30 minutos.
"""),
        code("""
from ultralytics import YOLO

PESOS_INICIALES = "yolo11n.pt"
EPOCAS = 150
TAMANO_IMAGEN = 640

modelo = YOLO(PESOS_INICIALES)
modelo.train(
    data="/content/nex_v2/data.yaml",
    epochs=EPOCAS,
    imgsz=TAMANO_IMAGEN,
    batch=16,
    patience=40,
    seed=0,
    project="/content/runs",
    name="nex_v2",
    exist_ok=True,
)
"""),
        md("""
## 4. Comparar con el modelo actual

La linea base es `jarras_nex_v1.pt` evaluado sobre el **mismo** conjunto de validacion, contando sus
38 clases como las 21 fusionadas (asi se comparan en igualdad de condiciones).
"""),
        code(f"""
LINEA_BASE = {json.dumps(BASELINE)}

mejor = YOLO("/content/runs/nex_v2/weights/best.pt")
val = mejor.val(data="/content/nex_v2/data.yaml", split="val", plots=True)
ap50 = dict(zip(val.box.ap_class_index.tolist(), val.box.ap50.tolist()))
ap = dict(zip(val.box.ap_class_index.tolist(), val.box.ap.tolist()))
jarra = list(mejor.names.values()).index("Jarra")

nuevo = {{
    "mAP50": val.box.map50,
    "mAP50-95": val.box.map,
    "Jarra mAP50": ap50.get(jarra, 0.0),
    "Jarra mAP50-95": ap.get(jarra, 0.0),
}}
print(f"{{'Metrica':<16}}{{'v1 (actual)':>12}}{{'v2 (nuevo)':>12}}")
for k in LINEA_BASE:
    print(f"{{k:<16}}{{LINEA_BASE[k]:>12.3f}}{{nuevo[k]:>12.3f}}")
"""),
        code("""
# Evaluacion final sobre el conjunto de prueba (imagenes que el modelo nunca vio)
mejor.val(data="/content/nex_v2/data.yaml", split="test")
"""),
        md("""
## 5. Descargar el modelo nuevo
"""),
        code("""
import shutil
from google.colab import files

shutil.copy("/content/runs/nex_v2/weights/best.pt", "/content/jarras_nex_v2.pt")
files.download("/content/jarras_nex_v2.pt")
"""),
        md("""
## 6. Usarlo en la PC del proyecto

1. Copia `jarras_nex_v2.pt` a `C:\\PROYECTOSIA\\VISIONCOMPUTADORA-ATLAS\\models\\custom\\`.
2. En `config\\cameras.json` cambia `"modelo_default"` a `"models/custom/jarras_nex_v2.pt"`.
3. Reinicia `Iniciar_Nex.bat`. Para volver al modelo anterior, deshaz el cambio del paso 2.

Los nombres de las clases nuevas son los mismos que el sistema ya usa, asi que no hay que cambiar codigo.
"""),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    NOTEBOOK.write_text(build(), encoding="utf-8")
    print(f"Notebook generado: {NOTEBOOK}")
