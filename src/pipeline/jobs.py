"""
Analisis de videos en segundo plano (barra de progreso del Dashboard).

Solo se analiza UN video a la vez: YOLO ocupa toda la CPU/GPU y dos analisis en
paralelo tardarian lo mismo que uno tras otro. Pedir otro mientras hay uno en
curso lanza JobBusyError (HTTP 409).

Cada analisis queda en un registro propio (AnalysisJob) para que quien lo inicio
pueda esperar SU resultado, aunque despues se inicie otro.
"""

import itertools
import threading
from datetime import datetime

import cv2

from src.detection.tracker import Tracker
from src.pipeline.video_analysis import AnalysisProgress, analyze_video, save_report
from src.utils.security import mask_text


class JobBusyError(RuntimeError):
    """Ya hay un video analizandose."""


class AnalysisJob:
    """Estado de un analisis en curso o terminado."""

    _ids = itertools.count(1)

    def __init__(self, request):
        self.id = next(self._ids)
        self.request = request
        self.started_at = datetime.now()
        self.progress = AnalysisProgress()
        self.cancel_event = threading.Event()
        self.done = threading.Event()
        self.report = None
        self.run_id = None
        self.report_path = None
        self.error = None

    @property
    def running(self):
        return not self.done.is_set()


class VideoAnalysisJobs:
    """Ejecuta analisis de video en un hilo, de a uno.

    Args:
        snapshots_dir: Carpeta de las fotos de evidencia.
        output_dir: Carpeta de los reportes JSON.
        tracker_factory / capture_factory / save_fn: reemplazables en las pruebas.
    """

    def __init__(self, snapshots_dir, output_dir, tracker_factory=Tracker,
                 capture_factory=cv2.VideoCapture, save_fn=save_report):
        self.snapshots_dir = snapshots_dir
        self.output_dir = output_dir
        self.tracker_factory = tracker_factory
        self.capture_factory = capture_factory
        self.save_fn = save_fn
        self._lock = threading.Lock()
        self._job = None

    @property
    def current(self):
        return self._job

    @property
    def running(self):
        job = self._job
        return bool(job and job.running)

    def start(self, request):
        """Inicia el analisis en segundo plano y devuelve su AnalysisJob."""
        with self._lock:
            if self.running:
                raise JobBusyError("Ya hay un video analizandose en este momento")
            job = AnalysisJob(request)
            self._job = job
        threading.Thread(target=self._run, args=(job,), daemon=True,
                         name=f"nex-analisis-{job.id}").start()
        return job

    def cancel(self):
        job = self._job
        if job:
            job.cancel_event.set()

    def status(self):
        job = self._job
        if job is None:
            return {
                "activo": False, "camera_id": "", "fuente": "", "progreso_pct": 0.0,
                "progreso_conocido": False, "frames_procesados": 0, "total_frames": 0,
                "eventos_encontrados": 0, "jarras_encontradas": 0, "ultimo_reporte": None,
                "error": None,
            }
        p = job.progress
        pct = 100.0 if (job.report and not job.running) else (p.percent or 0.0)
        return {
            "activo": job.running,
            "camera_id": job.request.camera_id,
            "fuente": job.request.public_source,
            "progreso_pct": round(pct, 1),
            "progreso_conocido": p.total_frames is not None,
            "frames_procesados": p.frames_read,
            "total_frames": p.total_frames or 0,
            "eventos_encontrados": p.events,
            "jarras_encontradas": p.jars_lifted,
            "ultimo_reporte": job.report,
            "error": job.error,
        }

    def _run(self, job):
        try:
            def on_progress(progress):
                job.progress = progress

            report = analyze_video(
                job.request,
                snapshots_dir=self.snapshots_dir,
                tracker_factory=self.tracker_factory,
                capture_factory=self.capture_factory,
                on_progress=on_progress,
                should_cancel=job.cancel_event.is_set,
            )
            job.report_path, job.run_id = self.save_fn(report, self.output_dir, job.request.report_name)
            job.report = report
        except Exception as exc:  # noqa: BLE001 - se informa en el Dashboard
            job.error = mask_text(str(exc), job.request.source)
        finally:
            job.done.set()
