import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import logging
import traceback

from app_run_detector import run_detector_plot_result
from lidar_det.dataset import load_pcb


app = FastAPI()


@app.get("/")
def hello_world():
    return {"Hello": "world"}


@app.post("/detect")
def get_detection(file: UploadFile = File(), score_threshold: float = 0.5):
    # validate file
    filename = file.filename
    file_extension = filename.split(".")[-1] in ("pcd")
    if not file_extension:
        raise HTTPException(
            status_code=415, detail="Unsupported file provided. Only supports pcd now.")

    pc_stream = io.BytesIO(file.file.read())
    pc_stream.seek(0)

    try:
        pc = load_pcb(pc_stream)

        result_img = run_detector_plot_result(pc, score_threshold)

        memory_stream = io.BytesIO()
        result_img.save(memory_stream, format="PNG")
        memory_stream.seek(0)

        return StreamingResponse(memory_stream, media_type="image/png")
    except Exception as e:
        # Log full traceback to help debugging and return 500 with message
        logging.exception("Error in /detect endpoint")
        tb = traceback.format_exc()
        # Return a clear HTTP error with brief info (avoid huge trace in response)
        raise HTTPException(status_code=500, detail=f"Server error during detection: {str(e)}\nSee server logs for traceback.")
