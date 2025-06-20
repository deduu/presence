# app/services/image_pipeline.py
import logging, numpy as np, pathlib, datetime as dt
from typing import List
from app.utils.file_store import BASE
from app.db.base import create_session
from app.services.face_service import FaceService
from app.db.models.attendance import ImageCount, ImageRecord
from image_processor import ImageProcessor
from config import FACE_DISTANCE_THRESHOLD

logger = logging.getLogger(__name__)

class ImagePipeline:
    """High-level batch processor used by the /uploads route."""

    def __init__(self):
        self.processor = ImageProcessor(str(BASE / "raw"))

    async def process_batch(self, paths: List[str]):
        async with create_session() as session:
            face_svc = FaceService(session)

            known_ids, known_encs = await face_svc.get_all_known_faces()

            for path in paths:
                res = self.processor.process_image(path)
                if not res: continue

                det_time   = res["detection_time"]
                encodings  = res["face_encodings"]

                # Record image-level count
                session.add(ImageCount(
                    image_path   = path,
                    face_count   = len(encodings),
                    processed_time = det_time
                ))

                for enc in encodings:
                    enc_np = np.asarray(enc)
                    face_id = None

                    if known_encs:
                        dists = np.linalg.norm(np.stack(known_encs) - enc_np, axis=1)
                        min_d = dists.min()
                        if min_d < FACE_DISTANCE_THRESHOLD:
                            idx = int(dists.argmin())
                            face_id = known_ids[idx]
                            await face_svc.update_last_seen(face_id, det_time, enc_np)
                        else:
                            new_face = await face_svc.insert_new_face(enc_np, det_time)
                            known_ids.append(new_face.face_id)
                            known_encs.append(enc_np)
                            face_id = new_face.face_id
                    else:
                        new_face = await face_svc.insert_new_face(enc_np, det_time)
                        known_ids.append(new_face.face_id)
                        known_encs.append(enc_np)
                        face_id = new_face.face_id

                    session.add(ImageRecord(
                        image_path     = path,
                        face_id        = face_id,
                        detection_time = det_time
                    ))

            await session.commit()
            logger.info("Batch processed %d images", len(paths))
