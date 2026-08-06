import cv2
import numpy as np
from insightface.app import FaceAnalysis


face_app = None


def load_face_model():
    global face_app

    if face_app is None:

        face_app = FaceAnalysis(
            name="buffalo_s",
            providers=["CPUExecutionProvider"]
        )

        face_app.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )


def get_face_embedding(image):

    load_face_model()

    faces = face_app.get(image)

    if len(faces) == 0:
        return None

    return faces[0].embedding


def calculate_similarity(
    embedding1,
    embedding2
):

    embedding1 = embedding1 / np.linalg.norm(
        embedding1
    )

    embedding2 = embedding2 / np.linalg.norm(
        embedding2
    )

    return np.dot(
        embedding1,
        embedding2
    )