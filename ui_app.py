import os
import tkinter as tk
from tkinter import messagebox

import cv2

from verify_face import (
    get_face_embedding,
    calculate_similarity
)

from exam_launcher import open_exam


def get_base_dir():

    return os.path.dirname(
        os.path.abspath(__file__)
    )


def start_verification():

    student_id = student_entry.get().strip()

    if not student_id:

        messagebox.showerror(
            "Error",
            "Please enter Student ID."
        )

        return

    base_dir = get_base_dir()

    registered_photo = os.path.join(
        base_dir,
        "registered_faces",
        f"{student_id}.jpg"
    )

    if not os.path.exists(
        registered_photo
    ):

        messagebox.showerror(
            "Error",
            "Registered photo not found."
        )

        return

    start_button.config(
        state="disabled"
    )

    status_label.config(
        text="Status: Loading face model..."
    )

    root.update()

    registered_image = cv2.imread(
        registered_photo
    )

    registered_embedding = get_face_embedding(
        registered_image
    )

    if registered_embedding is None:

        messagebox.showerror(
            "Error",
            "No face found in registered photo."
        )

        start_button.config(
            state="normal"
        )

        return

    status_label.config(
        text="Status: Opening camera..."
    )

    root.update()

    camera = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():

        messagebox.showerror(
            "Error",
            "Camera cannot open."
        )

        start_button.config(
            state="normal"
        )

        return

    match_count = 0

    for i in range(5):

        status_label.config(
            text=f"Status: Checking {i + 1}/5"
        )

        root.update()

        for _ in range(10):

            camera.read()

        success, frame = camera.read()

        if not success:

            continue

        live_embedding = get_face_embedding(
            frame
        )

        if live_embedding is None:

            continue

        similarity = calculate_similarity(
            registered_embedding,
            live_embedding
        )

        if similarity >= 0.45:

            match_count += 1

    camera.release()

    if match_count >= 3:

        status_label.config(
            text=f"Status: Access Granted ({match_count}/5)"
        )

        messagebox.showinfo(
            "Access Granted",
            f"Face matched.\n\n"
            f"Matches: {match_count}/5"
        )

        open_exam()

    else:

        status_label.config(
            text=f"Status: Access Denied ({match_count}/5)"
        )

        messagebox.showerror(
            "Access Denied",
            f"Face did not match.\n\n"
            f"Matches: {match_count}/5"
        )

    start_button.config(
        state="normal"
    )


def clear_student_id():

    student_entry.delete(
        0,
        tk.END
    )

    status_label.config(
        text="Status: Waiting for Student ID"
    )


root = tk.Tk()

root.title(
    "Exam Face Authentication"
)

root.geometry(
    "500x350"
)

root.resizable(
    False,
    False
)


title_label = tk.Label(
    root,
    text="EXAM FACE AUTHENTICATION",
    font=("Arial", 18, "bold")
)

title_label.pack(
    pady=30
)


student_id_label = tk.Label(
    root,
    text="Student ID",
    font=("Arial", 12)
)

student_id_label.pack()


student_entry = tk.Entry(
    root,
    font=("Arial", 14),
    justify="center"
)

student_entry.pack(
    padx=80,
    pady=10,
    fill="x"
)


start_button = tk.Button(
    root,
    text="Start Verification",
    font=("Arial", 11, "bold"),
    command=start_verification
)

start_button.pack(
    padx=80,
    pady=10,
    fill="x"
)


clear_button = tk.Button(
    root,
    text="Clear",
    command=clear_student_id
)

clear_button.pack(
    padx=80,
    pady=5,
    fill="x"
)


status_label = tk.Label(
    root,
    text="Status: Waiting for Student ID"
)

status_label.pack(
    pady=20
)


student_entry.focus()

root.mainloop()