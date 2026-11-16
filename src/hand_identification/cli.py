import json
import sqlite3
import time

import cv2
import mediapipe as mp
import numpy as np

DATABASE = "hands.db"
THRESHOLD = 2.0

def hand_features(frame, detector):
    """19 proporcji dłoni z punktów MediaPipe (niezależnych od rozmiaru zdjęcia)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.process(rgb)

    if not result.multi_hand_landmarks:
        return None

    height, width = frame.shape[:2]
    points = np.array([
        (point.x * width, point.y * height)
        for point in result.multi_hand_landmarks[0].landmark
    ])

    palm_size = np.linalg.norm(points[9] - points[0])
    if palm_size < 20:
        return None

    distances = np.linalg.norm(points[1:] - points[0], axis=1) / palm_size
    return np.delete(distances, 8)  # Punkt 9 służy już jako skala.

def capture(count, seconds):
    """Zbierz pomiary z kamery w równych odstępach czasu."""
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Nie można otworzyć kamery.")

    measurements = []
    try:
        print("Umieść całą dłoń przed kamerą. Start za 3 sekundy...")
        time.sleep(3)

        with mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            model_complexity=0,
        ) as detector:
            start = time.monotonic()
            for i in range(count):
                time.sleep(max(0, start + i * seconds / count - time.monotonic()))
                ok, frame = camera.read()
                features = hand_features(frame, detector) if ok else None

                if features is not None:
                    measurements.append(features)
                print(f"{i + 1}/{count}: {'OK' if features is not None else 'nie wykryto dłoni'}")
    finally:
        camera.release()

    if len(measurements) < count * 0.75:
        raise RuntimeError("Za mało poprawnych pomiarów. Spróbuj ponownie.")
    return np.array(measurements)

def create_table(db):
    db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            name TEXT PRIMARY KEY,
            mean TEXT NOT NULL,
            std TEXT NOT NULL
        )
    """)

def save_template(name, mean, std):
    """Zapisz wzorzec użytkownika, bez przechowywania zdjęć."""
    with sqlite3.connect(DATABASE) as db:
        create_table(db)
        db.execute(
            "INSERT INTO users VALUES (?, ?, ?)",
            (name, json.dumps(mean.tolist()), json.dumps(std.tolist())),
        )

def list_users():
    """Pobierz użytkowników i ich wzorce z bazy danych."""
    with sqlite3.connect(DATABASE) as db:
        create_table(db)
        rows = db.execute("SELECT name, mean, std FROM users ORDER BY name").fetchall()

    return [
        (name, np.array(json.loads(mean)), np.array(json.loads(std)))
        for name, mean, std in rows
    ]

def match_score(samples, mean, std):
    """Średnia bezwzględna wartość Z-score: mniej znaczy bardziej podobna dłoń."""
    return float(np.abs((samples - mean) / std).mean())

def clear_screen():
    """Wyczyść konsolę (terminal ANSI)."""
    print("\033[2J\033[H", end="")

def menu_choice(allowed):
    while True:
        choice = input("Wybór: ").strip()
        if choice in allowed:
            return choice
        print("Nieprawidłowy wybór.")

def show_result(message):
    clear_screen()
    print(message)
    print("\n3. Wstecz")
    menu_choice({"3"})

def registration():
    while True:
        clear_screen()
        print("=== REJESTRACJA ===")
        print("- Pokaż całą dłoń od strony wewnętrznej.")
        print("- Rozsuń lekko palce i utrzymuj podobną pozycję.")
        print("- Kamera pobierze 20 pomiarów w 10 sekund.")
        print("\n1. Rozpocznij rejestrację")
        print("3. Wstecz")

        if menu_choice({"1", "3"}) == "3":
            return

        clear_screen()
        print("=== NAZWA UŻYTKOWNIKA ===")
        print("3. Wstecz")
        name = input("Login / ID: ").strip()
        if name == "3":
            continue
        if not name:
            show_result("Login nie może być pusty.")
            continue
        if any(user[0] == name for user in list_users()):
            show_result("Taki login już istnieje w bazie.")
            continue

        clear_screen()
        try:
            samples = capture(20, 10)
            mean = samples.mean(axis=0)
            std = np.maximum(samples.std(axis=0), 0.05)
            save_template(name, mean, std)
            show_result(f"Zarejestrowano użytkownika: {name}")
        except (RuntimeError, sqlite3.IntegrityError) as error:
            show_result(f"Błąd rejestracji: {error}")
        return

def verification():
    users = list_users()
    clear_screen()
    if not users:
        show_result("Brak użytkowników. Najpierw wykonaj rejestrację.")
        return

    print("=== WERYFIKACJA ===")
    print("Wybierz osobę, za którą się podajesz:")
    # Pomijamy numer 3, ponieważ oznacza on zawsze powrót.
    choices = {
        str(i if i < 3 else i + 1): user
        for i, user in enumerate(users, start=1)
    }
    for number, (name, _, _) in choices.items():
        print(f"{number}. {name}")
    print("3. Wstecz")

    choice = menu_choice(set(choices) | {"3"})
    if choice == "3":
        return

    name, mean, std = choices[choice]
    clear_screen()
    try:
        samples = capture(3, 1.5)
        score = match_score(samples, mean, std)
        status = "DOPASOWANO" if score < THRESHOLD else "BRAK ZGODNOŚCI"
        show_result(f"Weryfikacja: {name}\nZ-score: {score:.2f}\n{status}")
    except RuntimeError as error:
        show_result(f"Błąd weryfikacji: {error}")

def identification():
    users = list_users()
    clear_screen()
    if not users:
        show_result("Brak użytkowników. Najpierw wykonaj rejestrację.")
        return

    print("=== IDENTYFIKACJA ===")
    print("Program porówna Twoją dłoń ze wszystkimi zapisanymi osobami.")
    print("\n1. Rozpocznij skanowanie")
    print("3. Wstecz")
    if menu_choice({"1", "3"}) == "3":
        return

    clear_screen()
    try:
        samples = capture(3, 1.5)
        score, name = min(
            (match_score(samples, mean, std), name)
            for name, mean, std in users
        )
        status = "DOPASOWANO" if score < THRESHOLD else "BRAK ZGODNOŚCI"
        show_result(f"Najbliższa osoba: {name}\nZ-score: {score:.2f}\n{status}")
    except RuntimeError as error:
        show_result(f"Błąd identyfikacji: {error}")

def recognition():
    while True:
        clear_screen()
        print("=== ROZPOZNAWANIE ===")
        print("1. Weryfikacja (wskazujesz użytkownika)")
        print("2. Identyfikacja (najbliższy użytkownik)")
        print("3. Wstecz")

        match menu_choice({"1", "2", "3"}):
            case "1":
                verification()
            case "2":
                identification()
            case "3":
                return

def main():
    while True:
        clear_screen()
        print("=== BIOMETRIA DŁONI ===")
        print("1. Rejestracja")
        print("2. Rozpoznawanie")
        print("3. Wyjście")

        match menu_choice({"1", "2", "3"}):
            case "1":
                registration()
            case "2":
                recognition()
            case "3":
                clear_screen()
                return

if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nWyjście.")
