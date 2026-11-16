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
    """Zbierz pomiary z kamery, pokazując na żywo wykryty szkielet dłoni."""
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Nie można otworzyć kamery.")

    measurements = []
    try:
        with mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
        ) as detector:
            captured = 0
            print("Podgląd kamery: naciśnij Q, aby anulować.")
            while captured < count:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Nie udało się odczytać obrazu z kamery.")

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.process(rgb)
                features = None
                if result.multi_hand_landmarks:
                    landmarks = result.multi_hand_landmarks[0]
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS
                    )
                    height, width = frame.shape[:2]
                    points = np.array([
                        (point.x * width, point.y * height)
                        for point in landmarks.landmark
                    ])
                    palm_size = np.linalg.norm(points[9] - points[0])
                    if palm_size >= 20:
                        distances = np.linalg.norm(points[1:] - points[0], axis=1) / palm_size
                        features = np.delete(distances, 8)

                cv2.putText(
                    frame, f"Ujecia: {captured}/{count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                )
                cv2.imshow("Biometria dloni - Q: anuluj", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise RuntimeError("Skanowanie anulowane.")

                if features is not None:
                    measurements.append(features)
                    captured += 1
                    print(f"{captured}/{count}: OK")
                    if captured < count:
                        time.sleep(0.5)
    finally:
        camera.release()
        cv2.destroyAllWindows()

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

def delete_user(name):
    """Usuń wzorzec użytkownika z bazy."""
    with sqlite3.connect(DATABASE) as db:
        create_table(db)
        cursor = db.execute("DELETE FROM users WHERE name = ?", (name,))
        return cursor.rowcount > 0

def manage_database():
    """Wyświetl zapisane osoby i pozwól usunąć wybraną."""
    while True:
        users = list_users()
        clear_screen()
        print("=== BAZA DANYCH ===")
        if not users:
            print("Baza jest pusta.")
        else:
            for index, (name, _, _) in enumerate(users, start=1):
                print(f"{index}. {name}")
        print("\nU. Usuń osobę")
        print("3. Wstecz")
        choice = menu_choice({"u", "U", "3"})
        if choice == "3":
            return
        if not users:
            show_result("Nie ma osób do usunięcia.")
            continue

        number = menu_choice({str(i) for i in range(1, len(users) + 1)})
        name = users[int(number) - 1][0]
        confirm = input(f"Usunąć użytkownika {name}? (t/n): ").strip().lower()
        if confirm == "t":
            delete_user(name)
            show_result(f"Usunięto użytkownika: {name}")
        else:
            show_result("Anulowano usuwanie.")

def match_score(samples, mean, std):
    """Porównaj próbkę z uwzględnieniem rozrzutu próbki i wzorca."""
    sample_mean = samples.mean(axis=0)
    sample_std = samples.std(axis=0)
    combined_std = np.maximum(np.sqrt(std**2 + sample_std**2), 0.05)
    return float(np.abs((sample_mean - mean) / combined_std).mean())

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
        print("- Kamera pobierze 30 ujęć w 10 sekund.")
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
            samples = capture(30, 10)
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
        scores = sorted(
            (match_score(samples, mean, std), name)
            for name, mean, std in users
        )
        score, name = scores[0]
        ranking = "\n".join(f"{candidate}: {candidate_score:.2f}" for candidate_score, candidate in scores)
        status = "DOPASOWANO" if score < THRESHOLD else "BRAK ZGODNOŚCI"
        show_result(f"Najbliższa osoba: {name}\nWyniki:\n{ranking}\n{status}")
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
        print("3. Baza danych")
        print("4. Wyjście")

        match menu_choice({"1", "2", "3", "4"}):
            case "1":
                registration()
            case "2":
                recognition()
            case "3":
                manage_database()
            case "4":
                clear_screen()
                return

if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nWyjście.")
