# --------------------------- Homework_8 ---------------------------------
"""
Виконав: Віктор Нікоряк
Homework_8, Рівень складності: I
Умови: Реалізація методів класифікації / ідентифікації об’єктів на цифровому зображенні.

Мета:
Порахувати кількість об’єктів на зображенні з батиметричною інформацією. Виконати обробку TIFF-зображення,
перетворити його у grayscale, створити негатив, застосувати згладжувальні фільтри,
знайти кути методом Харріса і контури методом Кенні. Візуалізувати результати,
підрахувати кількість кутів та контурів.

Основні етапи:
1. Завантаження TIFF-зображення
2. Перетворення у grayscale та негатив
3. Застосування згладжувальних фільтрів: Gaussian, median, bilateral
4. Виявлення об’єктів методом Harris Corner Detector
5. Виявлення контурів методом Canny Edge Detection
6. Підрахунок кількості кутів та контурів
7. Візуалізація результатів
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# === 1. Завантаження та обробка зображення ===
def load_and_prepare_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_neg = 255 - gray
    blurred = cv2.GaussianBlur(gray_neg, (5, 5), sigmaX=1)
    return img, gray, gray_neg, blurred

# === 2. Harris Corner Detector ===
def detect_harris_corners(gray_image, block_size=2, ksize=3, k=0.04):
    gray_float = np.float32(gray_image)
    dst = cv2.cornerHarris(gray_float, block_size, ksize, k)
    dst_dilated = cv2.dilate(dst, None)
    threshold = 0.01 * dst_dilated.max()
    corners = dst_dilated > threshold
    num_corners = np.sum(corners)
    output = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    output[corners] = [0, 0, 255]
    return output, num_corners

# === 3. Canny + Contour Detector ===
def detect_canny_contours(gray_base_image, blurred_image):
    edges = cv2.Canny(blurred_image, threshold1=50, threshold2=150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)
    contour_output = cv2.cvtColor(gray_base_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_output, contours, -1, (0, 255, 0), 1)
    return edges, contour_output, num_contours

# === 4. Візуалізація результатів ===
def visualize_results(harris_img=None, num_corners=None,
                      edges=None, contour_img=None, num_contours=None):
    cols = sum(x is not None for x in [harris_img, edges, contour_img])
    fig, axs = plt.subplots(1, cols, figsize=(6 * cols, 6))
    axs = axs if isinstance(axs, np.ndarray) else [axs]
    i = 0
    if harris_img is not None:
        axs[i].imshow(harris_img)
        axs[i].set_title(f"Harris (кути): {num_corners}")
        axs[i].axis("off")
        i += 1
    if edges is not None:
        axs[i].imshow(edges, cmap='gray')
        axs[i].set_title("Canny edges")
        axs[i].axis("off")
        i += 1
    if contour_img is not None:
        axs[i].imshow(contour_img)
        axs[i].set_title(f"Контури (об'єкти): {num_contours}")
        axs[i].axis("off")
    plt.tight_layout()
    plt.show()

# ----------------------------------- БЛОК ЗАПУСКУ -----------------------------------
if __name__ == '__main__':
    while True:
        print("\nОберіть режим виконання:")
        print("1 - Harris Corner Detection")
        print("2 - Canny + Contour Detection")
        print("3 - Обидва методи з порівнянням")
        print("0 - Вийти")
        choice = input("Ваш вибір: ")

        if choice == '0':
            break
        elif choice in ['1', '2', '3']:
            path = r"C:\\Users\\user\\PycharmProjects\\Data_Science_Course_SSWU\\task_8\\images\\map.tif"
            try:
                img, gray, gray_neg, blurred = load_and_prepare_image(path)

                if choice == '1':
                    harris_img, num_corners = detect_harris_corners(gray_neg)
                    visualize_results(harris_img=harris_img, num_corners=num_corners)

                elif choice == '2':
                    edges, contour_img, num_contours = detect_canny_contours(gray, blurred)
                    visualize_results(edges=edges, contour_img=contour_img, num_contours=num_contours)

                elif choice == '3':
                    harris_img, num_corners = detect_harris_corners(gray_neg)
                    edges, contour_img, num_contours = detect_canny_contours(gray, blurred)
                    visualize_results(harris_img, num_corners, edges, contour_img, num_contours)

            except Exception as e:
                print(f"Помилка: {e}")
        else:
            print("Невірний вибір. Спробуйте ще раз.")

# --------------------------- Висновок Домашнього Завдання №8 ---------------------------
"""
У цій роботі було використано два підходи до виявлення обєктів:
1. Harris Corner Detection: показав усі локальні кути з високою роздільною здатністю (~314 000 кутів).
2. Canny Edge Detection + findContours: дав чисті обєкти з вираженими ізолініями (~900+ контурів).

Найбільш ефективним для аналізу ізоліній виявився підхід на основі контурів.
Отримані результати можуть бути застосовані для картографування, виявлення форм рельєфу або аналізу динаміки водойм.
"""