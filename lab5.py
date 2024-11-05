import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Определяем функцию для минимизации
def objective_function(x):
    x1, x2, x3 = x
    return 3 * (x1 - 4) ** 2 + 5 * (x2 + 3) ** 2 + 7 * (2 * x3 + 1) ** 2


# Класс, представляющий одну частицу
class Particle:
    def __init__(self, bounds):
        self.position = np.array([np.random.uniform(low, high) for low, high in bounds])
        self.velocity = np.zeros_like(self.position)
        self.best_position = self.position.copy()
        self.best_score = objective_function(self.position)

    def update_velocity(self, global_best_position, speed_coef, personal_best_coef, global_best_coef, compression_coef):
        # Обновляем скорость с учетом коэффициента сжатия
        inertia = speed_coef * self.velocity
        cognitive = personal_best_coef * np.random.rand() * (self.best_position - self.position)
        social = global_best_coef * np.random.rand() * (global_best_position - self.position)

        # Применяем коэффициент сжатия
        new_velocity = (inertia + cognitive + social) * compression_coef
        self.velocity = new_velocity

    def update_position(self, bounds):
        self.position += self.velocity
        for i in range(len(self.position)):
            if self.position[i] < bounds[i][0]:
                self.position[i] = bounds[i][0]
            elif self.position[i] > bounds[i][1]:
                self.position[i] = bounds[i][1]
        score = objective_function(self.position)
        if score < self.best_score:
            self.best_score = score
            self.best_position = self.position.copy()


# Класс для алгоритма роя частиц (PSO)
class PSO:
    def __init__(self, bounds, num_particles, speed_coef, personal_best_coef, global_best_coef, compression_coef,
                 iterations):
        self.bounds = bounds
        self.num_particles = num_particles
        self.speed_coef = speed_coef
        self.personal_best_coef = personal_best_coef
        self.global_best_coef = global_best_coef
        self.compression_coef = compression_coef
        self.iterations = iterations
        self.particles = [Particle(bounds) for _ in range(num_particles)]
        self.global_best_position = min(self.particles, key=lambda p: p.best_score).best_position
        self.global_best_score = objective_function(self.global_best_position)

    def optimize(self):
        for _ in range(self.iterations):
            for particle in self.particles:
                particle.update_velocity(self.global_best_position, self.speed_coef, self.personal_best_coef,
                                         self.global_best_coef, self.compression_coef)
                particle.update_position(self.bounds)
                if particle.best_score < self.global_best_score:
                    self.global_best_position = particle.best_position
                    self.global_best_score = particle.best_score
        return self.global_best_position, self.global_best_score


# Класс для графического интерфейса с использованием tkinter
class PSO_GUI:
    def __init__(self, root):
        self.root = root
        root.title("Алгоритм роя частиц (PSO) с коэффициентом сжатия")

        # Поля для ввода параметров
        tk.Label(root, text="Количество частиц:").grid(row=0, column=0)
        self.num_particles_entry = tk.Entry(root)
        self.num_particles_entry.insert(0, "300")
        self.num_particles_entry.grid(row=0, column=1)

        tk.Label(root, text="Количество итераций:").grid(row=1, column=0)
        self.iterations_entry = tk.Entry(root)
        self.iterations_entry.insert(0, "100")
        self.iterations_entry.grid(row=1, column=1)

        tk.Label(root, text="Коэффициент скорости:").grid(row=2, column=0)
        self.speed_coef_entry = tk.Entry(root)
        self.speed_coef_entry.insert(0, "0.3")
        self.speed_coef_entry.grid(row=2, column=1)

        tk.Label(root, text="Коэффициент лучшего значения (личного):").grid(row=3, column=0)
        self.personal_best_coef_entry = tk.Entry(root)
        self.personal_best_coef_entry.insert(0, "2")
        self.personal_best_coef_entry.grid(row=3, column=1)

        tk.Label(root, text="Коэффициент глобального лучшего значения:").grid(row=4, column=0)
        self.global_best_coef_entry = tk.Entry(root)
        self.global_best_coef_entry.insert(0, "5")
        self.global_best_coef_entry.grid(row=4, column=1)

        tk.Label(root, text="Коэффициент сжатия:").grid(row=5, column=0)
        self.compression_coef_entry = tk.Entry(root)
        self.compression_coef_entry.insert(0, "0.5")
        self.compression_coef_entry.grid(row=5, column=1)

        # Кнопка для запуска оптимизации
        self.start_button = tk.Button(root, text="Запустить оптимизацию", command=self.run_pso)
        self.start_button.grid(row=6, column=0, columnspan=2)

        # Поле для вывода результатов
        self.result_label = tk.Label(root, text="")
        self.result_label.grid(row=7, column=0, columnspan=2)

        # Область для графика
        self.figure, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, root)
        self.canvas.get_tk_widget().grid(row=8, column=0, columnspan=2)

    def run_pso(self):
        try:
            # Получение значений из полей ввода
            num_particles = int(self.num_particles_entry.get())
            iterations = int(self.iterations_entry.get())
            speed_coef = float(self.speed_coef_entry.get())
            personal_best_coef = float(self.personal_best_coef_entry.get())
            global_best_coef = float(self.global_best_coef_entry.get())
            compression_coef = float(self.compression_coef_entry.get())

            # Границы поиска для каждой переменной
            bounds = [(0, 10), (-10, 10), (-5, 5)]  # Настроить по необходимости

            # Создание экземпляра PSO и запуск оптимизации
            pso = PSO(bounds, num_particles, speed_coef, personal_best_coef, global_best_coef, compression_coef,
                      iterations)
            best_position, best_score = pso.optimize()

            # Вывод результатов
            self.result_label.config(text=f"Лучшее решение: {best_position}\nЗначение функции: {best_score}")

            # Отображение графика с положениями частиц
            self.ax.clear()
            self.ax.plot([p.position[0] for p in pso.particles], [p.position[1] for p in pso.particles], 'ko')
            self.ax.set_title("Позиции частиц")
            self.canvas.draw()

        except ValueError:
            messagebox.showerror("Ошибка ввода", "Пожалуйста, введите корректные числа.")


# Основная часть программы для запуска интерфейса
if __name__ == "__main__":
    root = tk.Tk()
    gui = PSO_GUI(root)
    root.mainloop()
