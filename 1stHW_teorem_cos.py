##Подгружаем нужную для кода библиотеку
import math

##Запрашиваем у пользователя данные и сохраняем их
first_side = float(input("Введите длину первой стороны: "))
second_side = float(input("Введите длину второй стороны: "))
angle = float(input("Введите значение угла между сторонами в градусах: "))

##Вычисления по теореме косинуса
third_side = math.sqrt(first_side**2 + second_side**2 - 2*first_side*second_side * math.cos(math.radians(angle)))

##Вывод результата для пользователя
print(f"Длина третьей стороны: {third_side}")
