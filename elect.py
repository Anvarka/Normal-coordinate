import numpy as np


f=open('input.xyz', 'r')

# создаем словарь атомных масс химических элементов таблицы Менделеева
mass_atoms = {"H": 1, "He": 4, "Li": 7, "Be": 9, "B": 11, "C": 12, "N": 14, "O": 16, "F": 19, "Ne": 20,
              "Na": 23, "Mg": 24, "Al": 27, "Si": 28, "P": 31, "S": 32, "Cl": 35.5, "Ar": 40,
              "K": 39, "Ca": 40, "Sc": 45, "Ti": 48, "V": 51, "Cr": 52, "Mn": 55, "Fe": 56, "Co": 59, "Ni": 59,
              "Cu": 64, "Zn": 65, "Ga": 70, "Ge": 73, "As": 75, "Se": 79, "Br": 80, "Kr": 84,
              "Rb": 85, "Sr": 88, "Y": 89, "Zr": 91, "Nb": 93, "Mo": 96, "Tc": 99, "Ru": 101, "Rh": 103, "Pd": 106,
              "Ag": 108, "Cd": 112, "In": 115, "Sn": 119, "Sb": 122, "Te": 128, "I": 127, "Xe": 131,
              "Cs": 133, "Ba": 137, "La*": 139     , "Hf": 178, "Ta": 181, "W": 184, "Re": 186, "Os": 190, "Ir": 192, "Pt": 195,
              "Au": 197, "Hg": 200, "Tl": 204, "Pb": 207, "Bi": 209, "Po": 209, "At": 210, "Rn": 222,
              "Fr": 223, "Ra": 226, "Ac**": 227,    "Rf": 261, "Db": 262, "Sg": 263, "Bh": 262, "Hs": 265, "Mt": 266, "Ds": 271}

# количество атомов в  молекуле
N = int(f.readline())

# матрица с исходными координатами
coordinate_atoms = np.empty((N, 3))
print(coordinate_atoms)

# матрица масс
mass = np.zeros(N)

# комментарий (ненужная строка)
f.readline()

# объявляем компоненты тензора инерции
I_xx = 0
I_yy = 0
I_zz = 0
I_xy = 0
I_xz = 0
I_yz = 0

# объявляем три вектора трансляции
v_1 = np.zeros(3 * N)
v_2 = np.zeros(3 * N)
v_3 = np.zeros(3 * N)

# объявляем три вектора вращения
w_1 = np.zeros(3 * N)
w_2 = np.zeros(3 * N)
w_3 = np.zeros(3 * N)

# объявляем координаты центра масс
summ_mass_x = 0
summ_mass_y = 0
summ_mass_z = 0

# общая масса
summ_mass = 0

# координаты в центре масс
coordinate_atoms_cm = np.zeros((N, 3))


for i in range(N):   # рассматриваем каждый атом

    # узнаем массу атома из словаря
    m = mass_atoms[f.read(2).replace(" ", "")]

    # вычисляем три вектора трансляции
    v_1[3*i] = np.sqrt(m)
    v_2[1+3*i] = np.sqrt(m)
    v_3[2+3*i] = np.sqrt(m)

    # взяли одну из N строк
    stroka = f.readline()

    # вытаскиваем координаты из файла
    coordinate_atoms[i][0] = float(stroka.split()[0])
    coordinate_atoms[i][1] = float(stroka.split()[1])
    coordinate_atoms[i][2] = float(stroka.split()[2])

    # получаем компоненты центра масс
    summ_mass_x = summ_mass_x + m * coordinate_atoms[i][0]
    summ_mass_y = summ_mass_y + m * coordinate_atoms[i][1]
    summ_mass_z = summ_mass_z + m * coordinate_atoms[i][2]
    summ_mass = summ_mass + m

    # заполняем матрицу масс
    mass[i] = m


# получаем координаты центра масс
x1 = summ_mass_x/summ_mass
y1 = summ_mass_y/summ_mass
z1 = summ_mass_z/summ_mass


for i in range(N):      # поиск тензора инерции

    # смещаем все координаты в центр масс
    coordinate_atoms_cm[i][0] = coordinate_atoms[i][0] - x1
    coordinate_atoms_cm[i][1] = coordinate_atoms[i][1] - y1
    coordinate_atoms_cm[i][2] = coordinate_atoms[i][2] - z1

    # вычисляем диагональные элементы
    I_xx = I_xx + m*(coordinate_atoms_cm[i][1]**2+coordinate_atoms_cm[i][2]**2)
    I_yy = I_yy + m*(coordinate_atoms_cm[i][0]**2+coordinate_atoms_cm[i][2]**2)
    I_zz = I_zz + m*(coordinate_atoms_cm[i][0]**2+coordinate_atoms_cm[i][1]**2)

    # вычисляем недиагональные элементы
    I_xy = I_xy + m*coordinate_atoms_cm[i][0]*coordinate_atoms_cm[i][1]
    I_xz = I_xz + m*coordinate_atoms_cm[i][0]*coordinate_atoms_cm[i][2]
    I_yz = I_yz + m*coordinate_atoms_cm[i][1]*coordinate_atoms_cm[i][2]


# получаем тензор момента инерции
I = np.array([[I_xx, I_xy, I_xz],
              [I_xy, I_yy, I_yz],
              [I_xz, I_yz, I_zz]])

# матрица, которая диагонализует I
_, M = np.linalg.eig(I)

# проверка, будет ли диагональной
M1 = np.linalg.inv(M)
print(np.matmul(np.matmul(M1, I), M))


for i in range(N):

    # первый вращательный вектор
    w_1[3*i] = (coordinate_atoms_cm[i][1]* M[2][0]-coordinate_atoms_cm[i][2]*M[1][0]) / np.sqrt(mass[i])
    w_1[3*i + 1] = (coordinate_atoms_cm[i][1]* M[2][1]-coordinate_atoms_cm[i][2]*M[1][1]) / np.sqrt(mass[i])
    w_1[3*i + 2] = (coordinate_atoms_cm[i][1]* M[2][2]-coordinate_atoms_cm[i][2]*M[1][2]) / np.sqrt(mass[i])

    # второй вращательный вектор
    w_2[3*i] = (coordinate_atoms_cm[i][2]* M[0][0]-coordinate_atoms_cm[i][2]*M[2][0]) / np.sqrt(mass[i])
    w_2[3*i + 1] = (coordinate_atoms_cm[i][2]* M[0][1]-coordinate_atoms_cm[i][2]*M[2][1]) / np.sqrt(mass[i])
    w_2[3*i + 2] = (coordinate_atoms_cm[i][2]* M[0][2]-coordinate_atoms_cm[i][2]*M[2][2]) / np.sqrt(mass[i])

    # третий вращательный вектор
    w_3[3*i] = (coordinate_atoms_cm[i][2]* M[1][0]-coordinate_atoms_cm[i][1]*M[0][0]) / np.sqrt(mass[i])
    w_3[3*i + 1] = (coordinate_atoms_cm[i][2]* M[1][1]-coordinate_atoms_cm[i][1]*M[0][1]) / np.sqrt(mass[i])
    w_3[3*i + 2] = (coordinate_atoms_cm[i][2]* M[1][2]-coordinate_atoms_cm[i][1]*M[0][2]) / np.sqrt(mass[i])


# Соединяем матрицы
D = np.column_stack((v_1, v_2, v_3, w_1, w_2, w_3, np.eye(3*N, 3*N-6)))

# Ортогонализация
W, _ = np.linalg.qr(D)

# выкинули первые шесть векторов
W = W[:, 6:]
print(W)

# получаем в нормальных модах
A = np.matmul(np.matmul(W, coordinate_atoms), W)
print(A)

# получаем диагональную матрицу
_, L = np.linalg.eig(A)
X_end = np.matmul(np.matmul(L, A), L)
print(X_end)
