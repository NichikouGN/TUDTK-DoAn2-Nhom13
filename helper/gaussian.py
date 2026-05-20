epsilon = 1e-9                  # Ngưỡng sai số làm tròn về 0
warning_epsilon = 1e-6          # Ngưỡng cảnh báo mất ổn định số học

# Làm sạch sai số chấm động
def clean_value(x):
    # Nếu là số phức
    if isinstance(x, complex) or type(x).__name__ in ('complex128', 'complex64'):
        r = clean_value(x.real)
        i = clean_value(x.imag)
        if i == 0.0:
            return r
        return complex(r, i)
        
    # Nếu là số thực
    if abs(x) < epsilon:
        return 0.0
        
    # rounded_x = round(float(x), 8)
    # if abs(x - rounded_x) < epsilon:
    #     return rounded_x + 0.0
        
    # return float(x) + 0.0
    return float(x)

# Nhận ma trận A và vector b, thực hiện phép khử Gauss để đưa về dạng bậc thang (REF) trên ma trận mở rộng A|b
#
# Tham số đầu vào:
#   A (array): Ma trận hệ số
#   b (array): Vector hệ số tự do, có thể là None để tái sử dụng hàm cho các file determinant.py, inverse.py, rank_basis.py
#   to_rref: Tùy chọn trả về REF hoặc RREF (mặc định REF)
#   silent: Đưa ra cảnh báo mất ổn định số học (mặc định False)
#
# Kết quả trả về:
#   REF hoặc RREF của A|b hoặc A
#   Nghiệm x hoặc None nếu b = None hoặc hệ vô nghiệm
#   Số lần hoán vị dòng
def gaussian_elimination(A, b = None, to_rref = False, silent = False):
    # Kiểm tra tính hợp lệ của dữ liệu đầu vào
    if len(A) == 0:
        raise ValueError("Ma trận A không được rỗng")

    for row in A:
        if len(row) != len(A[0]):
            raise ValueError("Tất cả các dòng của ma trận A phải có cùng độ dài")

    # Chuẩn hóa tất cả phần tử về kiểu số thực/số phức
    A = [[x if isinstance(x, complex) else float(x) for x in row] for row in A]

    # Ghép ma trận mở rộng M = A|b
    if b is not None:
        if len(A) != len(b):
            raise ValueError("Ma trận A và vector b phải có cùng số dòng")
        b = [float(x) for x in b]
        M = [A[i] + [b[i]] for i in range(len(A))]
    else:
        # Nếu b = None, chỉ copy ma trận A vào M
        M = [row[:] for row in A]

    nrows = len(M)
    ncols = len(M[0])
    swap_count = 0
    cur_row = 0
    
    for k in range(ncols-1):
        if cur_row >= nrows:
            break

        pivot = (cur_row, k)
        # Tìm phần tử có trị tuyệt đối lớn nhất trong cột hiện tại để làm phần tử chốt
        for row in range(cur_row + 1, nrows):
            if abs(M[row][k]) > abs(M[pivot[0]][pivot[1]]):
                pivot = (row, k)

        pivot_val = abs(M[pivot[0]][pivot[1]])

        # Nếu |M_pk| = 0 (trong máy tính coi như < epsilon)
        if pivot_val < epsilon:
            if b is not None:
                if not silent:
                    print(f"Không tồn tại pivot tại cột {k} (hệ không có nghiệm duy nhất)")
            
            M[pivot[0]][pivot[1]] = 0
            continue

        # Nếu 0 < |M_pk| < warning_epsilon
        elif pivot_val < warning_epsilon:
            if not silent:
                print(f"[Cảnh báo]: pivot tại {(pivot[0], pivot[1])} gần bằng 0, hệ có thể bị mất ổn định số học")

        # Hoán vị dòng nếu phần tử lớn nhất không ở dòng hiện tại
        if pivot[0] != cur_row:
            M[cur_row], M[pivot[0]] = M[pivot[0]], M[cur_row]
            swap_count += 1

        if not to_rref:
            # Khử Gauss để trả REF
            for row in range(cur_row + 1, nrows):
                factor = M[row][k] / M[cur_row][k]
                for col in range(k, ncols):
                    M[row][col] -= factor * M[cur_row][col]
                    if abs(M[row][col]) < epsilon:
                        M[row][col] = 0.0
        
        else:
            # Khử Gauss-Jordan để trả RREF
            pivot_val = M[cur_row][k]
            
            # Chia toàn bộ dòng chốt hiện tại chia cho phần tử chốt
            for col in range(k, ncols):
                M[cur_row][col] /= pivot_val
                if abs(M[cur_row][col]) < epsilon:
                    M[cur_row][col] = 0.0
                    
            # Triệt tiêu các phần tử cùng cột ở mọi dòng khác
            for row in range(nrows):
                if row != cur_row:
                    factor = M[row][k]
                    # Chỉ tính toán nếu phần tử cần khử thực sự khác 0
                    if abs(factor) >= epsilon:
                        for col in range(k, ncols):
                            M[row][col] -= factor * M[cur_row][col]
                            if abs(M[row][col]) < epsilon:
                                M[row][col] = 0.0
        
        cur_row += 1

    # Bóc tách lại thành ma trận U và vector c tùy thuộc vào việc có truyền b hay không
    if b is not None:
        U = [row[:-1] for row in M]
        c = [row[-1] for row in M]

        # Kiểm tra trường hợp vô nghiệm: Dòng có dạng [0, 0, ..., 0 | c] với c != 0
        for row in range(nrows):
            if all(abs(M[row][col]) < epsilon for col in range(ncols - 1)) and abs(M[row][ncols-1]) >= epsilon:
                return U, None, swap_count
            
        # Thế ngược để tìm vector nghiệm x
        x = back_substitution(U, c)
        U = [[clean_value(val) for val in row] for row in U]
    else:
        # Nếu b = None, toàn bộ ma trận M sau khi biến đổi chính là U
        U = [[clean_value(val) for val in row] for row in M]
        x = None
    
    return U, x, swap_count


# Nhận ma trận U và vector c (một phần của dạng bậc thang từ A|b) và thực hiện phép thế ngược
#
# Tham số đầu vào:
#   U (array): Ma trận hệ số đã có dạng tam giác trên
#   c (array): Vector hệ số tự do đã qua biến đổi
#
# Kết quả trả về:
#   Nghiệm x: mảng Float nếu nghiệm duy nhất, mảng String nếu vô số nghiệm
def back_substitution(U, c):
    nrows = len(U)
    ncols = len(U[0])
    pivot_cols = []

    # Tìm tất cả các cột chứa phần tử chốt
    cur_row = 0 
    for k in range(ncols):
        if cur_row >= nrows:
            break
       
        if abs(U[cur_row][k]) < epsilon:
           continue
       
        pivot_cols.append((cur_row, k))
        cur_row += 1

    # Phân loại cột chốt và cột biến tự do
    pivot_only_cols = [c for (r,c) in pivot_cols]
    free_vars = [col for col in range(ncols) if col not in pivot_only_cols]

    # Khởi tạo mảng nghiệm dùng Dictionary để lưu hệ số
    # Ví dụ cách lưu hệ số: {'const': 5.0, 't1': -2.0} --> '5.0 - 2.0*t1'
    sol = [{} for _ in range(ncols)]

    # Gán biến tự do t1, t2... cho các cột không có phần tử chốt
    if free_vars:
        for i, col in enumerate(free_vars):
            sol[col] = {'const': 0.0, f"t{i+1}": 1.0}

    # Thực hiện phép thế ngược
    for row, col in reversed(pivot_cols):
        # Bắt đầu với hằng số từ vector c
        expr = {'const': c[row]}

        # Trừ giá trị các ẩn đã biết ở sau. Công thức: x + y = c => x = c - y
        for j in range(ncols - 1, col, -1):
            if sol[j]:
                coeff = U[row][j]
                if abs(coeff) > epsilon:
                    # Trừ tương ứng từng hệ số trong Dictionary
                    for key, val in sol[j].items():
                        expr[key] = expr.get(key, 0.0) - (coeff * val)

        # Chia cho phần tử chốt để ra nghiệm cuối của ẩn. Công thức: ax = b => x = b/a
        pivot_val = U[row][col]
        for key in expr:
            expr[key] = expr[key] / pivot_val

        sol[col] = expr

    # Định dạng đầu ra: ép Float nếu nghiệm duy nhất hoặc ráp thành String nếu vô số nghiệm
    
    # TH1: Hệ có nghiệm duy nhất
    if not free_vars:
        return [clean_value(sol[i].get('const', 0.0)) for i in range(ncols)]

    # TH2: Hệ vô số nghiệm -> Ráp Dictionary thành chuỗi
    final_sol = []
    for i in range(ncols):
        terms = []
        d = sol[i]
        
        # Xử lý phần hằng số
        const_val = round(d.get('const', 0.0), 6)
        if abs(const_val) > epsilon or not d:
            terms.append(str(const_val))
            
        # Xử lý phần biến tự do t1, t2...
        for key, val in d.items():
            if key != 'const':
                val_rounded = round(val, 6)
                if abs(val_rounded) > epsilon:
                    # Kiểm tra hệ số có xấp xỉ 1.0 hoặc -1.0 để bỏ hệ số (in "t1" thay "1.0*t1")
                    is_one = abs(abs(val_rounded) - 1.0) < epsilon
                    coeff_str = "" if is_one else f"{abs(val_rounded)}*"
                    
                    # Ráp chuỗi cùng dấu +/-
                    if val_rounded > 0 and terms:
                        terms.append(f"+ {coeff_str}{key}")
                    elif val_rounded > 0:
                        terms.append(f"{coeff_str}{key}")
                    else: 
                        terms.append(f"- {coeff_str}{key}")
                    
        # Nếu tất cả hệ số đều bằng 0, gán nghiệm bằng 0
        if not terms:
            final_sol.append("0")
        else:
            final_sol.append(" ".join(terms))
            
    return final_sol