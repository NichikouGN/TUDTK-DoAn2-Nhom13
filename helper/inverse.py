from helper.gaussian import gaussian_elimination

def inverse(A):
    n = len(A)
    # Kiểm tra ma trận rỗng và ma trận vuông
    if n == 0 or n != len(A[0]):
        raise ValueError("Chỉ ma trận vuông mới có ma trận nghịch đảo.")

    # (A^-1)^T
    inv_A_cols = []
    
    # Tạo các cột ma trận đơn vị I và dùng hàm Gauss giải từng cột
    for i in range(n):
        # Tạo cột thứ i của I
        e_i = [1.0 if j == i else 0.0 for j in range(n)]
        
        # Tái sử dụng hàm Gauss để tính từng cột của ma trận nghịch đảo
        # A x A^-1 = I => A x (Cột i của A^-1) = (Cột i của I)
        U, x, _ = gaussian_elimination(A, b = e_i, silent = True)
        
        '''
        Kiểm tra tính khả nghịch dựa vào kết quả x.
        Nếu A khả nghịch, tồn tại duy nhất A^-1 => mỗi cột của A^-1 phải được xác định duy nhất.
        '''

        # Nếu vô nghiệm (None) hoặc vô số nghiệm (mảng string)
        if x is None or (len(x) > 0 and isinstance(x[0], str)):
            print("Ma trận không khả nghịch.")
            return None
            
        # Nếu có nghiệm duy nhất, x chính là cột của ma trận nghịch đảo
        inv_A_cols.append(x)
        
    # Chuyển (A^-1)^T thành A^-1
    inv_A = [[inv_A_cols[j][i] for j in range(n)] for i in range(n)]
    
    return inv_A