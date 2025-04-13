#!/usr/bin/env python3
import sys
import dat  # Giả sử module dat.py nằm cùng thư mục hoặc đã được đặt đúng package

def main():
    # Khởi tạo model với các file vector và từ điển
    model = dat.Model("glove.840B.300d.txt", "words.txt")
    
    # Nhập vào chuỗi từ không giới hạn số từ (ví dụ 50 từ hoặc nhiều hơn)
    input_str = input("Nhập vào các từ (cách nhau bằng khoảng trắng): ").strip()
    words_list = input_str.split()
    
    # Kiểm tra nếu tổng số từ nhỏ hơn 10 thì dừng
    if len(words_list) < 10:
        print("Lỗi: Cần ít nhất 10 từ để tạo thành một mẫu.")
        sys.exit(1)
    
    # Tính số mẫu (mỗi mẫu gồm 10 từ)
    num_groups = len(words_list) // 10  # bỏ qua phần dư nếu không đủ 10 từ
    
    for group_idx in range(num_groups):
        # Lấy 10 từ đầu tiên của nhóm (mẫu)
        group = words_list[group_idx*10 : (group_idx+1)*10]
        # Tính DAT score cho nhóm này
        score = model.dat(group, minimum=10)
        if score is None:
            print(f"Mẫu {group_idx+1}: Không đủ từ hợp lệ để tính toán DAT score.")
        else:
            print(f"Mẫu {group_idx+1}: DAT score: {score:.4f}")

if __name__ == "__main__":
    main()