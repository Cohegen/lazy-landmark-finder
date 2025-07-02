import os
import shutil
from collections import defaultdict

def get_top_landmarks(src_dir, top_n=20):
    counts = []
    for landmark in os.listdir(src_dir):
        landmark_path = os.path.join(src_dir, landmark)
        if os.path.isdir(landmark_path):
            num_images = len([f for f in os.listdir(landmark_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))])
            counts.append((landmark, num_images))
    counts.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in counts[:top_n]]

def copy_landmarks(src_dir, dst_dir, landmark_list):
    os.makedirs(dst_dir, exist_ok=True)
    for landmark in landmark_list:
        src_path = os.path.join(src_dir, landmark)
        dst_path = os.path.join(dst_dir, landmark)
        if os.path.exists(dst_path):
            shutil.rmtree(dst_path)
        shutil.copytree(src_path, dst_path)
        print(f'Copied {landmark} to {dst_path}')

if __name__ == '__main__':
    src_dir = 'data/egypt_landmarks/images'
    dst_dir = 'dataset'
    top_landmarks = get_top_landmarks(src_dir, top_n=20)
    print('Top 20 landmarks:', top_landmarks)
    copy_landmarks(src_dir, dst_dir, top_landmarks)
    print('Done!')
        
