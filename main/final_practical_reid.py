import cv2
import numpy as np
import os

class FinalPracticalReID:
    def __init__(self):
        # 더 엄격한 임계값 설정
        self.strict_threshold = 0.88    # 88% 이상 - 같은 사람 (높은 신뢰도)
        self.medium_threshold = 0.85    # 85% 이상 - 같은 사람 (중간 신뢰도)  
        self.low_threshold = 0.80       # 80% 이상 - 다른 사람 (낮은 신뢰도)
        # 88% 미만은 모두 다른 사람으로 판정
        
    def extract_robust_features(self, image_path):
        """안정적인 특징 추출"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"이미지 읽기 실패: {image_path}")
                return None
            
            h, w, _ = image.shape
            
            # 1. 중앙 영역에 집중 (배경 노이즈 최소화)
            center_margin = min(w, h) // 10  # 10% 여백
            center_image = image[center_margin:h-center_margin, center_margin:w-center_margin]
            
            if center_image.size == 0:
                center_image = image  # 이미지가 너무 작으면 전체 사용
            
            gray = cv2.cvtColor(center_image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(center_image, cv2.COLOR_BGR2HSV)
            
            features = []
            
            # 2. 색상 특징 (상체/하체 분리)
            h_center, w_center = gray.shape
            
            # 상체 (상위 60%)
            upper_region = hsv[:int(h_center*0.6), :]
            # 하체 (하위 60%)  
            lower_region = hsv[int(h_center*0.4):, :]
            
            def extract_color_features(region):
                if region.size == 0:
                    return np.zeros(48)  # 16+16+16
                
                # HSV 히스토그램 (간소화)
                h_hist = cv2.calcHist([region], [0], None, [16], [0, 180])
                s_hist = cv2.calcHist([region], [1], None, [16], [0, 256])
                v_hist = cv2.calcHist([region], [2], None, [16], [0, 256])
                
                # 정규화
                h_hist = h_hist.flatten() / (np.sum(h_hist) + 1e-7)
                s_hist = s_hist.flatten() / (np.sum(s_hist) + 1e-7)
                v_hist = v_hist.flatten() / (np.sum(v_hist) + 1e-7)
                
                return np.concatenate([h_hist, s_hist, v_hist])
            
            upper_colors = extract_color_features(upper_region)
            lower_colors = extract_color_features(lower_region)
            
            features.extend(upper_colors)
            features.extend(lower_colors)
            
            # 3. 텍스처 특징 (단순화된 LBP)
            def simple_texture(img_region):
                if img_region.size == 0:
                    return np.zeros(8)
                
                # 단순 에지 밀도
                edges = cv2.Canny(img_region, 50, 150)
                edge_density = np.sum(edges > 0) / (img_region.shape[0] * img_region.shape[1])
                
                # 방향성 특징
                grad_x = cv2.Sobel(img_region, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(img_region, cv2.CV_64F, 0, 1, ksize=3)
                
                magnitude = np.sqrt(grad_x**2 + grad_y**2)
                angle = np.arctan2(grad_y, grad_x)
                
                # 방향 히스토그램 (8방향)
                angle_hist, _ = np.histogram(angle.flatten(), bins=8, range=(-np.pi, np.pi), weights=magnitude.flatten())
                angle_hist = angle_hist / (np.sum(angle_hist) + 1e-7)
                
                return np.concatenate([[edge_density], angle_hist[:7]])  # 8차원
            
            upper_texture = simple_texture(gray[:int(h_center*0.6), :])
            lower_texture = simple_texture(gray[int(h_center*0.4):, :])
            
            features.extend(upper_texture)
            features.extend(lower_texture)
            
            # 4. 기본 형태 특징
            # 종횡비
            aspect_ratio = w_center / h_center
            
            # 밝기 통계
            brightness_stats = [
                np.mean(gray),
                np.std(gray),
                np.mean(gray[:h_center//2, :]),  # 상체 밝기
                np.mean(gray[h_center//2:, :]),  # 하체 밝기
            ]
            
            # 색상 통계
            color_stats = [
                np.mean(hsv[:, :, 0]),  # 평균 색조
                np.mean(hsv[:, :, 1]),  # 평균 채도
                np.mean(hsv[:, :, 2]),  # 평균 명도
            ]
            
            features.extend([aspect_ratio])
            features.extend(brightness_stats)
            features.extend(color_stats)
            
            return np.array(features)
            
        except Exception as e:
            print(f"특징 추출 오류 {image_path}: {e}")
            return None
    
    def compare_final(self, img1_path, img2_path):
        """최종 실용적 비교"""
        
        print(f"🔍 분석 1: {os.path.basename(img1_path)}")
        features1 = self.extract_robust_features(img1_path)
        
        print(f"🔍 분석 2: {os.path.basename(img2_path)}")
        features2 = self.extract_robust_features(img2_path)
        
        if features1 is None or features2 is None:
            return {
                'similarity': 0.0,
                'is_same_person': False,
                'confidence': '오류',
                'reason': '특징 추출 실패'
            }
        
        similarity = self.cosine_similarity(features1, features2)
        

        
        # 판단 로직
        if similarity >= self.strict_threshold:
            confidence = "높음"
            is_same = True
            reason = f"높은 유사도 ({similarity:.3f} >= {self.strict_threshold})"
        elif similarity >= self.medium_threshold:
            confidence = "중간"
            is_same = True
            reason = f"중간 유사도 ({similarity:.3f} >= {self.medium_threshold})"
        elif similarity >= self.low_threshold:
            confidence = "낮음"
            is_same = False
            reason = f"낮은 유사도 ({similarity:.3f} >= {self.low_threshold})"
        else:
            confidence = "매우 낮음"
            is_same = False
            reason = f"매우 낮은 유사도 ({similarity:.3f})"
        
        return {
            'similarity': similarity,
            'is_same_person': is_same,
            'confidence': confidence,
            'reason': reason,
            'feature_dims': len(features1)
        }

    def cosine_similarity(self,a, b): 
            a = np.asarray(a).flatten() 
            b = np.asarray(b).flatten() 
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-7)
    
