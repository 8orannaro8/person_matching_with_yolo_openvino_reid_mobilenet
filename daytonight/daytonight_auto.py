import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
from torch.nn import functional as F
import time
from tqdm import tqdm

# 생성기 네트워크 정의 (원본 CycleGAN 구현과 일치하도록 수정)
class ResnetBlock(nn.Module):
    """ResNet의 Residual Block 정의 - 원본 CycleGAN 구현과 일치"""
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=True):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ResnetGenerator(nn.Module):
    """원본 CycleGAN의 ResNet 기반 생성기"""
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
        super(ResnetGenerator, self).__init__()
        assert(n_blocks >= 0)
        
        if type(norm_layer) == nn.BatchNorm2d:
            use_bias = False
        else:
            use_bias = True
            
        # 모델 구성
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        # 다운샘플링
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        # 여러 개의 Residual 블록
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, 
                                  use_dropout=use_dropout, use_bias=use_bias)]

        # 업샘플링
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
            
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

def process_image_in_patches(model, img_tensor, patch_size=256, overlap=32, device='cuda'):
    """큰 이미지를 패치로 나누어 처리"""
    c, h, w = img_tensor.shape
    stride = patch_size - overlap
    
    # 결과 이미지를 저장할 텐서
    result = torch.zeros_like(img_tensor).to(device)
    # 각 위치별 가중치 합계를 저장할 텐서 (블렌딩용)
    weight_sum = torch.zeros((1, h, w)).to(device)
    
    # 가중치 맵 생성 (가장자리에 가까울수록 가중치 감소)
    def create_weight_map(size):
        weight = torch.ones((size, size), device=device)
        for i in range(overlap//2):
            weight[i, :] *= (i + 1) / (overlap//2 + 1)
            weight[size-1-i, :] *= (i + 1) / (overlap//2 + 1)
            weight[:, i] *= (i + 1) / (overlap//2 + 1)
            weight[:, size-1-i] *= (i + 1) / (overlap//2 + 1)
        return weight.unsqueeze(0)  # [1, size, size]
    
    # 패치 가중치 맵
    patch_weight = create_weight_map(patch_size)
    
    # 패치 단위로 처리
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # 패치 경계 계산
            end_y = min(y + patch_size, h)
            end_x = min(x + patch_size, w)
            
            # 패치가 이미지 경계에 걸칠 경우 시작점 조정
            start_y = max(0, end_y - patch_size)
            start_x = max(0, end_x - patch_size)
            
            # 패치 추출
            patch = img_tensor[:, start_y:end_y, start_x:end_x].clone()
            
            # 패치 크기가 patch_size와 다른 경우 리사이즈
            if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
                patch = F.interpolate(
                    patch.unsqueeze(0), size=(patch_size, patch_size), 
                    mode='bilinear', align_corners=False
                ).squeeze(0)
            
            # 패치 처리
            with torch.no_grad():
                processed_patch = model(patch.unsqueeze(0).to(device)).squeeze(0)
            
            # 처리된 패치가 원본 패치와 크기가 다른 경우 리사이즈
            if processed_patch.shape[1] != patch.shape[1] or processed_patch.shape[2] != patch.shape[2]:
                processed_patch = F.interpolate(
                    processed_patch.unsqueeze(0), size=(end_y - start_y, end_x - start_x), 
                    mode='bilinear', align_corners=False
                ).squeeze(0)
            
            # 현재 위치의 가중치 맵 계산
            current_weight = patch_weight
            if current_weight.shape[1] != processed_patch.shape[1] or current_weight.shape[2] != processed_patch.shape[2]:
                current_weight = F.interpolate(
                    current_weight.unsqueeze(0), size=(end_y - start_y, end_x - start_x), 
                    mode='bilinear', align_corners=False
                ).squeeze(0)
            
            # 가중치 적용하여 결과에 추가
            result[:, start_y:end_y, start_x:end_x] += processed_patch * current_weight
            weight_sum[:, start_y:end_y, start_x:end_x] += current_weight
    
    # 가중치로 나누어 최종 결과 계산
    result = result / (weight_sum + 1e-8)  # 0으로 나누기 방지
    
    return result

def day_to_night_highres(input_path, model, device='cuda', patch_size=256, overlap=32, quality=95):
    """
    높은 해상도를 유지하며 낮 이미지를 밤 이미지로 변환
    
    Args:
        input_path: 입력 이미지 경로
        model: 사전 로드된 생성기 모델
        device: 계산 장치
        patch_size: 패치 크기
        overlap: 패치 간 겹침 영역
        quality: JPEG 저장 품질 (1-100)
    
    Returns:
        변환된 이미지 객체 (PIL Image)
    """
    try:
        # 이미지 로드
        img = Image.open(input_path).convert('RGB')
        original_size = img.size
        
        # 이미지 크기 확인
        if max(original_size) > patch_size:
            # 큰 이미지는 패치 처리
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            img_tensor = transform(img)
            
            # 패치 단위로 이미지 처리
            with torch.no_grad():
                result_tensor = process_image_in_patches(
                    model, img_tensor, patch_size=patch_size, 
                    overlap=overlap, device=device
                )
                
            # 이미지로 변환
            result_img = transforms.ToPILImage()(result_tensor.cpu() * 0.5 + 0.5)
            
        else:
            # 작은 이미지는 직접 처리
            transform = transforms.Compose([
                transforms.Resize((patch_size, patch_size), Image.LANCZOS),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # 변환 실행
            with torch.no_grad():
                fake_night = model(img_tensor)
            
            # 결과를 원본 크기로 리사이즈
            result = fake_night.squeeze(0).cpu() * 0.5 + 0.5
            result_img = transforms.ToPILImage()(result)
            result_img = result_img.resize(original_size, Image.LANCZOS)
        
        return result_img
        
    except Exception as e:
        print(f"이미지 처리 중 오류 ({input_path}): {e}")
        return None

def direct_save_with_quality(input_path, model, device='cuda', patch_size=256, overlap=32, quality=95):
    """
    높은 해상도와 품질을 유지하며 이미지 변환 후 직접 저장
    
    Args:
        input_path: 입력/출력 이미지 경로 (같은 파일로 덮어쓰기)
        model: 사전 로드된 생성기 모델
        device: 계산 장치
        patch_size: 패치 크기
        overlap: 패치 간 겹침 영역
        quality: JPEG 저장 품질 (1-100)
    
    Returns:
        성공 여부 (Boolean)
    """
    try:
        # 이미지 변환
        result_img = day_to_night_highres(input_path, model, device, patch_size, overlap)
        
        if result_img is not None:
            # 고품질로 저장 (JPEG 포맷인 경우)
            if input_path.lower().endswith(('.jpg', '.jpeg')):
                result_img.save(input_path, quality=quality, subsampling=0)
            else:
                result_img.save(input_path)
            return True
        else:
            return False
            
    except Exception as e:
        print(f"파일 저장 중 오류 발생 ({input_path}): {e}")
        return False

def find_all_images(input_folder, file_extensions=['.jpg', '.jpeg', '.JPG', '.JPEG']):
    """모든 하위 폴더의 이미지 파일 찾기"""
    all_image_files = []
    
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if any(file.endswith(ext) for ext in file_extensions):
                file_path = os.path.join(root, file)
                all_image_files.append(file_path)
    
    return all_image_files

def process_folder_highres(input_folder, model_path, file_extensions=['.jpg', '.jpeg', '.JPG', '.JPEG'], 
                          patch_size=256, overlap=32, batch_size=50, quality=95):
    """
    폴더 내 이미지 파일을 고품질로 변환하고 직접 덮어쓰는 방식
    
    Args:
        input_folder: 입력 폴더 경로
        model_path: 모델 파일 경로
        file_extensions: 처리할 파일 확장자 목록
        patch_size: 패치 크기
        overlap: 패치 간 겹침 영역
        batch_size: 배치 크기
        quality: JPEG 저장 품질 (1-100)
    """
    # GPU 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 모델 로드
    print(f"모델 로드 중: {model_path}")
    G_day2night = ResnetGenerator().to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # 모델 로드 시도
        if isinstance(checkpoint, dict):
            try:
                G_day2night.load_state_dict(checkpoint, strict=False)
            except:
                # 다양한 키 이름 시도
                if 'state_dict' in checkpoint:
                    G_day2night.load_state_dict(checkpoint['state_dict'], strict=False)
                elif 'net_G_A' in checkpoint:
                    G_day2night.load_state_dict(checkpoint['net_G_A'], strict=False)
                elif 'G_A' in checkpoint:
                    G_day2night.load_state_dict(checkpoint['G_A'], strict=False)
                else:
                    print("모델 구조가 일치하지 않습니다. strict=False로 시도합니다.")
                    G_day2night.load_state_dict(checkpoint, strict=False)
        else:
            G_day2night.load_state_dict(checkpoint, strict=False)
            
        print("모델 로드 성공!")
        
    except Exception as e:
        print(f"모델 로드 오류: {e}")
        return
    
    G_day2night.eval()  # 평가 모드 설정

    # 이미지 파일 찾기
    print(f"'{input_folder}' 폴더에서 이미지 탐색 중...")
    image_files = find_all_images(input_folder, file_extensions)
    
    if not image_files:
        print(f"'{input_folder}'에서 이미지를 찾을 수 없습니다.")
        return
    
    print(f"총 {len(image_files)}개 이미지 발견")
    
    # 상태 파일 경로
    status_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_images.txt')
    
    # 이미 처리된 파일 목록 로드
    processed_files = set()
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            for line in f:
                processed_files.add(line.strip())
        print(f"이미 처리된 {len(processed_files)}개 파일을 건너뜁니다.")
    
    # 처리할 파일만 필터링
    pending_files = [f for f in image_files if f not in processed_files]
    print(f"처리해야 할 파일: {len(pending_files)}개")
    
    # 변환 상태 추적
    total_files = len(pending_files)
    successful_files = 0
    failed_files = 0
    start_time = time.time()
    
    # 실패한 파일 목록
    failed_log = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'failed_images.txt')
    
    with open(status_file, 'a') as status_f, open(failed_log, 'w') as failed_f:
        # 배치 단위로 처리
        for i in range(0, total_files, batch_size):
            batch_end = min(i + batch_size, total_files)
            batch_files = pending_files[i:batch_end]
            
            print(f"\n배치 처리 중: {i+1}~{batch_end}/{total_files}")
            
            # 현재 배치의 이미지 처리
            for input_path in tqdm(batch_files, desc="이미지 변환"):
                try:
                    # 고품질 변환 및 저장
                    success = direct_save_with_quality(
                        input_path, G_day2night, device, patch_size, overlap, quality
                    )
                    
                    if success:
                        successful_files += 1
                        status_f.write(f"{input_path}\n")
                        status_f.flush()
                    else:
                        failed_files += 1
                        failed_f.write(f"{input_path}\n")
                        failed_f.flush()
                    
                except Exception as e:
                    failed_files += 1
                    failed_f.write(f"{input_path}: {str(e)}\n")
                    failed_f.flush()
            
            # 배치 처리 후 상태 보고
            processed = i + len(batch_files)
            elapsed_time = time.time() - start_time
            
            # 진행률 및 예상 시간
            if processed > 0:
                avg_time = elapsed_time / processed
                remaining = total_files - processed
                est_time = avg_time * remaining
                
                # 시간 포맷팅
                est_hours = int(est_time // 3600)
                est_mins = int((est_time % 3600) // 60)
                
                print(f"진행 상황: {processed}/{total_files} 완료")
                print(f"성공: {successful_files}, 실패: {failed_files}")
                print(f"예상 남은 시간: {est_hours}시간 {est_mins}분")
            
            # 메모리 정리
            torch.cuda.empty_cache()
    
    # 최종 결과
    total_time = time.time() - start_time
    total_hours = int(total_time // 3600)
    total_mins = int((total_time % 3600) // 60)
    
    print("\n처리 완료!")
    print(f"총 처리 파일: {total_files}, 성공: {successful_files}, 실패: {failed_files}")
    print(f"총 소요 시간: {total_hours}시간 {total_mins}분")
    
    if failed_files > 0:
        print(f"실패한 이미지 목록: {failed_log}")

if __name__ == "__main__":
    # 경로 설정
    input_folder = r"C:\Users\kkjoo\Downloads\Fashion\Training"  # 처리할 이미지 폴더
    model_path = "latest_net_G_A.pth"  # 모델 파일 경로
    
    # 변환 실행 (고해상도 품질 유지)
    process_folder_highres(
        input_folder=input_folder,
        model_path=model_path,
        file_extensions=['.jpg', '.jpeg', '.JPG', '.JPEG'],
        patch_size=256,  # 패치 크기
        overlap=32,      # 패치 간 겹침 영역
        batch_size=50,   # 배치 크기
        quality=95       # JPEG 품질 (1-100)
    )