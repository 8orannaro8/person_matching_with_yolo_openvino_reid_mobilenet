import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from torch.nn import functional as F

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
    """
    큰 이미지를 패치로 나누어 처리한 후 결과를 합치는 함수
    
    Args:
        model: 이미지 변환 모델
        img_tensor: 변환할 이미지 텐서 [C, H, W]
        patch_size: 패치 크기
        overlap: 패치 간 겹치는 픽셀 수
        device: 계산 장치
    
    Returns:
        결합된 결과 이미지 텐서 [C, H, W]
    """
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

def day_to_night(input_path, output_path, model_path, keep_original_resolution=True, patch_size=256, overlap=32, strict=False):
    """
    낮 이미지를 밤 이미지로 변환하는 함수
    
    Args:
        input_path: 입력 이미지 경로
        output_path: 출력 이미지 경로
        model_path: 사전 학습된 Generator 모델 경로
        keep_original_resolution: 원본 해상도 유지 여부
        patch_size: 패치 처리 시 패치 크기
        overlap: 패치 간 겹침 영역 크기
        strict: 모델 로딩 시 strict 모드 활성화 여부
    """
    # GPU 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 모델 로드
    G_day2night = ResnetGenerator().to(device)
    
    # 체크포인트 로드
    try:
        print(f"모델 파일 로드 중: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # 체크포인트 구조 확인
        print("체크포인트 키 확인:")
        if isinstance(checkpoint, dict) and any(k in checkpoint for k in ['state_dict', 'net_G_A', 'G_A']):
            # 일반적인 체크포인트 구조
            if 'state_dict' in checkpoint:
                G_day2night.load_state_dict(checkpoint['state_dict'], strict=strict)
                print("'state_dict' 키를 사용하여 모델 로드")
            elif 'net_G_A' in checkpoint:
                G_day2night.load_state_dict(checkpoint['net_G_A'], strict=strict)
                print("'net_G_A' 키를 사용하여 모델 로드")
            elif 'G_A' in checkpoint:
                G_day2night.load_state_dict(checkpoint['G_A'], strict=strict)
                print("'G_A' 키를 사용하여 모델 로드")
        else:
            # 직접 state_dict인 경우
            G_day2night.load_state_dict(checkpoint, strict=strict)
            print("직접 state_dict로 모델 로드")
            
        print("모델 로드 성공!")
        
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        print("\n모델 로드 실패. strict=False로 다시 시도...")
        
        try:
            # strict=False로 다시 시도
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                G_day2night.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                G_day2night.load_state_dict(checkpoint, strict=False)
            print("strict=False로 모델 로드 성공!")
        except Exception as e2:
            print(f"두 번째 시도에서도 실패: {e2}")
            print("프로그램을 종료합니다.")
            return False
    
    G_day2night.eval()  # 평가 모드로 설정
    
    # 이미지 로드
    try:
        print(f"이미지 파일 로드 중: {input_path}")
        img = Image.open(input_path).convert('RGB')
        original_width, original_height = img.size
        print(f"원본 이미지 크기: {original_width}x{original_height}")
        
        if keep_original_resolution and max(original_width, original_height) > patch_size:
            print(f"원본 해상도 유지 모드: 패치 크기 {patch_size}x{patch_size}, 겹침 {overlap}px로 처리")
            
            # 이미지를 텐서로 변환 (정규화만 적용)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            img_tensor = transform(img)
            
            # 패치 단위로 이미지 처리
            with torch.no_grad():
                result_tensor = process_image_in_patches(
                    G_day2night, img_tensor, patch_size=patch_size, 
                    overlap=overlap, device=device
                )
                
            # 이미지로 변환하여 저장
            result_img = transforms.ToPILImage()(result_tensor.cpu() * 0.5 + 0.5)
            result_img.save(output_path)
            print(f"변환된 이미지 크기: {result_img.size[0]}x{result_img.size[1]}")
            
        else:
            # 작은 이미지 또는 원본 해상도 유지를 원하지 않는 경우 일반 처리
            print("표준 처리 모드: 256x256 해상도로 변환 후 다시 원본 크기로 조정")
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # 변환 실행
            with torch.no_grad():
                fake_night = G_day2night(img_tensor)
            
            # 결과를 원본 크기로 리사이즈
            result = fake_night.squeeze(0).cpu() * 0.5 + 0.5
            result_img = transforms.ToPILImage()(result)
            result_img = result_img.resize((original_width, original_height), Image.LANCZOS)
            result_img.save(output_path)
        
        print(f"이미지 변환 완료: {input_path} -> {output_path}")
        return True
        
    except Exception as e:
        print(f"이미지 처리 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 현재 디렉토리의 파일 사용
    input_path = "test.jpg"  # 현재 디렉토리의 테스트 이미지
    output_path = "test_night.jpg"  # 변환된 이미지 저장 경로
    model_path = "latest_net_G_A.pth"  # 현재 디렉토리의 모델 파일
    
    # 변환 실행 (원본 해상도 유지)
    success = day_to_night(
        input_path, 
        output_path, 
        model_path, 
        keep_original_resolution=True,  # 원본 해상도 유지
        patch_size=256,  # 패치 크기
        overlap=32,      # 패치 간 겹침 영역
        strict=False     # 모델 구조 불일치 허용
    )
    
    if success:
        print("변환이 완료되었습니다!")
        print(f"결과 이미지: {os.path.abspath(output_path)}")
    else:
        print("변환에 실패했습니다.")