import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import traceback

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

def diagnose_single_image(image_path, output_path, model_path, verbose=True):
    """
    단일 이미지에 대한 변환을 시도하고 상세한 진단 정보 제공
    
    Args:
        image_path: 입력 이미지 경로
        output_path: 출력 이미지 경로
        model_path: 모델 파일 경로
        verbose: 상세 로그 출력 여부
    """
    try:
        # 1. 이미지 로드 테스트
        if verbose:
            print(f"1. 이미지 로드 테스트...")
        try:
            img = Image.open(image_path)
            if verbose:
                print(f"   이미지 크기: {img.size}")
                print(f"   이미지 모드: {img.mode}")
            img = img.convert('RGB')
            print(f"   [성공] 이미지 로드 및 RGB 변환")
        except Exception as e:
            print(f"   [실패] 이미지 로드 오류: {e}")
            return False
        
        # 2. 디바이스 설정 테스트
        if verbose:
            print(f"\n2. 디바이스 설정 테스트...")
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"   사용 디바이스: {device}")
            if device.type == 'cuda':
                print(f"   CUDA 버전: {torch.version.cuda}")
                print(f"   GPU: {torch.cuda.get_device_name(0)}")
                print(f"   메모리 할당: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
                print(f"   메모리 캐시: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                print(f"   사용 가능 메모리: {free_memory / 1024**2:.2f} MB")
            print(f"   [성공] 디바이스 설정")
        except Exception as e:
            print(f"   [실패] 디바이스 설정 오류: {e}")
            return False
        
        # 3. 모델 로드 테스트
        if verbose:
            print(f"\n3. 모델 로드 테스트...")
        try:
            G_day2night = ResnetGenerator().to(device)
            checkpoint = torch.load(model_path, map_location=device)
            
            # 다양한 키 이름으로 시도
            try:
                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint:
                        G_day2night.load_state_dict(checkpoint['state_dict'], strict=False)
                        print(f"   [성공] 'state_dict' 키로 모델 로드")
                    elif 'net_G_A' in checkpoint:
                        G_day2night.load_state_dict(checkpoint['net_G_A'], strict=False)
                        print(f"   [성공] 'net_G_A' 키로 모델 로드")
                    elif 'G_A' in checkpoint:
                        G_day2night.load_state_dict(checkpoint['G_A'], strict=False)
                        print(f"   [성공] 'G_A' 키로 모델 로드")
                    else:
                        # 키 목록 출력
                        if verbose and isinstance(checkpoint, dict):
                            print(f"   체크포인트 키: {list(checkpoint.keys())}")
                        G_day2night.load_state_dict(checkpoint, strict=False)
                        print(f"   [성공] 직접 state_dict로 모델 로드")
                else:
                    G_day2night.load_state_dict(checkpoint, strict=False)
                    print(f"   [성공] 직접 state_dict로 모델 로드")
            except Exception as inner_e:
                print(f"   [경고] 첫 번째 로드 시도 실패: {inner_e}")
                if isinstance(checkpoint, dict):
                    # 키 목록 출력
                    print(f"   체크포인트 키: {list(checkpoint.keys())}")
                print(f"   strict=False로 재시도 중...")
                G_day2night.load_state_dict(checkpoint, strict=False)
                print(f"   [성공] strict=False로 모델 로드")
            
            G_day2night.eval()
            print(f"   모델을 평가 모드로 설정")
            print(f"   [성공] 모델 로드 및 설정 완료")
        except Exception as e:
            print(f"   [실패] 모델 로드 오류: {e}")
            traceback.print_exc()
            return False
        
        # 4. 이미지 변환 테스트
        if verbose:
            print(f"\n4. 이미지 변환 테스트...")
        try:
            # 원본 이미지 크기 저장
            original_size = img.size
            
            # 이미지 변환을 위한 전처리
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            if verbose:
                print(f"   이미지 텐서 크기: {img_tensor.shape}")
                print(f"   메모리 할당: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            
            # 모델을 사용하여 이미지 변환
            with torch.no_grad():
                if verbose:
                    print(f"   이미지 변환 중...")
                fake_night = G_day2night(img_tensor)
                if verbose:
                    print(f"   변환된 텐서 크기: {fake_night.shape}")
            
            # 변환된 이미지를 원래 크기로 변환
            result = fake_night.squeeze(0).cpu() * 0.5 + 0.5
            result_img = transforms.ToPILImage()(result)
            result_img = result_img.resize(original_size, Image.LANCZOS)
            
            # 결과 이미지 저장
            result_img.save(output_path)
            print(f"   [성공] 이미지 변환 및 저장 완료: {output_path}")
        except Exception as e:
            print(f"   [실패] 이미지 변환 오류: {e}")
            traceback.print_exc()
            return False
            
        # 모든 과정 성공
        print(f"\n모든 진단 테스트가 성공적으로 완료되었습니다!")
        print(f"변환된 이미지: {output_path}")
        return True
        
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 진단을 위한 설정
    # 첫 번째 이미지만 테스트
    input_folder = r"C:\Users\kkjoo\Downloads\Fashion\Training"
    model_path = "latest_net_G_A.pth"  # 모델 파일 경로
    
    # 테스트할 첫 번째 이미지 찾기
    print("이미지 찾는 중...")
    sample_image = None
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                sample_image = os.path.join(root, file)
                break
        if sample_image:
            break
    
    if not sample_image:
        print(f"테스트할 이미지를 찾을 수 없습니다: {input_folder}")
    else:
        print(f"테스트할 이미지: {sample_image}")
        output_path = f"{sample_image}_night_test.jpg"
        
        # 단일 이미지에 대한 자세한 진단 실행
        success = diagnose_single_image(sample_image, output_path, model_path, verbose=True)
        
        if success:
            print("\n진단이 성공적으로 완료됐습니다. 단일 이미지는 정상적으로 변환됐습니다.")
            print("대량 처리 중 실패 이유를 찾으려면 실패 로그 파일을 확인하세요.")
            print("실패 로그 파일: failed_images.txt")
        else:
            print("\n진단이 실패했습니다. 문제를 해결한 후 다시 시도하세요.")