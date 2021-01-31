# PRDL #1

## Implementation of training tokenizer

huggingface의 tokenizers를 기반으로 학습하되, 전체 말뭉치를 대상으로 sampling 기능을 Shell script로 구현하여 사용해야함.

# requirements

1. train_tokenizer.py에 parameter를 전달해서 CBPE, BBPE, wordpiece를 학습할 수 있어야한다.
    1. python train_tokenizer.py 또는 run.sh (권장)
2. train_tokenizer.py에 특정 문서를 parameter로 sample_rate를 전달하면, 해당 문서에 라인을 sample해서 새로운 파일을 쓴다. 그리고 그 파일을 기반으로 학습한다.
3. morpheme-aware 학습이 가능해야한다. sample 파일 multi-processing 새로 써야한다.
    1. MeCab
4. 해당 app이 동작하는 환경을 도커 컨테이너로 구성한다. (Dockerfile)

## 간단 설명

---

### Handler.py

- huggingface tokenizer를 이용해서 cbpe, bbpe, wordpiece를 학습시키는 클래스
- 학습된 vocab은 동일폴더 json 형태로 저장됨

### Sampler.py

- 큰 사이즈의 코퍼스를 다루려면 많은 computational cost가 들기 때문에 일정 비율로 sampling하여 학습할 필요가 있음
- 주어진 비율로 파일을 샘플링하여 새로운 파일을 생성함

### Trainer.py

- args를 받아서 tokenizer를 학습함
- tokenizer는 bbpe, cbpe, wordpiece 3가지 중 하나를 선택할 수 있음
- sample_rate가 args로 주어지면 Sampler를 통해서 원본파일을 sampling 한 후에 학습하며, 주어지지 않은 경우에는 corpus 폴더 전체의 파일을 학습함

### TBD

- morpheme-aware를 위한 mecab을 적용해야함
- 완성된 버전을 dockerfile로 작성